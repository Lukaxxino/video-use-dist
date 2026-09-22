import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import uvicorn

# Reduce CUDA fragmentation on low-VRAM GPUs (e.g. 6 GB RTX 2060). Set here
# rather than passed explicitly to the diarization subprocess below, since a
# child process without an explicit env= already inherits this environ.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

app = FastAPI(title="NeMo Canary + Sortformer ASR API")

# Copied from nemo_server.py (same ASRModel.from_pretrained loading contract,
# same windowing/diarization-subprocess architecture) with the ASR model
# swapped to Canary-1b-v2 for a head-to-head comparison against Parakeet-TDT
# on Czech content — see the ASR comparison report. Canary is an attention
# encoder-decoder (AED) model, not RNNT/TDT, and its .transcribe() call needs
# source_lang/target_lang (ASR, not translation, so both are LANGUAGE).
MODEL_NAME = "nvidia/canary-1b-v2"
# diar_streaming_sortformer_4spk-v2 is CC-BY-4.0 (commercial use OK) and uses
# the identical NeMo SortformerEncLabelModel.from_pretrained() API as the
# older diar_sortformer_4spk-v1 (CC-BY-NC-4.0), so this is a drop-in swap for
# nemo_diarize_worker.py. Matches the tuned GH200 devcontainer
# (grace_hopper/asr_server.py, grace_hopper/diarize_server.py); the editor
# workstations run in a commercial context, so v1's non-commercial licence is
# not usable there.
DIAR_MODEL_NAME = "nvidia/diar_streaming_sortformer_4spk-v2"
LANGUAGE = "cs"

# Per-window language ID, run before the ASR pass (see
# _detect_languages_in_subprocess): Canary hardcodes source_lang/target_lang
# per call, so without this every window is forced through `LANGUAGE`'s
# decoding even when it contains a different spoken language. langid_ambernet
# (VoxLingua107-based, loaded via EncDecSpeakerLabelModel, NGC model name
# "langid_ambernet" — verified interactively, see nemo_langid_worker.py's
# docstring) reports one of 107 ISO 639-1-style codes per window; only the
# ones Canary-1b-v2 itself supports (its HF model card's "Supported
# Languages" list) are usable as source_lang/target_lang, so anything else —
# including confident, correct LID hits on a genuinely unsupported language
# like Turkish ("tr") — still falls back to `effective_language`.
#
# A whole-window LID vote (one call per ~75s window) does NOT catch a short
# foreign-language passage (a song, a few seconds of ambient speech) embedded
# inside an otherwise-`effective_language` window: a ~10-40s non-Czech
# passage surrounded by Czech speech is outvoted and the window still
# reports "cs" at >0.98 confidence (confirmed directly on this pipeline's two
# known trouble windows — a Turkish song and an English ambient-speech
# passage). LID *does* fire correctly (>0.85 confidence on the true
# language) on 5-8s sub-clips isolating just the foreign passage, so
# language routing here runs at SUBWINDOW_LID_SEC granularity instead: each
# ~75s window is sliced into SUBWINDOW_LID_SEC-long pieces, each piece gets
# its own LID call, and pieces are grouped back into contiguous same-language
# runs (see _resolve_window_runs) before the ASR pass. A window that's
# entirely one language collapses to a single run — one ASR call, same cost
# as before this existed. Only a window that genuinely contains a language
# switch pays for more than one ASR call.
LID_MODEL_NAME = "langid_ambernet"

# VAD/music gate, run per sub-window (same SUBWINDOW_LID_SEC granularity and
# same sliced audio files as LID, see below) before LID or ASR ever see that
# audio. Two purpose-built classifiers run together, not as an either/or
# switch — see nemo_vad_worker.py's own docstring for the full rationale and
# the real-audio verification (2026-08-18) behind the relative music/speech
# dominance test:
#   - MarbleNet (nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0): the PRIMARY
#     silence gate. A purpose-built binary speech/non-speech frame
#     classifier — simpler and more reliable for that one narrow job than
#     asking a broad multi-class tagger to do it. A sub-window with no
#     MarbleNet-detected speech skips LID+ASR entirely.
#   - PANNs (Cnn14 AudioSet-527 tagging, via panns_inference): checked only
#     on sub-windows MarbleNet already flagged as containing speech, for the
#     one thing MarbleNet cannot detect at all — music/singing dominance.
#     Catches the Turkish-song case (langid_ambernet is trained on spoken
#     language, not sung vocals, and reliably misclassifies it) without also
#     catching ordinary dialogue that merely has a quiet background music
#     bed (common in produced/broadcast audio) — verified against this
#     project's own real test-video audio, not just the docstring's claim.
VAD_MODEL_NAME = "nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0"
MUSIC_GATE_MODEL_NAME = "panns_cnn14"
CANARY_SUPPORTED_LANGUAGES = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru",
    "uk",
}
# Asymmetric (hysteresis) LID thresholds, matching the tuned GH200
# devcontainer (grace_hopper/asr_server.py). langid_ambernet is very
# decisive on clean single-language audio (>0.99 on a clean Czech clip), but
# a misroute forces Canary's source_lang/target_lang for a whole run, so an
# unearned switch *away* from the effective/default language costs far more
# than an unearned stay. It is therefore cheap to stay in / return to the
# default language (STAY), but leaving it needs notably higher confidence
# (SWITCH). No universal langid_ambernet threshold is documented; tuned
# empirically.
LID_CONFIDENCE_THRESHOLD_STAY = 0.5
LID_CONFIDENCE_THRESHOLD_SWITCH = 0.75

# Canary has its own built-in dynamic chunking for long-form audio (1s
# overlap, auto-enabled by its own .transcribe() on a single file per its
# model card), but this session's own windowing is kept anyway: it's already
# proven not to lose accuracy (see nemo_server.py's docstrings) and keeps
# both backends on identical audio chunks for a fair comparison.
WINDOW_SEC = 75.0
OVERLAP_SEC = 10.0

# Sub-window LID granularity (see LID_MODEL_NAME's docstring above). Raised
# from 2.0s to match the tuned GH200 devcontainer: spoken-language-ID
# accuracy degrades sharply below ~2s (Valente et al., Interspeech 2024),
# and 2s sub-windows were classifying short stretches of real Czech speech
# as Russian/Greek/Spanish and hard-forcing Canary to decode them wrong.
SUBWINDOW_LID_SEC = 4.0
MIN_FOREIGN_RUN_SEC = 2.0
# A single misclassified 4s sub-window can clear MIN_FOREIGN_RUN_SEC on its
# own, so a non-effective-language run must also span at least this many
# consecutive sub-windows to survive (a crude majority-vote smoothing
# stand-in; real posterior smoothing would need langid_ambernet's raw
# per-language scores, which nemo_langid_worker.py does not expose).
MIN_FOREIGN_RUN_SUBWINDOWS = 2
# Canary repetition-loop hallucinations are tied to long (~180s) inputs
# (NVIDIA-NeMo issues #9030, Speech discussion #8776); cap any single
# transcribe() call well under that even when the language never switches.
MAX_RUN_SEC = 35.0

_asr_model = None


def _compute_windows(
    duration_sec: float, window_sec: float = WINDOW_SEC, overlap_sec: float = OVERLAP_SEC
) -> list[tuple[float, float]]:
    """Splits [0, duration_sec) into overlapping (start, end) windows. A file
    no longer than one window is returned as a single window covering it."""
    if duration_sec <= window_sec:
        return [(0.0, duration_sec)]
    hop = window_sec - overlap_sec
    windows = []
    start = 0.0
    while True:
        end = min(start + window_sec, duration_sec)
        windows.append((start, end))
        if end >= duration_sec:
            break
        start += hop
    return windows


def _window_keep_ranges(
    windows: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """For each window, the [lower, upper) span of global time it is
    authoritative for — split from its neighbors at the midpoint of their
    shared overlap, so every instant is covered by exactly one window."""
    n = len(windows)
    keep_ranges = []
    for i, (start, end) in enumerate(windows):
        lower = 0.0 if i == 0 else (windows[i][0] + windows[i - 1][1]) / 2.0
        upper = end if i == n - 1 else (windows[i + 1][0] + windows[i][1]) / 2.0
        keep_ranges.append((lower, upper))
    return keep_ranges


def _compute_subwindows(
    duration_sec: float,
    subwindow_sec: float = SUBWINDOW_LID_SEC,
    min_tail_sec: float = 2.0,
) -> list[tuple[float, float]]:
    """Splits a single window's [0, duration_sec) into non-overlapping
    subwindow_sec-long pieces for LID (see LID_MODEL_NAME's docstring). A
    trailing remainder shorter than min_tail_sec is folded into the previous
    piece instead of becoming its own too-short-to-classify-reliably piece."""
    if duration_sec <= subwindow_sec:
        return [(0.0, duration_sec)]
    subs = []
    start = 0.0
    while start < duration_sec:
        end = start + subwindow_sec
        if end >= duration_sec or duration_sec - end < min_tail_sec:
            # Last piece, or the next piece would be a too-short remainder:
            # extend this piece to cover the rest instead of leaving a tiny
            # trailing piece of its own.
            subs.append((start, duration_sec))
            break
        subs.append((start, end))
        start = end
    return subs


def _route_lang(lid_result: dict, effective_language: str) -> str:
    """Resolves one LID result to the language an ASR call should actually
    use: the detected language, only if it's both one of Canary's supported
    languages and confident enough; `effective_language` otherwise (an
    unsupported-but-correct hit, e.g. "tr" for Turkish, or a low-confidence/
    ambiguous result)."""
    detected_lang = lid_result.get("language")
    confidence = lid_result.get("confidence", 0.0)
    threshold = (
        LID_CONFIDENCE_THRESHOLD_STAY
        if detected_lang == effective_language
        else LID_CONFIDENCE_THRESHOLD_SWITCH
    )
    if detected_lang in CANARY_SUPPORTED_LANGUAGES and confidence >= threshold:
        return detected_lang
    return effective_language


def _resolve_window_runs(
    subwindow_langs: list[tuple[float, float, str]],
    effective_language: str,
    min_foreign_run_sec: float = MIN_FOREIGN_RUN_SEC,
    min_foreign_run_subwindows: int = MIN_FOREIGN_RUN_SUBWINDOWS,
) -> list[tuple[float, float, str]]:
    """Groups a window's per-subwindow routed languages (window-local start,
    end, language) into contiguous same-language runs. A candidate non-
    effective_language run is downgraded back to effective_language before
    the final merge when it is either shorter than min_foreign_run_sec OR
    spans fewer than min_foreign_run_subwindows consecutive sub-windows (see
    MIN_FOREIGN_RUN_SEC / MIN_FOREIGN_RUN_SUBWINDOWS), so isolated LID noise
    doesn't fragment a window into extra ASR calls. A window whose subwindows
    all agree collapses to one run.

    `subwindow_langs` no longer has to cover the window's full [0, duration)
    span contiguously: the VAD/music gate (see VAD_MODEL_NAME's docstring
    above) drops silence- and music-dominant sub-windows before this is
    called, so real time gaps between kept sub-windows are expected, not a
    bug. `_merge_adjacent` only merges two entries that are both the same
    language AND touch in time (`prev_end == start`) — adjacency in the
    input list alone is not enough — so a gap correctly starts a new run
    instead of silently bridging over the dropped audio. An empty input
    (every sub-window in this window was gated out) returns an empty list,
    and the caller already treats zero runs as "no ASR call for this
    window", not an error."""

    def _merge_adjacent(spans):
        # Carries a count of how many input sub-windows folded into each run,
        # so the downgrade step can also reject a foreign run that spans too
        # few consecutive sub-windows. Only merges entries that are the same
        # language AND touch in time (`prev_end == start`) — a gap left by a
        # gated-out sub-window correctly starts a new run.
        merged = []
        for start, end, lang in spans:
            if merged and merged[-1][2] == lang and merged[-1][1] == start:
                prev_start, _, _, prev_count = merged[-1]
                merged[-1] = (prev_start, end, lang, prev_count + 1)
            else:
                merged.append((start, end, lang, 1))
        return merged

    runs = _merge_adjacent(subwindow_langs)
    downgraded = [
        (start, end, effective_language)
        if lang != effective_language
        and ((end - start) < min_foreign_run_sec or count < min_foreign_run_subwindows)
        else (start, end, lang)
        for start, end, lang, count in runs
    ]
    return [(s, e, l) for s, e, l, _ in _merge_adjacent(downgraded)]


def _split_long_runs(
    runs: list[tuple[float, float, str]], max_run_sec: float = MAX_RUN_SEC
) -> list[tuple[float, float, str]]:
    """Splits any run longer than max_run_sec into consecutive same-language
    sub-runs capped at that length. Canary's repetition-loop hallucination
    risk grows with input length, so no single transcribe() call should span
    a very long stretch even when the language never switches (see
    MAX_RUN_SEC). Matches the tuned GH200 devcontainer."""
    split_runs = []
    for start, end, lang in runs:
        run_start = start
        while end - run_start > max_run_sec:
            run_end = run_start + max_run_sec
            split_runs.append((run_start, run_end, lang))
            run_start = run_end
        split_runs.append((run_start, end, lang))
    return split_runs


def _merge_windowed_words(
    per_window_words: list[list[dict]], windows: list[tuple[float, float]]
) -> list[dict]:
    """Offsets each window's word timestamps to global time and keeps only
    the words that fall in that window's authoritative range, so a word
    seen by two overlapping windows is counted exactly once."""
    keep_ranges = _window_keep_ranges(windows)
    merged = []
    for (win_start, _win_end), (lower, upper), words in zip(
        windows, keep_ranges, per_window_words
    ):
        for w in words:
            global_start = w["start"] + win_start
            global_end = w["end"] + win_start
            if lower <= global_start < upper:
                merged.append(
                    {"word": w["word"], "start": global_start, "end": global_end}
                )
    merged.sort(key=lambda w: w["start"])
    return merged


def _merge_windowed_audio_events(
    per_window_gate: list, windows: list
) -> list:
    """Clips each window's non-speech (skip_silence/skip_music) gate spans to
    that window's authoritative keep-range — same overlap-dedup pattern as
    _merge_windowed_diarization, so a span seen by two overlapping windows is
    counted once — then merges adjacent same-type spans."""
    keep_ranges = _window_keep_ranges(windows)
    merged = []
    for i, events in enumerate(per_window_gate):
        win_start = windows[i][0]
        lower_keep, upper_keep = keep_ranges[i]
        for ev in events:
            global_start = ev["start"] + win_start
            global_end = ev["end"] + win_start
            overlap_start = max(global_start, lower_keep)
            overlap_end = min(global_end, upper_keep)
            if overlap_start < overlap_end:
                merged.append({
                    "start": round(overlap_start, 3),
                    "end": round(overlap_end, 3),
                    "type": ev["type"],
                    "music_probability": ev["music_probability"],
                })

    if not merged:
        return []

    merged.sort(key=lambda e: e["start"])
    stitched = [merged[0]]
    for curr in merged[1:]:
        prev = stitched[-1]
        if curr["type"] == prev["type"] and curr["start"] <= prev["end"] + 0.05:
            prev["end"] = max(prev["end"], curr["end"])
        else:
            stitched.append(curr)
    return stitched


def _merge_windowed_diarization(
    per_window_segments: list[list[dict]], windows: list[tuple[float, float]]
) -> list[dict]:
    """Offsets each window's diarization segments to global time, then
    resolves per-window-local speaker labels (NeMo assigns these fresh on
    every diarize() call) to a stable global roster by voting on which
    global speaker overlaps most in the shared audio between consecutive
    windows. Falls back to a new global speaker if no overlap is found
    (e.g. that speaker was silent during the shared audio)."""
    keep_ranges = _window_keep_ranges(windows)
    merged = []
    next_global_id = 0
    prev_global_segments = None
    prev_window_end = None

    for (win_start, win_end), (lower, upper), segs in zip(
        windows, keep_ranges, per_window_segments
    ):
        global_segs = [
            {"start": s["start"] + win_start, "end": s["end"] + win_start, "speaker": s["speaker"]}
            for s in segs
        ]

        if prev_global_segments is None:
            label_map = {}
            for lbl in sorted({s["speaker"] for s in global_segs}):
                label_map[lbl] = f"speaker_{next_global_id}"
                next_global_id += 1
        else:
            overlap_start, overlap_end = win_start, prev_window_end
            label_map = {}
            for lbl in sorted({s["speaker"] for s in global_segs}):
                overlaps: dict[str, float] = {}
                for s in global_segs:
                    if s["speaker"] != lbl:
                        continue
                    seg_start = max(s["start"], overlap_start)
                    seg_end = min(s["end"], overlap_end)
                    if seg_end <= seg_start:
                        continue
                    for ps in prev_global_segments:
                        dur = min(ps["end"], seg_end) - max(ps["start"], seg_start)
                        if dur > 0:
                            overlaps[ps["speaker"]] = overlaps.get(ps["speaker"], 0.0) + dur
                if overlaps:
                    label_map[lbl] = max(overlaps, key=overlaps.get)
                else:
                    label_map[lbl] = f"speaker_{next_global_id}"
                    next_global_id += 1

        relabeled = [
            {"start": s["start"], "end": s["end"], "speaker": label_map[s["speaker"]]}
            for s in global_segs
        ]
        for s in relabeled:
            clip_start = max(s["start"], lower)
            clip_end = min(s["end"], upper)
            if clip_end > clip_start:
                merged.append({"start": clip_start, "end": clip_end, "speaker": s["speaker"]})

        prev_global_segments = relabeled
        prev_window_end = win_end

    merged.sort(key=lambda s: s["start"])
    return merged


def _import_nemo_asr_classes():
    from nemo.collections.asr.models import EncDecMultiTaskModel
    return EncDecMultiTaskModel


def _load_asr_model(asr_model_cls) -> None:
    """Loads and caches the ASR model on first request so it stays warm
    across requests. Diarization deliberately has no equivalent warm,
    in-process model here — see _diarize_windows_in_subprocess and
    nemo_server.py's matching docstring: alternating ASR/diarizer calls in
    one process hard-crashes it on this torch/nemo_toolkit/Windows stack
    (reproduced and isolated separately, applies regardless of which ASR
    model is loaded), so diarization always runs in a fresh child process."""
    global _asr_model
    if _asr_model is not None:
        return
    print(f"[Canary Server] Loading ASR model {MODEL_NAME} ...")
    _asr_model = asr_model_cls.from_pretrained(model_name=MODEL_NAME)


def _diarize_windows_in_subprocess(
    window_paths: list, diar_model_name: str, tmpdir: Path
) -> list:
    """Diarizes each window audio file in a throwaway child process — see
    _load_asr_model's docstring for why this must never run in-process
    alongside the ASR model. Reuses the same worker nemo_server.py uses;
    Canary itself has no built-in diarization."""
    output_path = tmpdir / "diarization_output.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "nemo_diarize_worker.py"),
        "--diar-model-name",
        diar_model_name,
        "--output",
        str(output_path),
    ]
    for window_path in window_paths:
        cmd += ["--window", str(window_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "nemo_diarize_worker subprocess failed "
            f"(exit {result.returncode}):\n{result.stderr[-2000:]}"
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _detect_languages_in_subprocess(
    audio_paths: list, langid_model_name: str, tmpdir: Path
) -> list:
    """Runs language ID on each given audio file (subwindow-length pieces,
    per LID_MODEL_NAME's docstring, not whole ~75s windows) in a throwaway
    child process, before the ASR model is ever loaded in this parent
    process — see _load_asr_model's docstring for why a second NeMo model's
    inference calls must never alternate with the warm ASR model's in the
    same process. Must be called (and must return) before _load_asr_model /
    the ASR .transcribe() loop, exactly like _diarize_windows_in_subprocess
    must run after them: LID subprocess -> ASR in-process -> diarization
    subprocess, three sequential passes, never interleaved.

    Returns a list of {"language": <code>, "confidence": <float>} dicts, one
    per file, in the same order as audio_paths."""
    output_path = tmpdir / "langid_output.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "nemo_langid_worker.py"),
        "--langid-model-name",
        langid_model_name,
        "--output",
        str(output_path),
    ]
    for audio_path in audio_paths:
        cmd += ["--window", str(audio_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "nemo_langid_worker subprocess failed "
            f"(exit {result.returncode}):\n{result.stderr[-2000:]}"
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _gate_subwindows_in_subprocess(audio_paths: list, tmpdir: Path) -> list:
    """Runs the combined MarbleNet + PANNs VAD/music gate (see
    VAD_MODEL_NAME's docstring above and nemo_vad_worker.py) on each given
    sub-window audio file, in its own throwaway child process for the same
    reason _detect_languages_in_subprocess and
    _diarize_windows_in_subprocess do — must complete (and exit) before
    _load_asr_model / the ASR .transcribe() loop, and runs before the LID
    subprocess too, since its whole purpose is deciding which sub-windows
    LID is even allowed to see.

    Returns a list of {"has_speech", "speech_fraction", "music_probability",
    "speech_probability", "action"} dicts, one per file, in the same order
    as audio_paths. "action" is one of "process", "skip_silence", or
    "skip_music"."""
    output_path = tmpdir / "vad_gate_output.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "nemo_vad_worker.py"),
        "--output",
        str(output_path),
    ]
    for audio_path in audio_paths:
        cmd += ["--window", str(audio_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "nemo_vad_worker subprocess failed "
            f"(exit {result.returncode}):\n{result.stderr[-2000:]}"
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _mock_response() -> dict:
    return {
        "text": "Tohle je jen testovaci mock text, protoze nemo_toolkit neni nainstalovany.",
        "words": [
            {"word": "Tohle", "start": 0.0, "end": 0.4},
            {"word": "je", "start": 0.4, "end": 0.6},
            {"word": "jen", "start": 0.6, "end": 0.9},
            {"word": "testovaci", "start": 0.9, "end": 1.5},
            {"word": "mock", "start": 1.5, "end": 1.8},
            {"word": "text.", "start": 1.8, "end": 2.2},
        ],
        "speaker_segments": [{"start": 0.0, "end": 2.2, "speaker": "speaker_0"}],
        "audio_events": [],
        # Lets a caller (now or later) tell a real transcript apart from this
        # mock fallback instead of the two being indistinguishable on disk.
        "mock": True,
    }


def _parse_diar_segments(raw_segments: list) -> list:
    """Parses SortformerEncLabelModel.diarize()'s per-segment strings into
    {"start", "end", "speaker"} dicts. NeMo documents the format as
    "begin_seconds, end_seconds, speaker_index" (nvidia/diar_sortformer_4spk-v1
    model card); verify the exact separator against the installed
    nemo_toolkit version and adjust the split() below if it differs."""
    parsed = []
    for line in raw_segments:
        parts = str(line).replace(",", " ").split()
        start, end, speaker = float(parts[0]), float(parts[1]), parts[2]
        speaker_label = (
            speaker if speaker.startswith("speaker_") else f"speaker_{speaker}"
        )
        parsed.append({"start": start, "end": end, "speaker": speaker_label})
    return parsed


@app.get("/health")
async def health():
    """Reports this server's configured backend/model and whether
    nemo_toolkit is actually importable -- only the top-level package
    import (see `_import_nemo_asr_classes`'s docstring), never the model
    construction in `_load_asr_model`, so this never loads a second model
    just to answer a health check. `ready=False` is the only way a caller
    can tell this server apart from one that will silently fall back to the
    mock transcript below."""
    try:
        _import_nemo_asr_classes()
        ready = True
    except ImportError:
        ready = False
    return {
        "backend": "canary",
        "model": MODEL_NAME,
        "diarization_model": DIAR_MODEL_NAME,
        "language_id_model": LID_MODEL_NAME,
        "vad_model": VAD_MODEL_NAME,
        "music_gate_model": MUSIC_GATE_MODEL_NAME,
        "ready": ready,
    }


@app.post("/v1/transcribe")
async def transcribe_audio(file: UploadFile = File(None), language: str = Form(None)):
    """Receives an audio file, runs NeMo Canary + Sortformer on it, and
    returns raw {"text", "words", "speaker_segments"} JSON — the same
    contract nemo_server.py produces, so helpers/process_nemo_output.py
    needs no changes to consume either backend's output.

    language: Canary is a multitask model that needs an explicit
    source_lang/target_lang pair — unlike Parakeet-TDT, which auto-detects
    language and only takes it as an informational hint. `language` (falling
    back to `LANGUAGE`/`effective_language`) is the fallback for any span of
    audio whose sub-window-level detected language (see
    _detect_languages_in_subprocess, _route_lang, _resolve_window_runs) isn't
    both one of Canary's 25 supported languages and confident enough; spans
    that do meet that bar are decoded in their own detected language instead.
    """
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    effective_language = LANGUAGE if language is None else language

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / file.filename
        with open(audio_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            asr_model_cls = _import_nemo_asr_classes()
        except ImportError:
            print("[Canary Server] nemo_toolkit not installed. Returning MOCK data.")
            return _mock_response()

        _load_asr_model(asr_model_cls)

        info = sf.info(str(audio_path))
        duration_sec = info.frames / info.samplerate
        windows = _compute_windows(duration_sec)

        with tempfile.TemporaryDirectory() as window_tmpdir:
            window_tmpdir_path = Path(window_tmpdir)
            window_paths = []
            window_durations = []
            for i, (win_start, win_end) in enumerate(windows):
                window_durations.append(win_end - win_start)
                if len(windows) == 1:
                    window_paths.append(audio_path)
                    continue
                start_frame = int(win_start * info.samplerate)
                stop_frame = int(win_end * info.samplerate)
                chunk, sr = sf.read(
                    str(audio_path), start=start_frame, stop=stop_frame, dtype="float32"
                )
                window_path = window_tmpdir_path / f"window_{i:03d}.wav"
                sf.write(str(window_path), chunk, sr)
                window_paths.append(window_path)

            # Pass 1 of 3 (subprocess, before the ASR model is loaded/called
            # in this process): sub-window language ID — see LID_MODEL_NAME's
            # docstring for why this runs at SUBWINDOW_LID_SEC granularity
            # rather than once per whole ~75s window. Each window's audio is
            # sliced into pieces, every piece gets its own LID call in one
            # subprocess batch, and results are grouped back per window
            # below via _resolve_window_runs.
            subwindow_specs = []  # (window_idx, sub_start, sub_end), window-local time
            subwindow_paths = []
            for i, (window_path, duration) in enumerate(zip(window_paths, window_durations)):
                subs = _compute_subwindows(duration)
                if len(subs) == 1:
                    subwindow_specs.append((i, subs[0][0], subs[0][1]))
                    subwindow_paths.append(window_path)
                    continue
                window_audio, window_sr = sf.read(str(window_path), dtype="float32")
                for j, (sub_start, sub_end) in enumerate(subs):
                    sub_path = window_tmpdir_path / f"window_{i:03d}_lid_{j:03d}.wav"
                    sf.write(
                        str(sub_path),
                        window_audio[int(sub_start * window_sr):int(sub_end * window_sr)],
                        window_sr,
                    )
                    subwindow_specs.append((i, sub_start, sub_end))
                    subwindow_paths.append(sub_path)

            # Pass 0 of 3 (subprocess, before LID and before the ASR model is
            # loaded/called): the combined MarbleNet + PANNs VAD/music gate
            # (see VAD_MODEL_NAME's docstring above) runs on every sub-window
            # LID would otherwise see. Sub-windows it marks "skip_silence" or
            # "skip_music" never reach LID or ASR at all — they're just gaps
            # in this window's runs (see _resolve_window_runs), not silently
            # routed to effective_language.
            gate_results = _gate_subwindows_in_subprocess(subwindow_paths, window_tmpdir_path)
            gated_specs = []
            gated_paths = []
            per_window_gate_events = [[] for _ in windows]
            for spec, path, gate in zip(subwindow_specs, subwindow_paths, gate_results):
                if gate["action"] == "process":
                    gated_specs.append(spec)
                    gated_paths.append(path)
                else:
                    win_idx, sub_start, sub_end = spec
                    per_window_gate_events[win_idx].append({
                        "start": sub_start,
                        "end": sub_end,
                        "type": "music" if gate["action"] == "skip_music" else "silence",
                        "music_probability": gate["music_probability"],
                    })

            subwindow_lid_results = (
                _detect_languages_in_subprocess(gated_paths, LID_MODEL_NAME, window_tmpdir_path)
                if gated_paths
                else []
            )

            per_window_subwindow_langs = [[] for _ in windows]
            for (win_idx, sub_start, sub_end), lid_result in zip(
                gated_specs, subwindow_lid_results
            ):
                per_window_subwindow_langs[win_idx].append(
                    (sub_start, sub_end, _route_lang(lid_result, effective_language))
                )

            # Pass 2 of 3 (in-process, warm ASR model): one ASR call per
            # contiguous same-language run in each window (see
            # _resolve_window_runs). A window with no language switch — the
            # common case — resolves to exactly one run, so it costs exactly
            # one ASR call, same as before sub-window LID existed.
            per_window_words = []
            for i, (window_path, duration) in enumerate(zip(window_paths, window_durations)):
                runs = _split_long_runs(
                    _resolve_window_runs(per_window_subwindow_langs[i], effective_language)
                )
                words_for_window = []
                window_audio = None
                for run_start, run_end, run_lang in runs:
                    if len(runs) == 1:
                        run_audio_path = window_path
                    else:
                        if window_audio is None:
                            window_audio, window_sr = sf.read(str(window_path), dtype="float32")
                        run_audio_path = window_tmpdir_path / f"window_{i:03d}_run_{int(run_start*1000):06d}.wav"
                        sf.write(
                            str(run_audio_path),
                            window_audio[int(run_start * window_sr):int(run_end * window_sr)],
                            window_sr,
                        )
                    try:
                        hyp = _asr_model.transcribe(
                            [str(run_audio_path)],
                            source_lang=run_lang,
                            target_lang=run_lang,
                            timestamps=True,
                        )[0]
                    except NotImplementedError:
                        hyp = _asr_model.transcribe(
                            [str(run_audio_path)],
                            source_lang=run_lang,
                            target_lang=run_lang,
                        )[0]
                    words_for_window.extend(
                        {
                            "word": w["word"],
                            "start": w["start"] + run_start,
                            "end": w["end"] + run_start,
                        }
                        for w in hyp.timestamp["word"]
                    )
                    # Root-caused and verified empirically: PyTorch's CUDA caching
                    # allocator does not release reserved-but-unused blocks between
                    # successive .transcribe() calls on this warm model instance, so
                    # `reserved` memory (not `allocated` — that stays flat) grows
                    # call over call until this 6 GB card is forced into slow
                    # Windows GPU-memory oversubscription (shared/system-memory
                    # paging), which is what actually made per-window latency climb
                    # (63s -> 207s -> 230s on identical-size windows, reproduced even
                    # with no LID/diarization involved). Explicitly dropping the
                    # hypothesis and clearing both Python's and CUDA's allocators
                    # after every ASR call (not just every window — a window can now
                    # be more than one call, see _resolve_window_runs) keeps
                    # `reserved` flat and keeps latency flat.
                    del hyp
                    gc.collect()
                    torch.cuda.empty_cache()
                per_window_words.append(words_for_window)

            # Pass 3 of 3 (subprocess, after the ASR pass above): diarize,
            # exactly as before this change — still never interleaved with
            # the ASR model.
            diar_raw_per_window = _diarize_windows_in_subprocess(
                window_paths, DIAR_MODEL_NAME, window_tmpdir_path
            )
            per_window_diar = [_parse_diar_segments(raw) for raw in diar_raw_per_window]

        words = _merge_windowed_words(per_window_words, windows)
        speaker_segments = _merge_windowed_diarization(per_window_diar, windows)
        audio_events = _merge_windowed_audio_events(per_window_gate_events, windows)
        text = " ".join(w["word"] for w in words if w["word"].strip())

        return {
            "text": text,
            "words": words,
            "speaker_segments": speaker_segments,
            "audio_events": audio_events,
        }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Local NeMo Canary ASR + diarization API server")
    ap.add_argument(
        "--model-name", default=MODEL_NAME,
        help="NeMo/HF ASR model id (e.g. nvidia/canary-1b-v2)",
    )
    ap.add_argument(
        "--diar-model-name", default=DIAR_MODEL_NAME,
        help="NeMo/HF diarization model id",
    )
    ap.add_argument(
        "--langid-model-name", default=LID_MODEL_NAME,
        help="NeMo LID model name (e.g. langid_ambernet), used per window before ASR",
    )
    ap.add_argument(
        "--lid-confidence-threshold", type=float, default=LID_CONFIDENCE_THRESHOLD_SWITCH,
        help="Minimum langid_ambernet confidence to SWITCH a window's language "
        "away from --language to a different detected language (default: 0.75). "
        "Returning to / staying in --language keeps the fixed 0.5 STAY floor.",
    )
    ap.add_argument(
        "--language", default=LANGUAGE,
        help="Fallback source_lang/target_lang for windows whose detected "
        "language isn't a confident, Canary-supported hit (default: cs)",
    )
    ap.add_argument("--port", type=int, default=8002, help="Port to listen on")
    args = ap.parse_args()

    MODEL_NAME = args.model_name
    DIAR_MODEL_NAME = args.diar_model_name
    LID_MODEL_NAME = args.langid_model_name
    LID_CONFIDENCE_THRESHOLD_SWITCH = args.lid_confidence_threshold
    LANGUAGE = args.language

    print(
        f"Starting local NeMo Canary API server on port {args.port} "
        f"(model={MODEL_NAME}, diarizer={DIAR_MODEL_NAME}, langid={LID_MODEL_NAME})..."
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port)
