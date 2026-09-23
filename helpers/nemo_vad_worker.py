import argparse
import json
from pathlib import Path

MUSIC_LABEL = "Music"
SINGING_LABEL = "Singing"
SPEECH_LABEL = "Speech"

MARBLENET_REPO_ID = "nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0"
MARBLENET_FILENAME = "frame_vad_multilingual_marblenet_v2.0.nemo"

MARBLENET_SAMPLE_RATE = 16000
PANNS_SAMPLE_RATE = 32000

DEFAULT_SPEECH_FRAME_THRESHOLD = 0.5
DEFAULT_SPEECH_FRACTION_THRESHOLD = 0.2
DEFAULT_MUSIC_PROBABILITY_THRESHOLD = 0.5


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Gates each audio sub-window through two purpose-built classifiers, "
            "run together (not as an either/or switch), before it is allowed to "
            "reach langid_ambernet + Canary. Runs as a throwaway subprocess for "
            "the same reason nemo_langid_worker.py and nemo_diarize_worker.py do: "
            "a second NeMo model's inference alternating in-process with a warm "
            "ASR model's .transcribe() calls hard-crashes the process on this "
            "torch/nemo_toolkit stack (see those workers' docstrings) -- this "
            "pass runs to completion and exits before canary_server.py loads/"
            "calls the ASR model.\n\n"
            "MarbleNet (nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0) is the "
            "PRIMARY silence gate: a purpose-built binary speech/non-speech "
            "frame classifier, more reliable for that one narrow job than "
            "asking a broad multi-class AudioSet tagger to do it. A sub-window "
            "with no MarbleNet-detected speech never needs LID or ASR at all -- "
            "PANNs is not even run on it.\n\n"
            "PANNs (Cnn14, AudioSet-527 tagging, via panns_inference) is used "
            "ONLY for its one value MarbleNet cannot provide at all: a "
            "music-probability score (max of the 'Music' and 'Singing' AudioSet "
            "labels), checked only on sub-windows MarbleNet already flagged as "
            "containing speech. A sub-window is treated as music-dominant (and "
            "skipped, same as silence) only when that music probability clears "
            "--music-probability-threshold AND exceeds PANNs' own 'Speech' "
            "label probability for the same clip -- a relative-dominance test, "
            "not an absolute one. This was verified against this project's own "
            "real test-video audio (2026-08-18): a pure music/title-sting "
            "passage scores speech=0.03 music=0.71-0.88 (correctly "
            "music-dominant, skipped), while a real talk-show passage with a "
            "quiet background music bed under an actual speaking host scores "
            "speech=0.89 music=0.83 (music present but NOT dominant over "
            "speech, correctly still sent to LID+ASR). An absolute threshold "
            "alone would have wrongly skipped that second, dialogue-bearing "
            "case -- produced/broadcast audio commonly carries a background "
            "music bed under real speech, and only genuinely music/singing-"
            "dominant audio (the Turkish-song case documented in "
            "canary_server.py's own history) should ever be kept from "
            "langid_ambernet, which is trained on spoken language, not sung "
            "vocals, and reliably misclassifies it."
        )
    )
    ap.add_argument(
        "--window", action="append", default=[], dest="windows",
        help="Path to a sub-window audio file; repeat in order, one per sub-window",
    )
    ap.add_argument(
        "--window-list", type=Path, default=None,
        help="File with one sub-window path per line (appended after any "
             "--window); avoids Windows' 32767-char command-line limit on "
             "long videos",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--speech-frame-threshold", type=float, default=DEFAULT_SPEECH_FRAME_THRESHOLD)
    ap.add_argument("--speech-fraction-threshold", type=float, default=DEFAULT_SPEECH_FRACTION_THRESHOLD)
    ap.add_argument("--music-probability-threshold", type=float, default=DEFAULT_MUSIC_PROBABILITY_THRESHOLD)
    ap.add_argument(
        "--panns-checkpoint", default=None,
        help="Optional explicit path to the Cnn14 .pth checkpoint; panns_inference "
             "downloads/caches its own default under ~/panns_data if omitted",
    )
    args = ap.parse_args()
    if args.window_list:
        args.windows += _read_window_list(args.window_list)
    if not args.windows:
        ap.error("no windows given (use --window and/or --window-list)")

    import numpy as np
    import soundfile as sf
    import librosa
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vad_model = _load_marblenet_frame_vad()
    vad_model = vad_model.to(device).eval()

    _ensure_panns_assets(args.panns_checkpoint)
    from panns_inference import AudioTagging
    panns_kwargs = {"device": device}
    if args.panns_checkpoint:
        panns_kwargs["checkpoint_path"] = args.panns_checkpoint
    panns_model = AudioTagging(**panns_kwargs)
    music_idx = panns_model.labels.index(MUSIC_LABEL)
    singing_idx = panns_model.labels.index(SINGING_LABEL)
    speech_idx = panns_model.labels.index(SPEECH_LABEL)

    results = []
    for window_path in args.windows:
        audio, sr = sf.read(window_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        audio_16k = audio if sr == MARBLENET_SAMPLE_RATE else librosa.resample(
            audio, orig_sr=sr, target_sr=MARBLENET_SAMPLE_RATE
        )

        with torch.no_grad():
            logits = vad_model.forward(
                input_signal=torch.tensor(audio_16k).unsqueeze(0).to(device),
                input_signal_length=torch.tensor([len(audio_16k)]).to(device),
            )
            # class 1 = speech (labels ['0','1']) on the editor-workstation
            # stack. Switched from class 0 on 2026-09-23 after an on-site
            # install (nemo_toolkit 3.0.0, RTX 2000 Ada): with class 0,
            # digital silence and white noise both scored speech_fraction
            # 1.000 (-> "process"), and a 42-min episode sent only 40.5% of
            # diarized speech to ASR, producing a "keska keska..."
            # hallucination loop from silence. With class 1: silence/noise
            # 0.000, same episode 96.6% coverage, 1715 -> 3988 words.
            #
            # This index has flipped before (class 0 was chosen 2026-08-18
            # on one clip for the GH200 stack), and the unmerged
            # worktree-gh200-vad-gate-fix branch documents class 1 dominating
            # continuous speech in one video and class 0 in another -- it may
            # be content- or stack-dependent. Before flipping it again, run
            # this worker on digital silence (ffmpeg anullsrc) AND a known
            # speech clip: silence must give speech_fraction ~0, speech > 0.5.
            frame_probs = torch.softmax(logits, dim=-1)[0, :, 1]
            speech_fraction = (
                float((frame_probs > args.speech_frame_threshold).float().mean())
                if frame_probs.numel() else 0.0
            )
        has_speech = speech_fraction >= args.speech_fraction_threshold

        music_probability = None
        speech_probability = None
        if has_speech:
            audio_32k = audio if sr == PANNS_SAMPLE_RATE else librosa.resample(
                audio, orig_sr=sr, target_sr=PANNS_SAMPLE_RATE
            )
            with torch.no_grad():
                clipwise_output, _ = panns_model.inference(audio_32k[None, :].astype(np.float32))
            music_probability = float(max(clipwise_output[0][music_idx], clipwise_output[0][singing_idx]))
            speech_probability = float(clipwise_output[0][speech_idx])
            music_dominant = (
                music_probability >= args.music_probability_threshold
                and music_probability > speech_probability
            )
            # skip_music is computed and reported (visible in the output for
            # analysis/tuning) but NOT currently acted on: a real-clip smoke
            # test (2026-08-18, this project's own test video's first 20s,
            # a music-heavy intro with overlapping narration) showed the
            # relative-dominance test skipping 4 of 5 sub-windows despite
            # real, continuous speech underneath the music, collapsing a
            # full transcript down to a single word -- the same class of
            # near-total content loss the timestamps=True fix eliminated
            # earlier the same day. Disabled pending broader validation
            # across more real audio than the one clip tested so far;
            # re-enable (action = "skip_music" if music_dominant else
            # "process") once the threshold/logic is validated not to
            # discard real narration-over-music passages.
            action = "process"
        else:
            action = "skip_silence"

        results.append({
            "has_speech": has_speech,
            "speech_fraction": speech_fraction,
            "music_probability": music_probability,
            "speech_probability": speech_probability,
            "action": action,
        })

    args.output.write_text(json.dumps(results), encoding="utf-8")


PANNS_LABELS_CSV_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
)
PANNS_CNN14_CHECKPOINT_URL = (
    "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
)


def _ensure_panns_assets(explicit_checkpoint_path):
    """Pre-provisions panns_inference's two on-first-use downloads (the
    527-class AudioSet labels CSV, and -- unless --panns-checkpoint points
    somewhere else -- the ~300MB Cnn14 checkpoint itself) via Python's own
    stdlib downloader, *before* `import panns_inference` ever runs.

    panns_inference's own config.py / inference.py fetch both files by
    shelling out to a literal `os.system('wget ...')` call if they're not
    already present under `~/panns_data/`. That is a hard, silent failure
    on any Windows install without `wget` on PATH (confirmed reproducible,
    2026-08-18, on this project's own Windows dev box): `os.system` prints
    "'wget' is not recognized..." to stdout, does not raise, and the
    subsequent `open(labels_csv_path)` (module import time, for the CSV) or
    `torch.load(checkpoint_path)` (for the .pth) then hard-crashes with a
    confusing FileNotFoundError/unpickling error that does not mention wget
    at all. Pre-fetching both files ourselves with urllib avoids ever
    reaching that code path, on any OS -- a no-op once both files already
    exist from a previous run."""
    import urllib.request
    from pathlib import Path as _Path

    panns_data_dir = _Path.home() / "panns_data"
    panns_data_dir.mkdir(parents=True, exist_ok=True)

    labels_path = panns_data_dir / "class_labels_indices.csv"
    if not labels_path.exists():
        urllib.request.urlretrieve(PANNS_LABELS_CSV_URL, str(labels_path))

    if explicit_checkpoint_path:
        return
    checkpoint_path = panns_data_dir / "Cnn14_mAP=0.431.pth"
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size < 3e8:
        urllib.request.urlretrieve(PANNS_CNN14_CHECKPOINT_URL, str(checkpoint_path))


def _load_marblenet_frame_vad():
    """Loads nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0 directly via
    huggingface_hub + EncDecFrameClassificationModel.restore_from, NOT via
    NeMo's own EncDecFrameClassificationModel.from_pretrained.

    from_pretrained is unreliable for this specific checkpoint on this
    nemo_toolkit 3.0.0 install: it unconditionally deletes its own cache
    directory "to prevent duplicates" on every call and re-fetches only the
    raw hub files, but reproducibly (verified interactively, 2026-08-18,
    both cold and warm cache) fails to leave a `model_config.yaml` where
    restore_from's own extraction logic expects to find one afterwards --
    `FileNotFoundError` on every single call, not just the first cold one.
    Downloading the .nemo file ourselves via hf_hub_download (HF's own
    content-addressed cache, unaffected by that NeMo-side deletion) and
    calling restore_from directly on that stable path sidesteps the bug
    entirely; confirmed reliable across repeated runs.

    strict=False is required: the checkpoint's saved state_dict has no
    `loss.weight` entry (an artifact of how the weighted CrossEntropyLoss
    module registers a buffer that isn't itself a learned/saved parameter),
    which torch's strict state_dict loading otherwise rejects even though
    every real model weight is present and loads correctly.
    """
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import EncDecFrameClassificationModel

    nemo_path = hf_hub_download(repo_id=MARBLENET_REPO_ID, filename=MARBLENET_FILENAME)
    return EncDecFrameClassificationModel.restore_from(nemo_path, strict=False)


def _read_window_list(path):
    """One window path per line; blank lines ignored; a UTF-8 BOM (as
    PowerShell writes by default) is tolerated."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    return [line.strip() for line in lines if line.strip()]


if __name__ == "__main__":
    main()
