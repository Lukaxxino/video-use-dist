"""Pack one versioned transcription run into readable phrase-level markdown.

Groups word-level entries into phrase-level lines, breaking on any silence
>= 0.5s OR speaker change. Each phrase gets a [start-end] prefix. This is
the PRIMARY artifact the editor sub-agent reads to pick cuts — it fits one
hour of takes in a tenth the tokens of raw Scribe JSON and gives
word-boundary precision from text alone.

Input: <transcription-run>/transcript.json
Output: <transcription-run>/takes_packed.md

Usage:
    python helpers/pack_transcripts.py --transcription-dir <run_dir>
    python helpers/pack_transcripts.py --transcription-dir <run_dir> --silence-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def format_time(seconds: float) -> str:
    """Format a time in seconds as "NNN.NN" with fixed 6-char width for alignment."""
    return f"{seconds:06.2f}"


def format_duration(seconds: float) -> str:
    """Format a duration as "Ms" or "Mm SSs"."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}m {s:04.1f}s"


def group_into_phrases(
    words: list[dict],
    silence_threshold: float = 0.5,
) -> list[dict]:
    """Walk a Scribe word list, break into phrases on silence >= threshold
    OR speaker change. Returns list of {start, end, text, speaker_id}.

    Scribe `words` entries have types 'word', 'spacing', or 'audio_event'.
    We keep 'word' and 'audio_event' content in phrase text. 'spacing'
    entries carry the silence information via their start/end times.
    """
    phrases: list[dict] = []
    current_words: list[dict] = []
    current_start: float | None = None
    current_speaker: str | None = None

    def flush() -> None:
        nonlocal current_words, current_start, current_speaker
        if not current_words:
            return
        text_parts: list[str] = []
        for w in current_words:
            t = w.get("type", "word")
            raw = (w.get("text") or "").strip()
            if not raw:
                continue
            if t == "audio_event":
                if not raw.startswith("("):
                    raw = f"({raw})"
            text_parts.append(raw)
        if not text_parts:
            current_words = []
            current_start = None
            current_speaker = None
            return
        text = " ".join(text_parts)
        text = text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
        end_time = current_words[-1].get("end", current_words[-1].get("start", current_start or 0.0))
        phrases.append({
            "start": current_start,
            "end": end_time,
            "text": text,
            "speaker_id": current_speaker,
        })
        current_words = []
        current_start = None
        current_speaker = None

    prev_end: float | None = None

    for w in words:
        t = w.get("type", "word")
        if t == "spacing":
            # spacing entries mark the gaps between words; if the gap is long,
            # flush the current phrase.
            start = w.get("start")
            end = w.get("end")
            if start is not None and end is not None:
                gap = end - start
                if gap >= silence_threshold:
                    flush()
            continue

        # 'word' or 'audio_event'
        start = w.get("start")
        if start is None:
            continue
        speaker = w.get("speaker_id")

        # Flush on speaker change
        if current_speaker is not None and speaker is not None and speaker != current_speaker:
            flush()

        # Flush on a long gap from the previous kept token
        if prev_end is not None and start - prev_end >= silence_threshold:
            flush()

        if current_start is None:
            current_start = start
            current_speaker = speaker
        current_words.append(w)
        prev_end = w.get("end", start)

    flush()
    return phrases


def pack_one_file(json_path: Path, silence_threshold: float) -> tuple[str, float, list[dict]]:
    """Return (header_name, duration, phrases) for one transcript file."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    words = data.get("words", [])
    phrases = group_into_phrases(words, silence_threshold)
    if phrases:
        duration = phrases[-1]["end"] - phrases[0]["start"]
    else:
        duration = 0.0
    return json_path.stem, duration, phrases


def render_markdown(entries: list[tuple[str, float, list[dict]]], silence_threshold: float) -> str:
    lines: list[str] = []
    lines.append("# Packed transcripts")
    lines.append("")
    lines.append(f"Phrase-level, grouped on silences ≥ {silence_threshold:.1f}s or speaker change.")
    lines.append("Use `[start-end]` ranges to address cuts in the EDL.")
    lines.append("")
    for name, duration, phrases in entries:
        lines.append(f"## {name}  (duration: {format_duration(duration)}, {len(phrases)} phrases)")
        if not phrases:
            lines.append("  _no speech detected_")
            lines.append("")
            continue
        for p in phrases:
            spk = p.get("speaker_id")
            if spk is not None:
                # Scribe returns IDs like "speaker_0" — strip the prefix for readability
                spk_str = str(spk)
                if spk_str.startswith("speaker_"):
                    spk_str = spk_str[len("speaker_"):]
                spk_tag = f" S{spk_str}"
            else:
                spk_tag = ""
            lines.append(f"  [{format_time(p['start'])}-{format_time(p['end'])}]{spk_tag} {p['text']}")
        lines.append("")
    return "\n".join(lines)


def pack_transcription_run(
    transcription_dir: Path,
    silence_threshold: float = 0.5,
    output: Path | None = None,
) -> Path:
    """Pack the canonical processed transcript from one transcription run."""
    transcription_dir = Path(transcription_dir)
    transcript_path = transcription_dir / "transcript.json"
    entry = pack_one_file(transcript_path, silence_threshold)
    output_path = (
        Path(output)
        if output is not None
        else transcription_dir / "takes_packed.md"
    )
    output_path.write_text(
        render_markdown([entry], silence_threshold),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack Scribe transcripts into takes_packed.md")
    ap.add_argument(
        "--transcription-dir",
        type=Path,
        required=True,
        help="Selected versioned transcription run containing transcript.json",
    )
    ap.add_argument(
        "--silence-threshold",
        type=float,
        default=0.5,
        help="Break phrases on silences >= this (seconds). Default 0.5.",
    )
    ap.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path (default: <transcription-dir>/takes_packed.md)",
    )
    args = ap.parse_args()

    transcription_dir = args.transcription_dir.resolve()
    transcript_path = transcription_dir / "transcript.json"
    if not transcript_path.is_file():
        sys.exit(f"no transcript.json at {transcript_path}")

    out_path = pack_transcription_run(
        transcription_dir,
        silence_threshold=args.silence_threshold,
        output=args.output,
    )
    _name, total_duration, phrases = pack_one_file(
        transcript_path, args.silence_threshold
    )
    kb = out_path.stat().st_size / 1024
    print(f"packed transcript.json -> {out_path}")
    print(f"  {len(phrases)} phrases, {format_duration(total_duration)} total runtime")
    print(f"  {kb:.1f} KB")


if __name__ == "__main__":
    main()
