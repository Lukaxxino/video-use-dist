import json
from pathlib import Path

from helpers.process_whisper_output import _build_speaker_segments, _speaker_for


def process_nemo_to_scribe(
    raw_json_path: Path, output_json_path: Path, language_code: str = "ces"
) -> dict:
    """
    Transforms the RAW output from helpers/nemo_server.py (a word list plus
    speaker segments) into the same Scribe schema produced by
    process_whisper_to_scribe, so downstream code (expand_transcript_with_visuals,
    audio_analyzer.py, pack_transcripts.py) sees no difference between backends.

    language_code defaults to Czech ("ces") unless overridden.
    """
    if not raw_json_path.exists():
        raise FileNotFoundError(f"Raw NeMo JSON not found at {raw_json_path}")

    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    text = raw_data.get("text", "")
    words = raw_data.get("words", [])

    # Reuse the same speaker-segment shape whisper's helper expects
    # ({"speaker": ..., "timestamp": [start, end]}) rather than duplicating
    # the matching logic for NeMo's {"start", "end", "speaker"} shape.
    speaker_segs = _build_speaker_segments(
        [
            {"speaker": s.get("speaker"), "timestamp": [s.get("start"), s.get("end")]}
            for s in raw_data.get("speaker_segments", [])
        ]
    )

    formatted_words = []
    last_end = 0.0

    for w in words:
        start = float(w.get("start", last_end))
        end = float(w.get("end", start))
        word_text = (w.get("word") or "").strip()
        speaker_id = _speaker_for(start, speaker_segs)

        if start > last_end:
            formatted_words.append({
                "text": " ",
                "start": last_end,
                "end": start,
                "type": "spacing",
                "speaker_id": speaker_id,
                "logprob": 0.0,
            })

        if word_text:
            formatted_words.append({
                "text": word_text,
                "start": start,
                "end": end,
                "type": "word",
                "speaker_id": speaker_id,
                "logprob": 0.0,
            })

        last_end = end

    scribe_format = {
        "language_code": language_code,
        "language_probability": 1.0,
        "text": text,
        "words": formatted_words,
        # Passed through as-is from the ASR backend's own response (see
        # grace_hopper/asr_server.py / helpers/canary_server.py's VAD/music
        # gate) -- {"start", "end", "type": "music"|"silence",
        # "music_probability"} spans, not per-word data, so there is no
        # per-word reshaping to do here unlike `words` above.
        "audio_events": raw_data.get("audio_events", []),
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(scribe_format, f, indent=2, ensure_ascii=False)

    print(
        f"[Process NeMo] Transformed {len(words)} words into "
        f"{len(formatted_words)} Scribe-compatible word objects."
    )
    return scribe_format


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("raw_json", type=Path)
    ap.add_argument("out_json", type=Path)
    ap.add_argument(
        "--language",
        default="ces",
        help="language_code stamped on the output (default: ces/Czech)",
    )
    args = ap.parse_args()
    process_nemo_to_scribe(args.raw_json, args.out_json, language_code=args.language)
