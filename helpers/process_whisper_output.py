import json
from pathlib import Path


def _normalize_speaker(label) -> str:
    """Map pyannote labels ("SPEAKER_00") to the ElevenLabs style ("speaker_0")."""
    if label is None:
        return "speaker_0"
    s = str(label)
    tail = s.split("_")[-1]
    if tail.isdigit():
        return f"speaker_{int(tail)}"
    return s


def _build_speaker_segments(speakers: list) -> list:
    """From insanely-fast-whisper's `speakers` list (each entry has `speaker` +
    `timestamp` [start, end]), build sorted (start, end, speaker_id) tuples."""
    segs = []
    for e in speakers or []:
        ts = e.get("timestamp") or [None, None]
        start = ts[0]
        end = ts[1] if len(ts) > 1 else None
        if start is None:
            continue
        start = float(start)
        end = float(end) if end is not None else start
        segs.append((start, end, _normalize_speaker(e.get("speaker"))))
    segs.sort(key=lambda x: x[0])
    return segs


def _speaker_for(start_time: float, segs: list) -> str:
    """Speaker whose segment covers start_time; else the nearest by start; else speaker_0."""
    if not segs:
        return "speaker_0"
    for s, e, spk in segs:
        if s <= start_time <= e:
            return spk
    return min(segs, key=lambda x: abs(x[0] - start_time))[2]


def process_whisper_to_scribe(raw_json_path: Path, output_json_path: Path, language_code: str = "ces"):
    """
    Transforms the RAW output from insanely-fast-whisper into the ElevenLabs Scribe schema
    expected by video-use (main.py).

    language_code defaults to Czech ("ces", matching ElevenLabs' ISO 639-3) unless overridden.
    """
    if not raw_json_path.exists():
        raise FileNotFoundError(f"Raw whisper JSON not found at {raw_json_path}")
        
    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    chunks = raw_data.get("chunks", [])
    text = raw_data.get("text", "")
    # Diarization (when enabled) lands in a separate `speakers` list, NOT on each chunk.
    # Map speakers onto words by timestamp; falls back to a single speaker_0 if absent.
    speaker_segs = _build_speaker_segments(raw_data.get("speakers", []))

    formatted_words = []

    last_end = 0.0

    for chunk in chunks:
        # Some whisper outputs might give (start, end), others [start, end]
        timestamp = chunk.get("timestamp", [0.0, 0.0])
        start = float(timestamp[0]) if timestamp[0] is not None else last_end
        end = float(timestamp[1]) if len(timestamp) > 1 and timestamp[1] is not None else start + 0.5

        chunk_text = chunk.get("text", "")
        speaker_id = _speaker_for(start, speaker_segs)
        
        # Strip leading/trailing spaces for the actual word block
        clean_text = chunk_text.strip()
        
        # If there's a significant gap between the last word and this one, insert a spacing object
        if start > last_end:
            formatted_words.append({
                "text": " ",
                "start": last_end,
                "end": start,
                "type": "spacing",
                "speaker_id": speaker_id,
                "logprob": 0.0
            })
            
        if clean_text:
            formatted_words.append({
                "text": clean_text,
                "start": start,
                "end": end,
                "type": "word",
                "speaker_id": speaker_id,
                "logprob": 0.0
            })
            
        # If the chunk text had a trailing space, we can also inject a trailing space exactly at 'end'
        if chunk_text.endswith(" "):
            formatted_words.append({
                "text": " ",
                "start": end,
                "end": end,
                "type": "spacing",
                "speaker_id": speaker_id,
                "logprob": 0.0
            })
            
        last_end = end
        
    scribe_format = {
        "language_code": language_code,
        "language_probability": 1.0,
        "text": text,
        "words": formatted_words
    }
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(scribe_format, f, indent=2, ensure_ascii=False)
        
    print(f"[Process Whisper] Transformed {len(chunks)} chunks into {len(formatted_words)} Scribe-compatible word objects.")
    return scribe_format

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_json", type=Path)
    ap.add_argument("out_json", type=Path)
    ap.add_argument("--language", default="ces", help="language_code stamped on the output (default: ces/Czech)")
    args = ap.parse_args()
    process_whisper_to_scribe(args.raw_json, args.out_json, language_code=args.language)
