"""Transcribe a video via a pluggable ASR backend.

Extracts mono 16kHz audio via ffmpeg, POSTs it to the selected backend's
local server (WHISPER_URL / NEMO_URL, both speaking the same /v1/transcribe
contract — see ASR_BACKENDS below), optionally authenticating with the
matching *_API_KEY, then reshapes the raw output into the Scribe-style
word-level schema at <output_dir>/transcript.json.

All settings live in `.env`, so the same client works against local FastAPI
servers today and remote/on-prem ASR endpoints (URL + API key) tomorrow.

Cached: if the output file already exists, the upload is skipped.

Usage:
    python helpers/transcribe.py <video_path> --output-dir /custom/transcription-run
    python helpers/transcribe.py <video_path> --output-dir /custom/transcription-run --asr-backend nemo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

try:
    from helpers.env_config import get_env
except ImportError:  # when run as a standalone script from helpers/
    from env_config import get_env



def extract_audio(video_path: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-vn", "-i", str(video_path),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# Whisper decodes with ISO 639-1 codes ("cs"); ElevenLabs stamps ISO 639-3 ("ces").
# Map the common cases so the output's language_code matches the ElevenLabs schema.
ISO_639_1_TO_3 = {
    "cs": "ces", "en": "eng", "sk": "slk", "de": "deu", "fr": "fra",
    "es": "spa", "it": "ita", "pl": "pol", "ru": "rus", "uk": "ukr",
}


# Each backend speaks the same /v1/transcribe HTTP contract (multipart
# `file` + optional `language`), so adding one here is enough to make it
# selectable via --asr-backend — no other code needs to know which backend
# is in use.
ASR_BACKENDS = {
    "whisper": {
        "url_env": "WHISPER_URL",
        "default_url": "http://localhost:8000/v1/transcribe",
        "api_key_env": "WHISPER_API_KEY",
    },
    "nemo": {
        "url_env": "NEMO_URL",
        "default_url": "http://localhost:8001/v1/transcribe",
        "api_key_env": "NEMO_API_KEY",
    },
}


def _processor_for(asr_backend: str):
    """Lazily imports the raw-output normalizer for the given backend, so
    importing helpers.transcribe never requires nemo_toolkit or any other
    backend-specific dependency to be installed."""
    if asr_backend == "nemo":
        from helpers.process_nemo_output import process_nemo_to_scribe

        return process_nemo_to_scribe
    from helpers.process_whisper_output import process_whisper_to_scribe

    return process_whisper_to_scribe


def call_asr_server(
    audio_path: Path,
    url: str,
    language: str | None = None,
    api_key: str = "",
) -> dict:
    target_url = url.strip()
    if not target_url.endswith("/v1/transcribe"):
        target_url = f"{target_url.rstrip('/')}/v1/transcribe"

    print(f"  [ASR] Sending {audio_path.name} to {target_url} (language={language or 'auto'})")
    data = {}
    if language is not None:
        data["language"] = language
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with open(audio_path, "rb") as f:
        resp = requests.post(
            target_url,
            files={"file": (audio_path.name, f, "audio/wav")},
            data=data,
            headers=headers,
            timeout=3600,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"ASR server returned {resp.status_code}: {resp.text[:500]}")

    return resp.json()


def transcribe_one(
    video: Path,
    output_dir: Path,
    language: str | None = "cs",
    asr_backend: str = "whisper",
    verbose: bool = True,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    language: ISO 639-1 code forced on the Whisper decoder (default "cs"/Czech);
    informational only for the nemo backend, which auto-detects language.
    Pass None or "" to let Whisper auto-detect. The output's `language_code`
    is stamped with the matching ISO 639-3 (e.g. "ces") to match ElevenLabs.

    asr_backend: one of ASR_BACKENDS ("whisper" or "nemo"). Selects which
    *_URL/*_API_KEY env vars and which raw-output normalizer to use.

    Cached: returns existing path immediately if the transcript already exists.
    """
    if asr_backend not in ASR_BACKENDS:
        raise ValueError(
            f"Unknown --asr-backend '{asr_backend}'; choose one of "
            f"{sorted(ASR_BACKENDS)}"
        )
    backend = ASR_BACKENDS[asr_backend]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "transcript.json"

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    asr_url = get_env(backend["url_env"], backend["default_url"])
    asr_api_key = get_env(backend["api_key_env"], "")

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio)
        size_mb = audio.stat().st_size / (1024 * 1024)
        if verbose:
            print(f"  audio ready ({size_mb:.1f} MB)", flush=True)

        raw_out_path = output_dir / "transcript_raw.json"
        if raw_out_path.exists() and raw_out_path.stat().st_size > 0:
            print(f"  already transcribed (raw): {raw_out_path.name}")
        else:
            try:
                raw_payload = call_asr_server(
                    audio, asr_url, language=language, api_key=asr_api_key
                )
            except requests.exceptions.ConnectionError as exc:
                server_script = "nemo_server.py" if asr_backend == "nemo" else "whisper_server.py"
                raise RuntimeError(
                    f"Could not reach the '{asr_backend}' ASR server at {asr_url} — "
                    f"is it running? Start it with 'python helpers/{server_script}'."
                ) from exc
            raw_out_path.write_text(
                json.dumps(raw_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if verbose:
                print(f"  saved raw {asr_backend} output to {raw_out_path.name}")

        lang_code = ISO_639_1_TO_3.get(language, language) if language else "unknown"
        process_to_scribe = _processor_for(asr_backend)
        payload = process_to_scribe(raw_out_path, out_path, language_code=lang_code)

    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        if isinstance(payload, dict) and "words" in payload:
            print(f"    words: {len(payload['words'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video via a pluggable ASR backend")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Resolved transcription run directory",
    )
    ap.add_argument(
        "--language",
        type=str,
        default="cs",
        help="ISO 639-1 language forced on the Whisper decoder (default: cs/Czech). Pass '' to auto-detect.",
    )
    ap.add_argument(
        "--asr-backend",
        choices=sorted(ASR_BACKENDS),
        default="whisper",
        help="Which ASR backend to use (default: whisper)",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    output_dir = args.output_dir.resolve()
    transcribe_one(
        video=video,
        output_dir=output_dir,
        language=(args.language or None),
        asr_backend=args.asr_backend,
    )


if __name__ == "__main__":
    main()
