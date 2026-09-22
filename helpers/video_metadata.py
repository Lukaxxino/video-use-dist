import hashlib
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_HASH_CHUNK_SIZE = 8 * 1024 * 1024


def _probe_ffprobe(video_path: Path) -> dict:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(video_path),
            ],
            capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)
    except Exception as e:
        logger.warning(f"ffprobe failed for {video_path}, technical metadata fields will be null: {e}")
        return {}


def _parse_fps(r_frame_rate: str) -> float:
    try:
        num, den = r_frame_rate.split("/")
        num, den = float(num), float(den)
        return round(num / den, 2) if den else None
    except Exception:
        return None


def _frame_rate_mode(video_stream: dict) -> str:
    """Classify only when ffprobe supplies both nominal and average rate."""
    nominal = _parse_fps(video_stream.get("r_frame_rate") or "")
    average = _parse_fps(video_stream.get("avg_frame_rate") or "")
    if nominal is None or average is None or nominal <= 0 or average <= 0:
        return "unknown"
    return "constant" if abs(nominal - average) <= 0.001 else "variable"


def _sha1_of_file(video_path: Path) -> str:
    try:
        digest = hashlib.sha1()
        with open(video_path, "rb") as f:
            while True:
                chunk = f.read(_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except Exception as e:
        logger.warning(f"Could not compute checksum for {video_path}: {e}")
        return None


def _resolve_recorded_at(video_path: Path, recorded_date_override: str, format_tags: dict):
    if recorded_date_override:
        return recorded_date_override, "user_provided"
    creation_time = format_tags.get("creation_time")
    if creation_time:
        return creation_time, "container_creation_time"
    try:
        mtime = datetime.fromtimestamp(video_path.stat().st_mtime).isoformat()
    except OSError as e:
        logger.warning(f"Could not stat {video_path} for mtime fallback: {e}")
        mtime = None
    return mtime, "file_mtime_fallback"


def build_video_metadata(
    video_path: Path,
    vision_model: str = "",
    whisper_model: str = "",
    director: str = "",
    project: str = "",
    notes: str = "",
    recorded_date_override: str = "",
) -> dict:
    probe = _probe_ffprobe(video_path)
    fmt = probe.get("format", {})
    tags = fmt.get("tags", {})
    streams = probe.get("streams", [])

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

    duration_seconds = None
    if fmt.get("duration"):
        try:
            duration_seconds = round(float(fmt["duration"]), 2)
        except (TypeError, ValueError):
            duration_seconds = None

    resolution = None
    if video_stream.get("width") and video_stream.get("height"):
        resolution = f"{video_stream['width']}x{video_stream['height']}"

    fps = _parse_fps(video_stream["r_frame_rate"]) if video_stream.get("r_frame_rate") else None
    frame_rate_mode = _frame_rate_mode(video_stream)

    recorded_at, recorded_at_source = _resolve_recorded_at(video_path, recorded_date_override, tags)

    try:
        file_size_bytes = video_path.stat().st_size
    except OSError as e:
        logger.warning(f"Could not stat {video_path}: {e}")
        file_size_bytes = None

    return {
        "video_filename": video_path.name,
        "video_path": str(video_path.resolve()),
        "file_size_bytes": file_size_bytes,
        "video_checksum_sha1": _sha1_of_file(video_path),
        "duration_seconds": duration_seconds,
        "resolution": resolution,
        "fps": fps,
        "frame_rate_mode": frame_rate_mode,
        "video_codec": video_stream.get("codec_name"),
        "audio_codec": audio_stream.get("codec_name"),
        "vision_model": vision_model or None,
        "whisper_model": whisper_model or None,
        "recorded_at": recorded_at,
        "recorded_at_source": recorded_at_source,
        "scanned_at": datetime.now().isoformat(),
        "director": director or None,
        "project": project or None,
        "notes": notes or None,
    }
