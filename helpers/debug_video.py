import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

def format_srt_time(seconds: float) -> str:
    """Converts seconds into SRT time format HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_scene_srt(combined_json: dict, output_path: Path):
    scenes = combined_json.get("scenes", [])
    srt_content = []
    
    for i, scene in enumerate(scenes):
        start_time = scene.get("start_time", 0.0)
        end_time = scene.get("end_time", 0.0)
        scene_num = scene.get("scene_number", i + 1)
        
        start_str = format_srt_time(start_time)
        end_str = format_srt_time(end_time)
        
        srt_content.append(str(i + 1))
        srt_content.append(f"{start_str} --> {end_str}")
        srt_content.append(f"Scene {scene_num}")
        srt_content.append("")
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_content))
    logger.info(f"Generated scene SRT: {output_path}")

def collect_timed_words(combined_json: dict) -> list[dict]:
    """Flatten scene-centric transcript words and order them by timestamp."""
    words = [
        word
        for scene in combined_json.get("scenes", [])
        for word in scene.get("words", [])
    ]
    return sorted(words, key=lambda word: word.get("start", 0.0))


def generate_transcript_srt(words: list[dict], output_path: Path):
    srt_content = []
    
    # We will combine words into small chunks (e.g., 5 words) for readability
    chunk_size = 5
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        if not chunk:
            continue
            
        start_time = chunk[0].get("start", 0.0)
        end_time = chunk[-1].get("end", 0.0)
        text = " ".join([w.get("text", "") for w in chunk])
        
        start_str = format_srt_time(start_time)
        end_str = format_srt_time(end_time)
        
        idx = (i // chunk_size) + 1
        srt_content.append(str(idx))
        srt_content.append(f"{start_str} --> {end_str}")
        srt_content.append(text)
        srt_content.append("")
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_content))
    logger.info(f"Generated transcript SRT: {output_path}")

def escape_path_for_ffmpeg(path: Path) -> str:
    """Escapes Windows paths for ffmpeg filtergraph (e.g. C:\\ -> C\\:\\\\)"""
    p = str(path).replace('\\', '/')
    if len(p) > 1 and p[1] == ':':
        p = p[0] + "\\:" + p[2:]
    return p

def debug_mode_name(burn_scenes: bool, burn_transcript: bool) -> str | None:
    if burn_scenes and burn_transcript:
        return "debug-scenes-transcript.mp4"
    if burn_scenes:
        return "debug-scenes.mp4"
    if burn_transcript:
        return "debug-transcript.mp4"
    return None


def create_debug_video(
    video_path: Path,
    combined_json_path: Path,
    output_dir: Path,
    burn_scenes: bool,
    burn_transcript: bool,
) -> Path | None:
    mode_name = debug_mode_name(burn_scenes, burn_transcript)
    if mode_name is None:
        logger.info("Neither --burn-scenes nor --burn-transcript specified. Skipping debug video.")
        return None

    output_dir = Path(output_dir)
    debug_video_path = output_dir / mode_name
    
    if not combined_json_path.exists():
        logger.error(f"Cannot create debug video: {combined_json_path} not found.")
        return None
        
    with open(combined_json_path, "r", encoding="utf-8") as f:
        combined_json = json.load(f)
        
    filters = []
    
    if burn_scenes:
        scenes_srt_path = output_dir / "debug_scenes.srt"
        generate_scene_srt(combined_json, scenes_srt_path)
        # Alignment 8 = top center
        escaped_scenes = escape_path_for_ffmpeg(scenes_srt_path)
        filters.append(f"subtitles='{escaped_scenes}':force_style='Alignment=8,Fontsize=36,MarginV=20,PrimaryColour=&H0000FFFF'")
        
    if burn_transcript:
        transcript_srt_path = output_dir / "debug_transcript.srt"
        generate_transcript_srt(
            collect_timed_words(combined_json), transcript_srt_path
        )
        # Alignment 2 = bottom center
        escaped_transcript = escape_path_for_ffmpeg(transcript_srt_path)
        filters.append(f"subtitles='{escaped_transcript}':force_style='Alignment=2,Fontsize=28,MarginV=20,PrimaryColour=&H00FFFFFF'")
        
    filter_graph = ",".join(filters)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", filter_graph,
        "-c:a", "copy",
        str(debug_video_path)
    ]
    
    logger.info(f"Generating debug video: {debug_video_path.name}...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        logger.info(f"Debug video successfully created at {debug_video_path}")
        return debug_video_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create debug video. FFmpeg error: {e.stderr.decode('utf-8', errors='ignore')}")
        return None
