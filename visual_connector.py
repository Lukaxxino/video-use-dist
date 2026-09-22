import os
import cv2
import logging
from pathlib import Path
from typing import List, Dict, Any
from scenedetect import detect, ContentDetector

logger = logging.getLogger(__name__)

import json
import time
import base64
import requests
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_scene_screenshots(video_path: str, scenes: List[tuple], output_dir: Path, screenshots_per_scene: int) -> List[Dict[str, Any]]:
    """
    Extracts screenshots from a video for given scenes using ffmpeg.
    """
    # Try to get fps using ffprobe
    fps = 25.0
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v", "-of", "default=noprint_wrappers=1:nokey=1", "-show_entries", "stream=r_frame_rate", video_path]
        output = subprocess.check_output(cmd, text=True).strip()
        if "/" in output:
            num, den = output.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(output)
    except Exception as e:
        logger.warning(f"Could not get fps with ffprobe, falling back to 25.0: {e}")

    scene_data = []

    for i, scene in enumerate(scenes):
        scene_idx = i + 1
        scene_dir = output_dir / f"scene_{scene_idx:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        start_time_sec = scene[0].get_seconds()
        end_time_sec = scene[1].get_seconds()
        
        duration_sec = end_time_sec - start_time_sec
        
        # Calculate time intervals for screenshots
        if screenshots_per_scene <= 1:
            time_indices = [start_time_sec + duration_sec / 2]
        else:
            step = duration_sec / screenshots_per_scene
            time_indices = [start_time_sec + (j * step) + (step / 2) for j in range(screenshots_per_scene)]
            
        saved_screenshots = []
        for idx_in_scene, timestamp_sec in enumerate(time_indices):
            screenshot_filename = f"screenshot_{idx_in_scene + 1:02d}.jpg"
            screenshot_path = scene_dir / screenshot_filename
            
            # Extract frame using ffmpeg
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-ss", str(timestamp_sec), "-i", video_path, 
                    "-vframes", "1", "-q:v", "2", str(screenshot_path)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                if screenshot_path.exists():
                    saved_screenshots.append({
                        "path": str(screenshot_path),
                        "timestamp": timestamp_sec
                    })
            except Exception as e:
                logger.error(f"Failed to extract frame at {timestamp_sec}s: {e}")

        scene_data.append({
            "scene_number": scene_idx,
            "start_time": start_time_sec,
            "end_time": end_time_sec,
            "screenshots": saved_screenshots
        })

    return scene_data

def get_env_var(key: str) -> str:
    """Helper to get an environment variable, falling back to reading .env"""
    val = os.environ.get(key)
    if val: return val
    
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

def visual_description_agent_ollama(scene_info: dict) -> str:
    """
    Calls a local Ollama model to describe what is happening in the scene based on screenshots.
    """
    ollama_url = get_env_var("OLLAMA_URL") or "http://localhost:11434"
    model_name = get_env_var("OLLAMA_VISION_MODEL")
    
    if not model_name:
        return ""
        
    api_endpoint = f"{ollama_url.rstrip('/')}/api/generate"
    
    base64_images = []
    for screenshot in scene_info['screenshots']:
        path = screenshot['path']
        try:
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                base64_images.append(encoded)
        except Exception as e:
            logger.error(f"Failed to read image {path}: {e}")
            
    timestamps_text = ", ".join([f"Image {i+1} is at {s['timestamp']}s" for i, s in enumerate(scene_info['screenshots'])])
    
    prompt = (
        f"You are a video editor's assistant analyzing a sequence of {len(scene_info['screenshots'])} screenshots from a single video scene. "
        f"The timestamps for these images in order are: {timestamps_text}. "
        "Focus strictly on the VISUALS. "
        "Your job is to provide an overall visual description of the scene AND detail the specific action happening in each individual screenshot. "
        "Output your response STRICTLY as a JSON object with the following schema:\n"
        "{\n"
        '  "environment": "The setting or background (e.g., office, outdoor street)",\n'
        '  "subjects": "Who/what is on screen",\n'
        '  "action": "The overall action or camera movement",\n'
        '  "flaws_or_notes": "Any notable visual changes or flaws",\n'
        '  "screenshots": [\n'
        '    {"timestamp": 12.5, "description": "Specific action happening exactly at this moment in the image"}\n'
        '  ]\n'
        "}\n"
        "Ensure the 'screenshots' array contains exactly one object for each image provided, using its exact timestamp. "
        "Do not include any markdown formatting, backticks, or extra text. Output ONLY the raw JSON."
    )
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": base64_images,
        "stream": False,
        # Constrain the decoder to valid JSON so descriptions always parse
        # (some models otherwise wrap output in prose/markdown fences).
        "format": "json"
    }
    
    print(f"\n[VISION AI] Analyzuji scenu {scene_info['scene_number']} pres {model_name} ({ollama_url})...")

    content_list = [{"type": "text", "text": prompt}]
    for b64_img in base64_images:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
        })

    openai_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content_list}],
        "temperature": 0.2
    }

    # Pro LiteLLM / vLLM proxy voláme přímo /chat/completions
    openai_endpoints = [
        f"{ollama_url.rstrip('/')}/chat/completions",
        f"{ollama_url.rstrip('/')}/v1/chat/completions"
    ]

    for ep in openai_endpoints:
        try:
            resp = requests.post(ep, json=openai_payload, timeout=90)
            if resp.status_code == 200:
                o_data = resp.json()
                choices = o_data.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "").strip()
                    if text:
                        return _clean_json_markdown(text, scene_info['scene_number'])
        except Exception as ex:
            logger.debug(f"OpenAI endpoint {ep} failed: {ex}")

    # Fallback na Ollama native API (/api/generate)
    try:
        response = requests.post(api_endpoint, json=payload, timeout=90)
        if response.status_code == 200:
            data = response.json()
            text = data.get("response", "").strip()
            if text:
                return _clean_json_markdown(text, scene_info['scene_number'])
    except Exception as e:
        logger.debug(f"Ollama native endpoint failed: {e}")

    logger.error("Všechny Vision AI endpointy selhaly.")
    return ""


def _clean_json_markdown(text: str, scene_number: int) -> str:
    print(f"[VISION AI] Scena {scene_number} uspesne analyzovana.\n[VISION AI] Vystup:\n{text}\n")
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def visual_description_dispatcher(
    scene_info: dict, return_backend: bool = False
):
    """
    Decides whether to use Ollama, or fallback to mock description based on .env config.
    """
    if get_env_var("OLLAMA_VISION_MODEL"):
        result = visual_description_agent_ollama(scene_info)
        if result:
            return (result, "ollama") if return_backend else result
        logger.warning("Ollama call failed or returned empty.")
        
    logger.warning("All visual AI agents failed or are unconfigured. Falling back to mock description.")
    result = mock_visual_description_agent(scene_info)
    return (result, "mock") if return_backend else result

def mock_visual_description_agent(scene_info: dict) -> str:
    """
    A mock connector representing an AI agent that describes what is happening in the scene based on screenshots.
    """
    screenshots = [{"timestamp": s["timestamp"], "description": "Mock specific detail"} for s in scene_info["screenshots"]]
    res = {
        "environment": "Mock environment",
        "subjects": "Mock subjects",
        "action": "Mock action",
        "flaws_or_notes": "Mock notes",
        "screenshots": screenshots
    }
    return json.dumps(res)


def load_or_extract_scenes(
    video_path: Path,
    static_dir: Path,
    screenshots_per_scene: int = 3,
    require_cached: bool = False,
) -> List[Dict[str, Any]]:
    """Load or create immutable scene-extraction artifacts under STATIC."""
    video_path = Path(video_path)
    static_dir = Path(static_dir)
    cache_path = static_dir / "scenes.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if require_cached:
        raise FileNotFoundError(f"Required scene cache not found: {cache_path}")

    logger.info("Detecting scenes using PySceneDetect...")
    detected_scenes = detect(str(video_path), ContentDetector())
    if not detected_scenes:
        from scenedetect import FrameTimecode

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        detected_scenes = [
            (
                FrameTimecode(timecode=0, fps=fps),
                FrameTimecode(timecode=total_frames, fps=fps),
            )
        ]
        logger.info("No scenes detected, treating whole video as one scene.")

    screenshots_dir = static_dir / "scenes"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    scene_data = extract_scene_screenshots(
        str(video_path),
        detected_scenes,
        screenshots_dir,
        screenshots_per_scene,
    )
    static_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(scene_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return scene_data


def describe_scenes(
    scenes: List[Dict[str, Any]],
    visual_run_dir: Path,
    limit_visuals: int | None = None,
    require_cached: bool = False,
) -> tuple[List[Dict[str, Any]], dict]:
    """Attach visual descriptions to scene copies in one ACTIVE visual run."""
    visual_run_dir = Path(visual_run_dir)
    cache_path = visual_run_dir / "visual_descriptions_cache.json"
    described_scenes = [dict(scene) for scene in scenes]
    stats = {
        "visual_seconds": 0.0,
        "scenes_analyzed": 0,
        "screenshots_analyzed": 0,
        "backend": "cached" if require_cached else "not-run",
    }

    if require_cached:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Required visual description cache not found: {cache_path}"
            )
        visual_descriptions = json.loads(
            cache_path.read_text(encoding="utf-8")
        )
        missing_description = (
            '{"environment": "Missing in cache", "subjects": "Missing", '
            '"action": "Missing", "flaws_or_notes": "Missing", '
            '"screenshots": []}'
        )
        for scene in described_scenes:
            scene["visual_description"] = visual_descriptions.get(
                str(scene["scene_number"]), missing_description
            )
        return described_scenes, stats

    visual_descriptions = {}
    backends = set()
    visual_t0 = time.perf_counter()

    workers_str = get_env_var("OLLAMA_PARALLEL_WORKERS") or "1"
    try:
        max_workers = max(1, int(workers_str))
    except ValueError:
        max_workers = 1

    def process_single_scene(index, scene):
        if limit_visuals is not None and index >= limit_visuals:
            desc = (
                '{"environment": "Skipped", "subjects": "Skipped", '
                '"action": "Skipped", '
                '"flaws_or_notes": "Skipped due to --limit-visuals", '
                '"screenshots": []}'
            )
            b_end = "skipped"
            sc_count = 0
        else:
            dispatch_result = visual_description_dispatcher(
                scene, return_backend=True
            )
            if isinstance(dispatch_result, tuple) and len(dispatch_result) == 2:
                desc, b_end = dispatch_result
            else:
                desc, b_end = dispatch_result, "unknown"
            sc_count = len(scene.get("screenshots", []))
        return index, scene, desc, b_end, sc_count

    if max_workers > 1 and len(described_scenes) > 1:
        logger.info(f"Analyzing {len(described_scenes)} scenes in parallel using {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_scene, idx, sc)
                for idx, sc in enumerate(described_scenes)
            ]
            for future in as_completed(futures):
                idx, sc, description, backend, sc_count = future.result()
                if backend != "skipped":
                    backends.add(backend)
                    stats["scenes_analyzed"] += 1
                    stats["screenshots_analyzed"] += sc_count
                sc["visual_description"] = description
                visual_descriptions[str(sc["scene_number"])] = description
    else:
        for index, scene in enumerate(described_scenes):
            _, _, description, backend, sc_count = process_single_scene(index, scene)
            if backend != "skipped":
                backends.add(backend)
                stats["scenes_analyzed"] += 1
                stats["screenshots_analyzed"] += sc_count
            scene["visual_description"] = description
            visual_descriptions[str(scene["scene_number"])] = description

    stats["visual_seconds"] = round(time.perf_counter() - visual_t0, 3)
    if len(backends) == 1:
        stats["backend"] = next(iter(backends))
    elif len(backends) > 1:
        stats["backend"] = "mixed"
    visual_run_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(visual_descriptions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return described_scenes, stats
