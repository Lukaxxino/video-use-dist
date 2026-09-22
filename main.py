import sys
import json
import logging
import shutil
import contextlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# Ensure the script's directory is in the path to import local modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Makes a static ffmpeg/ffprobe build resolvable on PATH for this process
# (and everything it subprocess.run()s) without a separate manual install --
# static-ffmpeg downloads the platform build once and caches it under its
# own package directory on first use. Every helper that shells out to
# ffmpeg/ffprobe (audio_analyzer, debug_video, transcribe, video_metadata,
# visual_connector) is imported below, after this call, so they all see the
# adjusted PATH. Wrapped defensively: a machine with ffmpeg/ffprobe already
# on PATH by other means (or offline, with static-ffmpeg not yet able to
# download) must still work exactly as before this existed.
try:
    from static_ffmpeg import add_paths as _add_static_ffmpeg_paths

    _add_static_ffmpeg_paths()
except Exception:
    logging.getLogger(__name__).warning(
        "static-ffmpeg could not provide ffmpeg/ffprobe; falling back to "
        "whatever is already on PATH", exc_info=True
    )

from helpers.transcribe import transcribe_one, ASR_BACKENDS
from visual_connector import describe_scenes, load_or_extract_scenes
from helpers.audio_analyzer import analyze_audio_dynamics
from helpers.debug_video import create_debug_video
from helpers.analytics import StageTimer, detect_machine
from helpers.env_config import get_env
from helpers.local_service_autostart import ensure_canary_ready
from helpers.storage import (
    ARTIFACT_BASES,
    ArtifactKind,
    ArtifactRef,
    VideoWorkspace,
    read_manifest,
    resolve_output_root,
    source_identity,
    validate_source_identity,
    write_manifest,
)
from helpers.video_metadata import build_video_metadata
from helpers.scene_script import ensure as ensure_scene_script

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATIC_SCENE_DETECTOR = "PySceneDetect ContentDetector"
STATIC_SCREENSHOTS_PER_SCENE = 3

def expand_transcript_with_visuals(
    transcript_data: dict,
    scene_data: list,
    dynamics: list,
    provenance: dict,
) -> dict:
    """
    Expands the original ElevenLabs JSON transcript with visual scene descriptions
    and restructures everything into a scene-centric format, including any
    non-speech audio events (music/silence spans) the ASR backend reported.
    """
    all_words = transcript_data.get("words", [])
    all_audio_events = transcript_data.get("audio_events", [])
    dynamics_by_scene = {
        item.get("scene_number"): item.get("dynamics", {})
        for item in dynamics
    }

    formatted_scenes = []
    for scene in scene_data:
        raw_desc = scene.get('visual_description', '{}')
        try:
            parsed_desc = json.loads(raw_desc)
        except Exception:
            parsed_desc = {"raw": raw_desc}

        start_t = scene['start_time']
        end_t = scene['end_time']

        scene_words = [w for w in all_words if start_t <= w.get("start", start_t) <= end_t]
        speakers = list(set([w.get("speaker_id") for w in scene_words if w.get("speaker_id")]))
        # Audio events are spans, not points -- a point-based start check
        # (like scene_words above) would miss an event that straddles this
        # scene's boundary, so this is an actual interval-overlap test.
        scene_audio_events = [
            e for e in all_audio_events
            if e.get("start", 0.0) < end_t and e.get("end", 0.0) > start_t
        ]

        formatted_scenes.append({
            "scene_number": scene['scene_number'],
            "start_time": start_t,
            "end_time": end_t,
            "speakers": speakers,
            "visual_description": parsed_desc,
            "words": scene_words,
            "audio_events": scene_audio_events,
            "audio_dynamics": dynamics_by_scene.get(scene["scene_number"], {}),
        })
        
    return {
        "language_code": transcript_data.get("language_code", "unknown"),
        "text": transcript_data.get("text", ""),
        "scenes": formatted_scenes,
        "provenance": provenance,
    }

def _created_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _artifact_label(artifact: ArtifactRef, *manifest_keys: str) -> str:
    for key in manifest_keys:
        value = artifact.manifest.get(key)
        if value:
            return str(value)
    return str(artifact.path.name.split(" - ", 1)[-1])


def _remove_reserved_artifact(
    workspace: VideoWorkspace, artifact: ArtifactRef
) -> None:
    root = workspace.root.resolve()
    target = artifact.path.resolve()
    expected_base = (
        workspace.root / ARTIFACT_BASES[artifact.kind]
    ).resolve()
    expected_base.relative_to(root)
    if target.parent != expected_base or target == root:
        raise RuntimeError(
            "Refusing to remove artifact outside expected artifact base "
            f"{expected_base}: {target}"
        )
    if target.exists():
        shutil.rmtree(target)


def _read_json(path: Path):
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def _cleanup_new_static(
    workspace: VideoWorkspace, static_dir: Path
) -> None:
    workspace_root = workspace.root.resolve()
    expected_static = (workspace.root / "STATIC").resolve()
    expected_static.relative_to(workspace_root)
    target = static_dir.resolve()
    if target != expected_static:
        raise RuntimeError(
            f"Refusing to clean unexpected STATIC path: {target}"
        )
    if target.exists():
        shutil.rmtree(target)
    (target / "scenes").mkdir(parents=True, exist_ok=True)


def _has_unmanifested_static_data(static_dir: Path) -> bool:
    if not static_dir.is_dir():
        return False
    for child in static_dir.iterdir():
        if child.name == "scenes" and child.is_dir():
            if any(child.iterdir()):
                return True
        else:
            return True
    return False


def _load_static(
    workspace: VideoWorkspace,
    video_path: Path,
    current_source: dict,
    skip_extraction: bool,
    timer: StageTimer,
) -> list[dict]:
    static_dir = workspace.root / "STATIC"
    manifest_path = static_dir / "manifest.json"
    cache_path = static_dir / "scenes.json"
    if manifest_path.is_file():
        manifest = read_manifest(manifest_path)
        if not manifest.get("complete"):
            raise RuntimeError(f"STATIC manifest is incomplete: {manifest_path}")
        manifest_source = manifest.get("source_video")
        if not isinstance(manifest_source, dict):
            raise ValueError(f"STATIC manifest has no source identity: {manifest_path}")
        validate_source_identity(current_source, manifest_source)
        expected_config = {
            "scene_detector": STATIC_SCENE_DETECTOR,
            "screenshots_per_scene": STATIC_SCREENSHOTS_PER_SCENE,
        }
        mismatches = [
            key
            for key, expected in expected_config.items()
            if manifest.get(key) != expected
        ]
        if mismatches:
            raise ValueError(
                "STATIC configuration mismatch: " + ", ".join(mismatches)
            )
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"STATIC manifest exists but scenes.json is missing: {cache_path}"
            )
        require_cached = True
    else:
        if _has_unmanifested_static_data(static_dir):
            raise RuntimeError(
                "STATIC contains unmanifested data; refusing to overwrite "
                f"or clean it: {static_dir}"
            )
        if skip_extraction:
            raise FileNotFoundError(
                f"--skip-extraction requires complete STATIC data at {static_dir}"
            )
        require_cached = False

    creating_static = not manifest_path.is_file()
    try:
        with timer.stage("scene_detection_and_visuals"):
            scenes = load_or_extract_scenes(
                video_path,
                static_dir,
                screenshots_per_scene=STATIC_SCREENSHOTS_PER_SCENE,
                require_cached=require_cached,
            )
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Scene extraction produced no cache: {cache_path}"
            )
        if creating_static:
            write_manifest(
                manifest_path,
                {
                    "artifact_type": "static",
                    "created_at": _created_at(),
                    "source_video": current_source,
                    "scene_detector": STATIC_SCENE_DETECTOR,
                    "screenshots_per_scene": STATIC_SCREENSHOTS_PER_SCENE,
                    "scene_count": len(scenes),
                    "screenshot_count": sum(
                        len(scene.get("screenshots", []))
                        for scene in scenes
                    ),
                    "complete": True,
                },
            )
        return scenes
    except Exception:
        if creating_static:
            _cleanup_new_static(workspace, static_dir)
        raise


def analyze_video(
    video_path: Path,
    skip_transcription: bool = False,
    skip_extraction: bool = False,
    limit_visuals: int = None,
    skip_visual_analysis: bool = False,
    skip_json: bool = False,
    transcription_only: bool = False,
    skip_audio_analysis: bool = False,
    burn_scenes: bool = False,
    burn_transcript: bool = False,
    deployment: str = "",
    whisper_model: str = "",
    asr_backend: str = "whisper",
    nemo_model: str = "",
    vision_model: str = "",
    analytics_csv: Path = None,
    language: str = "cs",
    director: str = "",
    project: str = "",
    notes: str = "",
    recorded_date: str = "",
    output_root: Path = None,
    transcription_run: int = None,
    visual_run: int = None,
    combined_run: int = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> Path:
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    workspace = VideoWorkspace(
        video_path, resolve_output_root(video_path, output_root)
    )
    current_source = source_identity(video_path)

    if transcription_only:
        if skip_json:
            raise ValueError(
                "--transcription-only cannot be combined with --skip-json"
            )
        if visual_run is not None:
            raise ValueError(
                "--transcription-only cannot be combined with --visual-run"
            )
        if skip_visual_analysis:
            raise ValueError(
                "--transcription-only cannot be combined with "
                "--skip-visual-analysis"
            )
        if limit_visuals is not None:
            raise ValueError(
                "--transcription-only cannot be combined with --limit-visuals"
            )
        if burn_scenes or burn_transcript:
            raise ValueError(
                "--transcription-only cannot be combined with "
                "--burn-scenes/--burn-transcript"
            )

    if skip_json:
        combined = workspace.select_artifact(
            ArtifactKind.COMBINED,
            combined_run,
            expected_source_identity=current_source,
        )
        combined_path = combined.path / "combined_analysis.json"
        if not combined_path.is_file():
            raise FileNotFoundError(
                f"Selected combined artifact has no payload: {combined_path}"
            )
        ensure_scene_script(combined_path)
        if burn_scenes or burn_transcript:
            create_debug_video(
                video_path,
                combined_path,
                combined.path,
                burn_scenes,
                burn_transcript,
            )
        return combined_path
    if combined_run is not None:
        raise ValueError("--combined-run can only be used with --skip-json")

    workspace.ensure_tree()
    timer = StageTimer(on_stage=progress_callback)
    static_scenes = _load_static(
        workspace, video_path, current_source, skip_extraction, timer
    )

    if asr_backend == "nemo":
        resolved_asr_model = (
            nemo_model
            or get_env("NEMO_MODEL_NAME")
            or "nemo-unknown"
        )
    else:
        resolved_asr_model = (
            whisper_model
            or get_env("WHISPER_MODEL_NAME")
            or "whisper-unknown"
        )
    transcription_created = False
    if transcription_run is not None:
        transcription = workspace.select_artifact(
            ArtifactKind.TRANSCRIPTION,
            transcription_run,
            expected_source_identity=current_source,
        )
    elif skip_transcription:
        transcription = workspace.select_artifact(
            ArtifactKind.TRANSCRIPTION,
            expected_source_identity=current_source,
        )
    else:
        transcription = workspace.reserve_artifact(
            ArtifactKind.TRANSCRIPTION, resolved_asr_model
        )
        transcription_created = True

    transcript_path = transcription.path / "transcript.json"
    raw_transcript_path = transcription.path / "transcript_raw.json"
    dynamics_path = transcription.path / "audio_dynamics.json"
    try:
        if transcription_created:
            with timer.stage("transcription"):
                produced = transcribe_one(
                    video_path,
                    transcription.path,
                    language=(language or None),
                    asr_backend=asr_backend,
                )
            if Path(produced) != transcript_path:
                raise RuntimeError(
                    f"Transcription producer returned unexpected path: {produced}"
                )
        if not transcript_path.is_file():
            raise FileNotFoundError(
                f"Selected transcription has no payload: {transcript_path}"
            )
        if transcription_created and not raw_transcript_path.is_file():
            raise FileNotFoundError(
                "Transcription producer created no raw payload: "
                f"{raw_transcript_path}"
            )
        transcript_data = _read_json(transcript_path)
        if dynamics_path.is_file():
            dynamics = _read_json(dynamics_path)
        elif skip_audio_analysis:
            raise FileNotFoundError(
                "--skip-audio-analysis requires cached "
                f"audio_dynamics.json at {dynamics_path}"
            )
        else:
            with timer.stage("audio_dynamics"):
                dynamics = analyze_audio_dynamics(
                    video_path, transcript_data, static_scenes, dynamics_path
                )
            if not dynamics_path.is_file():
                raise FileNotFoundError(
                    f"Audio analysis produced no cache: {dynamics_path}"
                )

        if transcription_created:
            transcript_manifest = {
                "artifact_id": transcription.artifact_id,
                "artifact_type": transcription.kind.value,
                "created_at": _created_at(),
                "source_video": current_source,
                "asr_backend": asr_backend,
                "asr_model": resolved_asr_model,
                "language": language or "auto",
                "diarization_enabled": (
                    bool(get_env("HF_TOKEN"))
                    if asr_backend == "whisper"
                    else True
                ),
                "backend": f"{asr_backend}-server",
                "backend_configured": bool(
                    get_env("NEMO_URL")
                    if asr_backend == "nemo"
                    else get_env("WHISPER_URL")
                ),
                "complete": True,
            }
            write_manifest(
                transcription.path / "manifest.json", transcript_manifest
            )
            transcription = ArtifactRef(
                transcription.artifact_id,
                transcription.kind,
                transcription.path,
                transcript_manifest,
            )
    except Exception:
        if transcription_created:
            _remove_reserved_artifact(workspace, transcription)
        raise

    if transcription_only:
        logger.info("Transcription-only run complete: %s", transcript_path)
        return transcript_path

    resolved_vision_model = (
        vision_model or get_env("OLLAMA_VISION_MODEL") or "vision-mock"
    )
    visual_created = False
    if visual_run is not None:
        visual = workspace.select_artifact(
            ArtifactKind.VISUAL,
            visual_run,
            expected_source_identity=current_source,
        )
    elif skip_visual_analysis:
        visual = workspace.select_artifact(
            ArtifactKind.VISUAL,
            expected_source_identity=current_source,
        )
    else:
        visual = workspace.reserve_artifact(
            ArtifactKind.VISUAL, resolved_vision_model
        )
        visual_created = True

    try:
        with timer.stage("visual_description", total=len(static_scenes)):
            scene_data, visual_stats = describe_scenes(
                static_scenes,
                visual.path,
                limit_visuals=limit_visuals if visual_created else None,
                require_cached=not visual_created,
            )
        visual_cache = visual.path / "visual_descriptions_cache.json"
        if not visual_cache.is_file():
            raise FileNotFoundError(
                f"Visual analysis produced no cache: {visual_cache}"
            )
        if visual_created:
            ollama_url = get_env("OLLAMA_URL")
            visual_manifest = {
                "artifact_id": visual.artifact_id,
                "artifact_type": visual.kind.value,
                "created_at": _created_at(),
                "source_video": current_source,
                "vision_model": resolved_vision_model,
                "backend": visual_stats.get("backend", "unknown"),
                "backend_configured": bool(ollama_url),
                "limit_visuals": limit_visuals,
                "scene_count": len(scene_data),
                "scenes_analyzed": visual_stats.get("scenes_analyzed", 0),
                "screenshots_analyzed": visual_stats.get(
                    "screenshots_analyzed", 0
                ),
                "visual_seconds": visual_stats.get("visual_seconds", 0.0),
                "complete": True,
            }
            write_manifest(visual.path / "manifest.json", visual_manifest)
            visual = ArtifactRef(
                visual.artifact_id, visual.kind, visual.path, visual_manifest
            )
    except Exception:
        if visual_created:
            _remove_reserved_artifact(workspace, visual)
        raise

    transcription_label = _artifact_label(transcription, "asr_model", "whisper_model")
    visual_label = _artifact_label(visual, "vision_model")
    combined = workspace.reserve_artifact(
        ArtifactKind.COMBINED,
        f"{transcription_label} + {visual_label}",
    )
    output_path = combined.path / "combined_analysis.json"
    provenance = {
        "transcription_artifact_id": transcription.artifact_id,
        "transcription_artifact_path": str(transcription.path.resolve()),
        "visual_artifact_id": visual.artifact_id,
        "visual_artifact_path": str(visual.path.resolve()),
    }
    try:
        combined_json = expand_transcript_with_visuals(
            transcript_data, scene_data, dynamics, provenance
        )
        combined_json["metadata"] = build_video_metadata(
            video_path=video_path,
            vision_model=visual_label,
            whisper_model=transcription_label,
            director=director,
            project=project,
            notes=notes,
            recorded_date_override=recorded_date,
        )
        output_path.write_text(
            json.dumps(combined_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        ensure_scene_script(output_path)
        if burn_scenes or burn_transcript:
            with timer.stage("debug_video"):
                create_debug_video(
                    video_path,
                    output_path,
                    combined.path,
                    burn_scenes,
                    burn_transcript,
                )

        scenes = combined_json.get("scenes", [])
        visual_seconds = visual_stats.get("visual_seconds", 0.0)
        screenshots_analyzed = visual_stats.get("screenshots_analyzed", 0)
        screenshots_per_min = (
            round(screenshots_analyzed / (visual_seconds / 60.0), 2)
            if visual_seconds > 0
            else 0.0
        )
        timer.add_meta(
            video=video_path.name,
            output_path=str(output_path),
            transcription_artifact_id=transcription.artifact_id,
            visual_artifact_id=visual.artifact_id,
            combined_artifact_id=combined.artifact_id,
            num_scenes=len(scenes),
            num_words=sum(len(scene.get("words", [])) for scene in scenes),
            vision_model=visual_label,
            scenes_analyzed=visual_stats.get("scenes_analyzed", 0),
            screenshots_analyzed=screenshots_analyzed,
            visual_seconds=visual_seconds,
            screenshots_per_min=screenshots_per_min,
        )
        timer.save(
            combined.path / "analysis_report.json", output_path=output_path
        )
        timer.log_summary(logger)
        write_manifest(
            combined.path / "manifest.json",
            {
                "artifact_id": combined.artifact_id,
                "artifact_type": combined.kind.value,
                "created_at": _created_at(),
                "source_video": current_source,
                **provenance,
                "complete": True,
            },
        )
    except Exception:
        _remove_reserved_artifact(workspace, combined)
        raise

    flags = [f"--language {language or 'auto'}"]
    for enabled, flag in (
        (skip_transcription, "--skip-transcription"),
        (skip_extraction, "--skip-extraction"),
        (skip_visual_analysis, "--skip-visual-analysis"),
        (skip_audio_analysis, "--skip-audio-analysis"),
        (burn_scenes, "--burn-scenes"),
        (burn_transcript, "--burn-transcript"),
    ):
        if enabled:
            flags.append(flag)
    if limit_visuals is not None:
        flags.append(f"--limit-visuals {limit_visuals}")
    if transcription_run is not None:
        flags.append(f"--transcription-run {transcription_run:03d}")
    if visual_run is not None:
        flags.append(f"--visual-run {visual_run:03d}")

    env_csv = get_env("ANALYTICS_CSV")
    csv_path = (
        analytics_csv
        or (Path(env_csv) if env_csv else None)
        or Path(__file__).resolve().parent / "analysis_runs.csv"
    )
    timer.append_csv(
        csv_path,
        deployment=(
            deployment or get_env("DEPLOYMENT") or detect_machine()
        ),
        whisper_model=transcription_label,
        output_path=str(output_path),
        flags=" ".join(flags),
        metadata=combined_json.get("metadata"),
    )
    logger.info("Combined analysis saved to: %s", output_path)
    return output_path

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Create or select versioned video-analysis artifacts."
    )
    ap.add_argument("video", type=Path, help="Source video to analyze")
    ap.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the canonical '<video> edit' workspace root",
    )
    ap.add_argument(
        "--transcription-run",
        type=int,
        default=None,
        help="Select an existing complete transcription artifact ID",
    )
    ap.add_argument(
        "--visual-run",
        type=int,
        default=None,
        help="Select an existing complete visual-cache artifact ID",
    )
    ap.add_argument(
        "--combined-run",
        type=int,
        default=None,
        help="Select an existing combined artifact ID with --skip-json",
    )
    ap.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Select the newest compatible transcription artifact",
    )
    ap.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Require and reuse compatible STATIC scene extraction",
    )
    ap.add_argument(
        "--limit-visuals",
        type=int,
        default=None,
        help="Limit a new visual artifact to its first N scenes",
    )
    ap.add_argument(
        "--skip-visual-analysis",
        action="store_true",
        help="Select the newest compatible visual-cache artifact",
    )
    ap.add_argument(
        "--skip-json",
        action="store_true",
        help="Select an existing combined artifact without running producers",
    )
    ap.add_argument(
        "--transcription-only",
        action="store_true",
        help=(
            "Run scene extraction, ASR, and audio dynamics, then stop; "
            "skip visual analysis and the combined artifact"
        ),
    )
    ap.add_argument(
        "--burn-scenes",
        action="store_true",
        help="Write scene-number debug media beside the combined artifact",
    )
    ap.add_argument(
        "--burn-transcript",
        action="store_true",
        help="Write transcript debug media beside the combined artifact",
    )
    ap.add_argument(
        "--skip-audio-analysis",
        action="store_true",
        help="Require audio_dynamics.json in the selected transcription run",
    )
    ap.add_argument(
        "--language",
        type=str,
        default="cs",
        help="Whisper ISO 639-1 language (default cs; pass '' for auto)",
    )
    ap.add_argument(
        "--deployment",
        type=str,
        default="",
        help="Deployment label recorded for this versioned combined run",
    )
    ap.add_argument(
        "--whisper-model",
        type=str,
        default="",
        help="Model label for a new versioned transcription run",
    )
    ap.add_argument(
        "--asr-backend",
        choices=sorted(ASR_BACKENDS),
        default="whisper",
        help="Which ASR backend to use for a new transcription run (default: whisper)",
    )
    ap.add_argument(
        "--nemo-model",
        type=str,
        default="",
        help="Model label for a new versioned transcription run (nemo backend)",
    )
    ap.add_argument(
        "--analytics-csv",
        type=Path,
        default=None,
        help="Shared benchmark CSV receiving the versioned combined path",
    )
    ap.add_argument(
        "--director",
        type=str,
        default="",
        help="Director/shooter metadata in the combined artifact",
    )
    ap.add_argument(
        "--project",
        type=str,
        default="",
        help="Project/shoot metadata in the combined artifact",
    )
    ap.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-text shoot metadata in the combined artifact",
    )
    ap.add_argument(
        "--recorded-date",
        type=str,
        default="",
        help="Recording ISO date override in the combined artifact",
    )

    args = ap.parse_args()

    # A local Canary ASR server only needs to be running when this call will
    # actually reach it: --skip-json returns a previously-selected combined
    # artifact without transcribing anything, and --skip-transcription skips
    # the ASR stage outright. ensure_canary_ready itself further narrows
    # this to asr_backend == "nemo" with a local NEMO_URL -- see its
    # docstring in helpers/local_service_autostart.py.
    needs_asr = not args.skip_json and not args.skip_transcription
    with contextlib.ExitStack() as services:
        if needs_asr:
            services.enter_context(ensure_canary_ready(args.asr_backend))

        analyze_video(
            video_path=args.video.resolve(),
            skip_transcription=args.skip_transcription,
            skip_extraction=args.skip_extraction,
            limit_visuals=args.limit_visuals,
            skip_visual_analysis=args.skip_visual_analysis,
            skip_json=args.skip_json,
            transcription_only=args.transcription_only,
            skip_audio_analysis=args.skip_audio_analysis,
            burn_scenes=args.burn_scenes,
            burn_transcript=args.burn_transcript,
            deployment=args.deployment,
            whisper_model=args.whisper_model,
            asr_backend=args.asr_backend,
            nemo_model=args.nemo_model,
            analytics_csv=args.analytics_csv,
            language=args.language,
            director=args.director,
            project=args.project,
            notes=args.notes,
            recorded_date=args.recorded_date,
            output_root=args.output_root,
            transcription_run=args.transcription_run,
            visual_run=args.visual_run,
            combined_run=args.combined_run,
        )

if __name__ == "__main__":
    main()
