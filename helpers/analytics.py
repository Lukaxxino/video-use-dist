"""Lightweight per-stage timing for the analyzer pipeline.

Wrap each stage of main.py in `timer.stage("name")`; at the end call
`timer.save(...)` to write a report and `timer.log_summary(logger)` to
print a table. No external dependencies — this is the measurement tool
for the "what is how fast" side of the descriptor work.

Accuracy metrics (WER, diarization quality vs a reference) are out of
scope here; this module only measures wall-clock speed.
"""

from __future__ import annotations

import csv
import inspect
import json
import platform
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


def detect_machine() -> str:
    """Human-readable machine label for the `deployment` column, e.g.
    "NVIDIA GeForce RTX 2060 (6 GB)". Falls back to OS/arch when no NVIDIA GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, timeout=10, stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        name, mem = [x.strip() for x in out.split(",")]
        gb = round(int(mem.split()[0]) / 1024)
        return f"{name} ({gb} GB)"
    except Exception:
        return f"{platform.system()} {platform.machine()} (no NVIDIA GPU)"

# Fixed column order for the benchmark CSV. New runs append one row.
CSV_COLUMNS = [
    "run_at",
    "deployment",
    "whisper_model",
    "vision_model",
    "flags",
    "num_scenes",
    "num_words",
    "transcription_s",
    "scene_detection_and_visuals_s",
    "visual_llm_s",
    "scenes_analyzed",
    "screenshots_analyzed",
    "screenshots_per_min",
    "audio_dynamics_s",
    "total_s",
    "output_path",
    "video_filename",
    "video_path",
    "file_size_bytes",
    "video_checksum_sha1",
    "duration_seconds",
    "resolution",
    "fps",
    "video_codec",
    "audio_codec",
    "recorded_at",
    "recorded_at_source",
    "director",
    "project",
    "notes",
]


def _callback_accepts_counts(callback: Optional[Callable]) -> bool:
    """Whether `callback` can receive the optional `completed`/`total`
    keyword arguments without raising `TypeError`.

    Detected structurally (via `inspect.signature`) rather than by trying
    and catching, so a callback that happens to raise `TypeError` for an
    unrelated reason is never misread as "doesn't accept counts". A plain
    `lambda name, status: ...` (no `**kwargs`, no `completed`/`total`
    parameters) reports `False`, which is exactly what keeps the existing
    two-positional-argument callback contract intact.
    """
    if callback is None:
        return False
    try:
        parameters = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        return False
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return True
    names = {
        parameter.name
        for parameter in parameters
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {"completed", "total"}.issubset(names)


class StageTimer:
    def __init__(self, on_stage: Optional[Callable[[str, str], None]] = None) -> None:
        self._t0 = time.perf_counter()
        self.stages: list[dict] = []
        self.meta: dict = {}
        self.on_stage = on_stage
        self._on_stage_accepts_counts = _callback_accepts_counts(on_stage)

    def _emit(self, name: str, status: str, completed: Optional[int], total: Optional[int]) -> None:
        if self.on_stage is None:
            return
        if self._on_stage_accepts_counts:
            self.on_stage(name, status, completed=completed, total=total)
        else:
            self.on_stage(name, status)

    @contextmanager
    def stage(self, name: str, *, total: Optional[int] = None):
        """Time a block of work and record it under `name`.

        `total` is an optional, truthfully-known unit count for the stage
        (e.g. the number of scenes about to be visually described). When
        given (and positive), the stage reports a genuine determinate
        0-of-total progress on start and total-of-total on completion --
        never a fabricated in-between fraction, since nothing in this
        context-manager contract observes real progress mid-stage. When no
        total is knowable, the stage reports indeterminate progress, which
        a caller supporting structured counts (see `_callback_accepts_counts`)
        can render as a spinner instead of a fake percentage.
        """
        known_total = total if isinstance(total, int) and not isinstance(total, bool) and total > 0 else None
        self._emit(name, "started", completed=(0 if known_total else None), total=known_total)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stages.append({"stage": name, "seconds": round(elapsed, 3)})
            self._emit(name, "complete", completed=known_total, total=known_total)

    def add_meta(self, **kwargs) -> None:
        """Attach context to the report (video name, scene count, model, ...)."""
        self.meta.update(kwargs)

    def report(self) -> dict:
        total = round(time.perf_counter() - self._t0, 3)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_seconds": total,
            "meta": self.meta,
            "stages": self.stages,
        }

    def save(self, path: Path, output_path: Path | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        report = self.report()
        if output_path is not None:
            report["output_path"] = str(output_path)
        path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def log_summary(self, logger) -> None:
        rep = self.report()
        logger.info("--- Timing summary ---")
        for s in rep["stages"]:
            logger.info(f"  {s['stage']:<28} {s['seconds']:>8.2f}s")
        logger.info(f"  {'TOTAL':<28} {rep['total_seconds']:>8.2f}s")

    def append_csv(self, path: Path, deployment: str, whisper_model: str, output_path: str, flags: str = "", metadata: dict = None) -> None:
        """Append one benchmark row to a CSV, creating it (with header) if new.

        Each run of the pipeline adds a row so runs can be compared over time.
        `flags` is the set of pipeline switches used for the run, for reproducibility.
        `metadata` is the same dict written to combined_analysis.json["metadata"]
        (see helpers/video_metadata.py) — its fields are flattened into their own
        columns so each benchmark row is also traceable to the source footage.
        """
        rep = self.report()
        stage_secs = {s["stage"]: s["seconds"] for s in rep["stages"]}
        metadata = metadata or {}
        row = {
            "run_at": rep["generated_at"],
            "deployment": deployment,
            "whisper_model": whisper_model,
            "vision_model": rep["meta"].get("vision_model", ""),
            "flags": flags,
            "num_scenes": rep["meta"].get("num_scenes", ""),
            "num_words": rep["meta"].get("num_words", ""),
            "transcription_s": stage_secs.get("transcription", ""),
            "scene_detection_and_visuals_s": stage_secs.get("scene_detection_and_visuals", ""),
            "visual_llm_s": rep["meta"].get("visual_seconds", ""),
            "scenes_analyzed": rep["meta"].get("scenes_analyzed", ""),
            "screenshots_analyzed": rep["meta"].get("screenshots_analyzed", ""),
            "screenshots_per_min": rep["meta"].get("screenshots_per_min", ""),
            "audio_dynamics_s": stage_secs.get("audio_dynamics", ""),
            "total_s": rep["total_seconds"],
            "output_path": output_path,
            "video_filename": metadata.get("video_filename") or "",
            "video_path": metadata.get("video_path") or "",
            "file_size_bytes": metadata.get("file_size_bytes") if metadata.get("file_size_bytes") is not None else "",
            "video_checksum_sha1": metadata.get("video_checksum_sha1") or "",
            "duration_seconds": metadata.get("duration_seconds") if metadata.get("duration_seconds") is not None else "",
            "resolution": metadata.get("resolution") or "",
            "fps": metadata.get("fps") if metadata.get("fps") is not None else "",
            "video_codec": metadata.get("video_codec") or "",
            "audio_codec": metadata.get("audio_codec") or "",
            "recorded_at": metadata.get("recorded_at") or "",
            "recorded_at_source": metadata.get("recorded_at_source") or "",
            "director": metadata.get("director") or "",
            "project": metadata.get("project") or "",
            "notes": metadata.get("notes") or "",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        if is_new:
            fieldnames = CSV_COLUMNS
            existing_rows = None
        else:
            # The file may already have extra hand-added columns (e.g. the
            # documented `flaws`/`description` free-text fields) beyond what
            # this code writes. Keep the file's existing column order and
            # only append any new CSV_COLUMNS entries at the end, so a new
            # row never shifts into a hand-added column's position.
            with path.open("r", newline="", encoding="utf-8") as f:
                all_rows = list(csv.reader(f))
            existing_header, existing_rows = all_rows[0], all_rows[1:]
            fieldnames = existing_header + [c for c in CSV_COLUMNS if c not in existing_header]
            if fieldnames == existing_header:
                existing_rows = None  # no new columns: plain append, no rewrite needed

        if existing_rows is not None:
            # New columns were introduced: rewrite the header and pad existing
            # data rows with empty values for the newly added columns, so every
            # row keeps the same field count as the header (RFC 4180).
            padded_rows = [
                row + [""] * (len(fieldnames) - len(row)) for row in existing_rows
            ]
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(fieldnames)
                writer.writerows(padded_rows)
            with path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
            return

        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
