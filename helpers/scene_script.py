"""Generate a lean, deterministic scene script from a combined_analysis.json.

Tagged-line format extending helpers/pack_transcripts.py's phrase convention
(see docs/superpowers/specs/2026-08-05-scene-script-and-lookup-design.md):

    SCENE <n> [<start>-<end>] <duration>s spk:<comma-separated speaker numbers> [env:<text>]
    D <start>-<end> S<n>: <text>
    V <timestamp> <description>
    A <start>-<end> <music|silence>
    P S<n> wps=<x> loud=<x> pacing=<x> rel=<x>x

No per-word timestamps, no logprobs, no nested per-image visual JSON. This is
the file the Master Editor browses first; helpers/scene_lookup.py fetches one
scene's full untouched JSON once it's a real cut candidate.

Input: <combined-run>/combined_analysis.json
Output: <combined-run>/scene_script.txt

Usage:
    python helpers/scene_script.py --combined <combined_analysis.json path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from helpers.pack_transcripts import group_into_phrases


def _speaker_tag(speaker_id) -> str:
    """Render a Scribe speaker_id like "speaker_2" as "S2"."""
    text = str(speaker_id) if speaker_id is not None else "?"
    if text.startswith("speaker_"):
        return f"S{text[len('speaker_'):]}"
    return f"S{text}"


def _speaker_num(speaker_id) -> str:
    """Render a Scribe speaker_id like "speaker_2" as "2" for the header's spk: list."""
    text = str(speaker_id)
    if text.startswith("speaker_"):
        return text[len("speaker_") :]
    return text


def render_scene_block(scene: dict, source_key: str | None = None) -> str:
    """Render a single scene dictionary as a tagged-line scene_script block.

    When `source_key` is given, the scene header is tagged
    `SCENE <source_key>:<n> [...]` instead of `SCENE <n> [...]` so scene
    references stay unambiguous once multiple sources' scripts are combined
    into one document (two different sources may both have a "scene 1").
    Omitting `source_key` (the default) preserves the original single-source
    format byte-for-byte.
    """
    scene_number = scene.get("scene_number", 0)
    start_time = scene.get("start_time", 0.0)
    end_time = scene.get("end_time", 0.0)
    duration = max(0.0, end_time - start_time)

    speaker_ids = scene.get("speakers", []) or []
    spk_str = ",".join(_speaker_num(s) for s in speaker_ids) if speaker_ids else "none"

    visual_desc = scene.get("visual_description", {}) or {}
    environment = (visual_desc.get("environment") or "").strip()
    env_part = f" env:{environment}" if environment else ""

    scene_ref = f"{source_key}:{scene_number}" if source_key is not None else f"{scene_number}"

    header = (
        f"SCENE {scene_ref} [{start_time:.2f}-{end_time:.2f}] "
        f"{duration:.2f}s spk:{spk_str}{env_part}"
    )

    events: list[tuple[float, int, str]] = []

    words = scene.get("words", []) or []
    phrases = group_into_phrases(words)
    for phrase in phrases:
        p_start = phrase.get("start", start_time)
        p_end = phrase.get("end", end_time)
        text = (phrase.get("text") or "").strip()
        speaker_id = phrase.get("speaker_id")
        tag = _speaker_tag(speaker_id)
        line = f"D {p_start:.2f}-{p_end:.2f} {tag}: {text}"
        events.append((p_start, 1, line))  # priority 1: dialogue after visual on tie

    screenshots = visual_desc.get("screenshots", []) or []
    for shot in screenshots:
        timestamp = shot.get("timestamp", start_time)
        description = (shot.get("description") or "").strip()
        line = f"V {timestamp:.2f} {description}"
        events.append((timestamp, 0, line))  # priority 0: visual wins exact ties

    audio_events = scene.get("audio_events", []) or []
    for ev in audio_events:
        ev_start = ev.get("start", start_time)
        ev_end = ev.get("end", end_time)
        ev_type = ev.get("type", "silence")
        line = f"A {ev_start:.2f}-{ev_end:.2f} {ev_type}"
        events.append((ev_start, 2, line))  # priority 2: after dialogue/visual on tie

    events.sort(key=lambda item: (item[0], item[1]))
    body_lines = [line for _, _, line in events]

    audio_dynamics = scene.get("audio_dynamics", {}) or {}
    ordered_speakers = list(speaker_ids)
    for speaker_id in audio_dynamics:
        if speaker_id not in ordered_speakers:
            ordered_speakers.append(speaker_id)

    pacing_lines = []
    for speaker_id in ordered_speakers:
        dynamics = audio_dynamics.get(speaker_id)
        if not dynamics:
            continue
        tag = _speaker_tag(speaker_id)
        pacing_lines.append(
            f"P {tag} wps={dynamics.get('words_per_second', 0.0):.2f} "
            f"loud={dynamics.get('loudness', '?')} "
            f"pacing={dynamics.get('pacing', '?')} "
            f"rel={dynamics.get('relative_loudness_ratio', 0.0):.2f}x"
        )

    return "\n".join([header, *body_lines, *pacing_lines])


def render_metadata_header(metadata: dict) -> str:
    """Render metadata header block at top of scene_script.txt if metadata exists."""
    if not metadata:
        return ""
    lines = []
    video = metadata.get("video_filename")
    duration = metadata.get("duration_seconds")
    res = metadata.get("resolution")
    fps = metadata.get("fps")
    if video:
        parts = []
        if duration is not None:
            parts.append(f"{duration:.2f}s")
        if res and fps is not None:
            parts.append(f"{res}@{fps:.1f}fps")
        elif res:
            parts.append(res)
        detail_str = f" [{', '.join(parts)}]" if parts else ""
        lines.append(f"# VIDEO: {video}{detail_str}")

    recorded = metadata.get("recorded_at")
    if recorded:
        lines.append(f"# RECORDED: {recorded}")

    director = metadata.get("director")
    if director:
        lines.append(f"# DIRECTOR: {director}")

    project = metadata.get("project")
    if project:
        lines.append(f"# PROJECT: {project}")

    notes = metadata.get("notes")
    if notes:
        lines.append(f"# NOTES: {notes}")

    return "\n".join(lines)


def generate(combined_path: Path, output: Path | None = None) -> Path:
    """Read one combined_analysis.json and write its scene_script.txt."""
    combined_path = Path(combined_path)
    data = json.loads(combined_path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {}) or {}
    meta_header = render_metadata_header(metadata)
    blocks = [render_scene_block(scene) for scene in data.get("scenes", [])]

    all_parts = []
    if meta_header:
        all_parts.append(meta_header)
    all_parts.extend(blocks)

    output_path = output if output is not None else combined_path.parent / "scene_script.txt"
    output_path.write_text("\n\n".join(all_parts) + "\n", encoding="utf-8")
    return output_path


def ensure(combined_path: Path) -> Path:
    """Return the combined run's scene_script.txt, generating it if missing."""
    combined_path = Path(combined_path)
    output_path = combined_path.parent / "scene_script.txt"
    if output_path.is_file():
        return output_path
    return generate(combined_path, output_path)


def build_scene_script(source_key: str, combined_path: Path) -> str:
    """Build the source-keyed scene-script content for one Combined artifact.

    Thin extension of `generate`'s rendering: reuses `render_scene_block`
    (tagged with `source_key`) and `render_metadata_header` rather than
    duplicating any tagged-line logic. Returns the content as a string —
    callers decide if/when to persist it (see `ensure_source_script`), since
    the multi-source Resolve flow combines several sources' scripts into one
    document the editor provider browses.

    A `# SOURCE: <source_key>` line is prepended ahead of any metadata header
    for readability when several sources' scripts are concatenated one after
    another; the authoritative, machine-parseable scoping is the
    `SCENE <source_key>:<n>` tag on every scene header, which stays correct
    even if blocks are reordered or filtered independently of this line.
    """
    combined_path = Path(combined_path)
    data = json.loads(combined_path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {}) or {}
    meta_header = render_metadata_header(metadata)
    blocks = [
        render_scene_block(scene, source_key=source_key) for scene in data.get("scenes", [])
    ]

    all_parts = [f"# SOURCE: {source_key}"]
    if meta_header:
        all_parts.append(meta_header)
    all_parts.extend(blocks)

    return "\n\n".join(all_parts) + "\n"


def _provenance_marker(source_key: str, combined_path: Path) -> str:
    """A single-line provenance header identifying the exact source_key and
    Combined artifact content a source-keyed scene_script.txt was built from.
    """
    digest = hashlib.sha256(Path(combined_path).read_bytes()).hexdigest()
    return f"# PROVENANCE: source_key={source_key} sha256={digest}"


def ensure_source_script(source_key: str, combined_path: Path) -> Path:
    """Return the source-keyed `scene_script.<source_key>.txt` beside
    `combined_path`.

    Extends the single-source `ensure()` pattern with provenance validation
    for the multi-source Resolve flow, where several `(source_key,
    combined_path)` pairs may each get their own scene script: if the file
    is missing, build and write it (via `build_scene_script`) behind a
    provenance marker recording `source_key` and a content hash of the
    Combined artifact it was built from. If the file already exists, its
    first line is compared against the current provenance marker; a mismatch
    (stale content, or a different Combined artifact written to the same
    path under the same source_key) triggers a rebuild rather than silent
    reuse.

    Deliberately writes `scene_script.<source_key>.txt`, not plain
    `scene_script.txt` (Important #6, final whole-branch review):
    `main.py`'s CLI path calls `ensure()`, which owns the plain
    `scene_script.txt` filename (no provenance line, no source-key-tagged
    scene headers) -- writing this function's differently-shaped,
    source-keyed content to that same path would mean a later ordinary CLI
    Master Editor session reads a file in a format its own skill doc
    doesn't describe, and vice versa. `ensure()` itself is untouched, and
    this function's own filename never collides with it, so a CLI-triggered
    `ensure()` call and a panel-triggered `ensure_source_script()` call on
    the same Combined artifact directory can always coexist.
    """
    combined_path = Path(combined_path)
    output_path = combined_path.parent / f"scene_script.{source_key}.txt"
    marker = _provenance_marker(source_key, combined_path)

    if output_path.is_file():
        existing = output_path.read_text(encoding="utf-8")
        existing_first_line = existing.splitlines()[0] if existing else ""
        if existing_first_line == marker:
            return output_path

    content = build_scene_script(source_key, combined_path)
    output_path.write_text(marker + "\n" + content, encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, required=True, help="Path to combined_analysis.json")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path (default: beside --combined)")
    args = parser.parse_args(argv)

    combined_path = args.combined.resolve()
    if not combined_path.is_file():
        print(f"no combined_analysis.json at {combined_path}", file=sys.stderr)
        return 1

    output_path = generate(combined_path, args.output)
    print(f"generated scene script -> {output_path}")
    return 0


def render_detailed_scene_block(scene: dict, source_key: str | None = None) -> str:
    """Serialize one requested scene without losing any Combined fields.

    Detail rounds are the authoritative view used after the lean scene
    script has identified a real candidate.  Minified JSON removes the
    previous indentation overhead while preserving unknown/new fields,
    exact floating-point values, word metadata, visual structures, and
    audio dynamics byte-for-structure through ``json.loads``.  The caller
    already renders ``source_key`` in the section heading, so it is kept as
    a compatibility argument and is deliberately not injected into the
    source scene dictionary.
    """
    if not isinstance(scene, dict):
        raise ValueError("scene must be a dictionary")
    return json.dumps(scene, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
