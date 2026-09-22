"""CLI tool to fetch full, unmodified JSON for specified scenes on demand.

Used by the Master Editor agent during Pass 1-4 editorial work to pull exact
details (full words[] with timestamps/logprobs, visual_description screenshots,
audio_dynamics) for candidate cut scenes without loading the entire
combined_analysis.json file into the LLM prompt.

Usage:
    python helpers/scene_lookup.py --combined <path> --scenes 12,14,15-18 [-o output.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


def parse_scene_selector(selector_str: str) -> list[int]:
    """Parse comma-separated scene numbers and inclusive ranges like '12,14,15-18'."""
    seen: list[int] = []
    seen_set: set[int] = set()

    for part in selector_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                start, end = int(start_str), int(end_str)
            except ValueError as error:
                raise ValueError(f"invalid range or integer in '{part}'") from error
            if end < start:
                raise ValueError(f"invalid range '{part}': end is before start")
            numbers = range(start, end + 1)
        else:
            try:
                numbers = [int(part)]
            except ValueError as error:
                raise ValueError(f"invalid integer '{part}'") from error
        for number in numbers:
            if number not in seen_set:
                seen_set.add(number)
                seen.append(number)
    return seen


def _index_scenes_by_number(data: dict) -> dict[int, dict]:
    """Index one loaded combined_analysis.json payload's scenes by
    scene_number. Shared by `lookup_scenes` and `SceneLookup` so both extract
    scenes identically rather than duplicating this logic.
    """
    return {scene["scene_number"]: scene for scene in data.get("scenes", []) or []}


def lookup_scenes(combined_path: Path, scene_numbers: list[int]) -> list[dict]:
    """Return the exact, unmodified scene objects for the requested scene
    numbers, in the order requested.
    """
    data = json.loads(Path(combined_path).read_text(encoding="utf-8"))
    by_number = _index_scenes_by_number(data)
    missing = [n for n in scene_numbers if n not in by_number]
    if missing:
        available = ", ".join(str(n) for n in sorted(by_number)) or "none"
        raise ValueError(
            f"scene(s) {missing} not found in {combined_path}; "
            f"available scene numbers: {available}"
        )
    return [by_number[n] for n in scene_numbers]


class AmbiguousSceneReference(ValueError):
    """Raised when a bare (source_key-less) scene number resolves to more
    than one indexed source in a `SceneLookup`. Callers (ultimately an editor
    provider surfacing a scene reference from a proposal) must disambiguate
    by passing an explicit `source_key` instead of getting a silently-picked
    result.

    Subclasses `ValueError`, not bare `Exception` (Bundled Minor #1, final
    whole-branch review): `panel_server._run_coordinator` only catches
    `ValueError`/`KeyError`/`TaskCoordinatorError`/`GenerationCoordinatorError`
    and a handful of other specific types, never bare `Exception`, so this
    would otherwise surface as an unhandled 500 instead of a clean 400 if it
    ever reached a route.
    """

    def __init__(self, scene_number: int, source_keys: list[str]):
        self.scene_number = scene_number
        self.source_keys = list(source_keys)
        joined = ", ".join(sorted(self.source_keys))
        super().__init__(
            f"scene {scene_number} is ambiguous: it exists in multiple "
            f"sources [{joined}]; specify source_key explicitly"
        )


class SceneLookup:
    """Indexes scenes from multiple selected Combined artifacts by
    `(source_key, scene_number)`, loading each Combined exactly once at
    construction, and serves exact, JSON-serializable scene copies via
    `fetch`.

    Built for the Resolve panel's multi-source generation flow: one instance
    is constructed per task with the set of selected `(source_key,
    combined_path)` pairs, then reused for every scene reference the editor
    provider makes during that task.
    """

    def __init__(self, sources: Iterable[tuple[str, Path]]):
        self._scenes_by_key: dict[tuple[str, int], dict] = {}
        self._sources_by_number: dict[int, list[str]] = {}
        self._known_sources: list[str] = []

        for source_key, combined_path in sources:
            if source_key in self._known_sources:
                raise ValueError(f"duplicate source_key {source_key!r} in SceneLookup sources")
            self._known_sources.append(source_key)
            data = json.loads(Path(combined_path).read_text(encoding="utf-8"))
            by_number = _index_scenes_by_number(data)
            for scene_number, scene in by_number.items():
                # Round-trip through JSON so cached state is an independent,
                # JSON-serializable copy that a caller mutating a fetched
                # result can never corrupt.
                self._scenes_by_key[(source_key, scene_number)] = json.loads(
                    json.dumps(scene)
                )
                self._sources_by_number.setdefault(scene_number, []).append(source_key)

    def scene_bounds(self, source_key: str) -> list[tuple[int, float, float]]:
        """Return `(scene_number, start_time, end_time)` for every scene
        indexed under `source_key`, sorted by `start_time`.

        Lets a caller (the EDL validator's word-boundary check) map an
        arbitrary `[start, end)` time window back to the scene(s) it
        overlaps without already knowing which scene number(s) that window
        falls in — `fetch` requires scene numbers up front, so this is the
        piece it doesn't otherwise expose.
        """
        if source_key not in self._known_sources:
            known = ", ".join(sorted(set(self._known_sources))) or "none"
            raise ValueError(f"unknown source '{source_key}'; known sources: {known}")

        bounds = [
            (scene_number, float(scene["start_time"]), float(scene["end_time"]))
            for (sk, scene_number), scene in self._scenes_by_key.items()
            if sk == source_key
        ]
        bounds.sort(key=lambda item: item[1])
        return bounds

    def fetch(self, source_key: str | None, scene_numbers: list[int]) -> list[dict]:
        """Return exact scene copies for `scene_numbers` under `source_key`,
        each with a `source_key` field injected; the rest of each dict stays
        byte-equivalent, structurally, to what `lookup_scenes` would return
        for that scene.

        If `source_key` is omitted (`None`), each scene number is resolved
        across every indexed source: a number that matches exactly one
        source resolves to it (source_key inferred rather than supplied); a
        number matching more than one source raises
        `AmbiguousSceneReference`.
        """
        results: list[dict] = []
        for scene_number in scene_numbers:
            if source_key is not None:
                resolved_source = source_key
                if source_key not in self._known_sources:
                    known = ", ".join(sorted(set(self._known_sources))) or "none"
                    raise ValueError(
                        f"unknown source '{source_key}'; known sources: {known}"
                    )
                key = (source_key, scene_number)
                if key not in self._scenes_by_key:
                    available = ", ".join(
                        str(n)
                        for (sk, n) in sorted(self._scenes_by_key)
                        if sk == source_key
                    ) or "none"
                    raise ValueError(
                        f"scene(s) [{scene_number}] not found for source "
                        f"'{source_key}'; available scene numbers: {available}"
                    )
            else:
                matches = self._sources_by_number.get(scene_number, [])
                if not matches:
                    available = ", ".join(str(n) for n in sorted(self._sources_by_number)) or "none"
                    raise ValueError(
                        f"scene(s) [{scene_number}] not found in any indexed "
                        f"source; available scene numbers: {available}"
                    )
                if len(matches) > 1:
                    raise AmbiguousSceneReference(scene_number, matches)
                resolved_source = matches[0]

            scene_copy = json.loads(
                json.dumps(self._scenes_by_key[(resolved_source, scene_number)])
            )
            scene_copy["source_key"] = resolved_source
            results.append(scene_copy)

        return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, required=True, help="Path to combined_analysis.json")
    parser.add_argument(
        "--scenes", type=str, required=True,
        help="Comma-separated scene numbers/ranges, e.g. 12,14,15-18",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write JSON to this file instead of stdout")
    args = parser.parse_args(argv)

    combined_path = args.combined.resolve()
    if not combined_path.is_file():
        print(f"no combined_analysis.json at {combined_path}", file=sys.stderr)
        return 1

    try:
        scene_numbers = parse_scene_selector(args.scenes)
        scenes = lookup_scenes(combined_path, scene_numbers)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    payload = json.dumps(scenes, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
        print(f"wrote {len(scenes)} scene(s) -> {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
