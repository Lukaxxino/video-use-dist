"""Canonical output locations and versioned artifact storage for video analysis."""

import argparse
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

try:
    from helpers.env_config import get_env
except ModuleNotFoundError:  # Supports the documented `python helpers/storage.py` entry point.
    from env_config import get_env

try:
    from helpers.file_lock import exclusive_file_lock
except ModuleNotFoundError:  # Supports the documented `python helpers/storage.py` entry point.
    from file_lock import exclusive_file_lock


class ArtifactKind(str, Enum):
    TRANSCRIPTION = "transcription"
    VISUAL = "visual"
    COMBINED = "combined"
    XML = "xml"
    MEDIA_POOL_SELECTS = "media_pool_selects"


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: int
    kind: ArtifactKind
    path: Path
    manifest: dict


class ArtifactNotFoundError(LookupError):
    """Raised when no complete artifact can satisfy a selection request."""


ARTIFACT_BASES = {
    ArtifactKind.TRANSCRIPTION: Path("ACTIVE") / "TRANSCRIPTION",
    ArtifactKind.VISUAL: Path("ACTIVE") / "VISUAL CACHE",
    ArtifactKind.COMBINED: Path("COMBINED ANALYSIS"),
    ArtifactKind.XML: Path("XML"),
    ArtifactKind.MEDIA_POOL_SELECTS: Path("MEDIA POOL SELECTS"),
}
ARTIFACT_CLAIM_FILENAME = "claim.json"
ARTIFACT_CLAIM_SCHEMA_VERSION = 1


class VideoWorkspace:
    def __init__(self, video_path: Path, root: Path):
        self.video_path = Path(video_path)
        self.root = Path(root)

    def ensure_tree(self) -> None:
        (self.root / "STATIC" / "scenes").mkdir(parents=True, exist_ok=True)
        self._claims_base.mkdir(parents=True, exist_ok=True)
        for relative_path in ARTIFACT_BASES.values():
            (self.root / relative_path).mkdir(parents=True, exist_ok=True)

    def reserve_artifact(self, kind: ArtifactKind, label: str) -> ArtifactRef:
        sanitized_label = safe_label(label)
        if not sanitized_label:
            raise ValueError("Artifact label must be non-empty after sanitization")
        self.ensure_tree()
        base = self.root / ARTIFACT_BASES[kind]
        while True:
            artifact_id = self._claim_next_artifact_id()
            path = base / f"{artifact_id:03d} - {sanitized_label}"
            try:
                path.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            write_manifest(
                self._claims_base
                / f"{artifact_id:03d}"
                / ARTIFACT_CLAIM_FILENAME,
                {
                    "schema_version": ARTIFACT_CLAIM_SCHEMA_VERSION,
                    "artifact_id": artifact_id,
                    "artifact_type": kind.value,
                    "artifact_path": str(path.resolve()),
                },
            )
            return ArtifactRef(artifact_id, kind, path, {})

    def select_artifact(
        self,
        kind: ArtifactKind,
        artifact_id: int | None = None,
        expected_source_identity: dict | None = None,
    ) -> ArtifactRef:
        candidates = self._complete_artifacts(kind, expected_source_identity)
        if artifact_id is not None:
            candidates = [item for item in candidates if item.artifact_id == int(artifact_id)]
        if candidates:
            return max(candidates, key=lambda item: item.artifact_id)
        available = self._complete_artifacts(kind, None)
        if artifact_id is not None and expected_source_identity is not None:
            requested_id = int(artifact_id)
            unfiltered_match = next(
                (item for item in available if item.artifact_id == requested_id),
                None,
            )
            if unfiltered_match is not None:
                source = unfiltered_match.manifest.get("source_video")
                if isinstance(source, dict):
                    try:
                        validate_source_identity(expected_source_identity, source)
                    except ValueError as identity_error:
                        raise ArtifactNotFoundError(
                            f"{kind.value} artifact {requested_id} exists but was "
                            f"rejected due to a {identity_error}"
                        ) from identity_error
        available_ids = ", ".join(str(item.artifact_id) for item in available) or "none"
        requested = f" {artifact_id}" if artifact_id is not None else ""
        raise ArtifactNotFoundError(
            f"No complete {kind.value} artifact{requested}; available IDs: {available_ids}"
        )

    def list_complete(
        self,
        kind: ArtifactKind,
        expected_source_identity: dict | None = None,
    ) -> list[ArtifactRef]:
        """Public, newest-first listing of complete artifacts for `kind`."""
        return sorted(
            self._complete_artifacts(kind, expected_source_identity),
            key=lambda item: item.artifact_id,
            reverse=True,
        )

    def _next_artifact_id(self) -> int:
        highest_id = 0
        if self._claims_base.exists():
            for claim in self._claims_base.iterdir():
                if claim.is_dir() and claim.name.isdigit():
                    highest_id = max(highest_id, int(claim.name))
        for relative_path in ARTIFACT_BASES.values():
            base = self.root / relative_path
            if not base.exists():
                continue
            for child in base.iterdir():
                match = re.match(r"^(\d+)\s+-\s+", child.name)
                if child.is_dir() and match:
                    highest_id = max(highest_id, int(match.group(1)))
        return highest_id + 1

    @property
    def _claims_base(self) -> Path:
        return self.root / ".artifact-id-claims"

    def _claim_next_artifact_id(self) -> int:
        while True:
            artifact_id = self._next_artifact_id()
            try:
                (self._claims_base / f"{artifact_id:03d}").mkdir(exist_ok=False)
            except FileExistsError:
                continue
            return artifact_id

    def _complete_artifacts(
        self, kind: ArtifactKind, expected_source_identity: dict | None
    ) -> list[ArtifactRef]:
        base = self.root / ARTIFACT_BASES[kind]
        if not base.exists():
            return []
        artifacts = []
        for child in base.iterdir():
            match = re.match(r"^(\d+)\s+-\s+", child.name)
            manifest_path = child / "manifest.json"
            if not child.is_dir() or not match or not manifest_path.is_file():
                continue
            manifest = read_manifest(manifest_path)
            directory_id = int(match.group(1))
            if (
                not manifest.get("complete")
                or manifest.get("artifact_id") != directory_id
                or manifest.get("artifact_type") != kind.value
            ):
                continue
            if expected_source_identity is not None:
                source = manifest.get("source_video")
                if not isinstance(source, dict):
                    continue
                try:
                    validate_source_identity(expected_source_identity, source)
                except ValueError:
                    continue
            artifacts.append(ArtifactRef(directory_id, kind, child, manifest))
        return artifacts


def remove_complete_artifact(workspace_root: Path, kind: ArtifactKind, artifact_id: int) -> None:
    """Permanently delete one complete, claim-validated artifact directory
    (Task 6: `GenerationCoordinator.confirm_import_success` uses this to
    remove an approval-superseded `ArtifactKind.XML` candidate/strategy
    session once Resolve reports a verified import).

    Mirrors `main.py`'s own `_remove_reserved_artifact` cleanup-on-failure
    helper -- resolve the real directory, verify it is really the expected,
    claim-provenanced artifact directory, then `rmtree` it -- generalized to
    a caller removing a *complete* artifact well after creation rather than
    one that just failed mid-write. The claim marker under
    `.artifact-id-claims` is deliberately left in place: `artifact_id` is a
    permanent allocation that is never freed for reuse, exactly like every
    other artifact-directory removal already in this codebase.
    """
    workspace = VideoWorkspace(Path(), workspace_root)
    matches = [item for item in workspace.list_complete(kind) if item.artifact_id == artifact_id]
    if not matches:
        raise ArtifactNotFoundError(
            f"No complete {kind.value} artifact {artifact_id} to remove in {workspace_root}"
        )
    target = matches[0].path.resolve()
    base = (Path(workspace_root) / ARTIFACT_BASES[kind]).resolve()
    if target.parent != base:
        raise ValueError(f"Refusing to remove artifact outside its expected base {base}: {target}")
    validate_artifact_claim(workspace_root, ArtifactRef(artifact_id, kind, target, {}))
    shutil.rmtree(target)


def resolve_output_root(video_path: Path, override: Path | None = None) -> Path:
    video_path = Path(video_path).resolve()
    if override is not None:
        return Path(override).resolve()
    if (
        video_path.parent.name.casefold() == "tested videos"
        and video_path.parent.parent.name.casefold() == "test videos"
    ):
        return video_path.parent.parent / "Finished analysis" / f"{video_path.stem} edit"
    return video_path.parent / f"{video_path.stem} edit"


def resolve_ai_edits_root() -> Path:
    """Return the configured canonical root for Resolve edit workspaces."""
    configured = get_env("AI_EDITS_ROOT")
    if not configured:
        raise RuntimeError("AI_EDITS_ROOT is not configured")
    return Path(os.path.expandvars(configured)).resolve()


def _validate_source_identity_shape(identity: dict) -> None:
    if not isinstance(identity, dict):
        raise ValueError("Source identity must be a dictionary")
    required_types = {
        "filename": str,
        "size_bytes": int,
        "sha1": str,
    }
    for field, expected_type in required_types.items():
        value = identity.get(field)
        if isinstance(value, bool) or not isinstance(value, expected_type):
            raise ValueError(
                f"Source identity field {field!r} must be a {expected_type.__name__}"
            )


def source_key(identity: dict) -> str:
    """Return the short, content-derived key for one source identity."""
    _validate_source_identity_shape(identity)
    payload = f"{identity['filename']}\0{identity['size_bytes']}\0{identity['sha1']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _sorted_source_identities(identities: list[dict]) -> list[dict]:
    if not isinstance(identities, list):
        raise ValueError("Source identities must be a list")
    for identity in identities:
        _validate_source_identity_shape(identity)
    return sorted(identities, key=source_key)


def source_set_key(identities: list[dict]) -> str:
    """Return an order-independent short key for a collection of sources."""
    members = [source_key(identity) for identity in _sorted_source_identities(identities)]
    return hashlib.sha256("\0".join(members).encode("ascii")).hexdigest()[:8]


def _workspace_identity_error(reason: str) -> ValueError:
    return ValueError(f"Resolve workspace identity mismatch: {reason}")


def _validate_workspace_manifest(path: Path, expected: dict) -> None:
    try:
        actual = read_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise _workspace_identity_error(f"could not read {path.name}: {error}") from error

    for field in ("schema_version", "workspace_type"):
        if actual.get(field) != expected[field]:
            raise _workspace_identity_error(
                f"{field} is {actual.get(field)!r}, expected {expected[field]!r}"
            )

    if expected["workspace_type"] == "source":
        if actual.get("source_key") != expected["source_key"]:
            raise _workspace_identity_error("source_key differs")
        source = actual.get("source_identity")
        if not isinstance(source, dict):
            raise _workspace_identity_error("source_identity is missing or invalid")
        try:
            _validate_source_identity_shape(source)
            validate_source_identity(expected["source_identity"], source)
        except ValueError as error:
            raise _workspace_identity_error(str(error)) from error
        return

    if actual.get("source_set_key") != expected["source_set_key"]:
        raise _workspace_identity_error("source_set_key differs")
    if actual.get("initial_project_name") != expected["initial_project_name"]:
        raise _workspace_identity_error("initial_project_name differs")
    actual_identities = actual.get("source_identities")
    if not isinstance(actual_identities, list):
        raise _workspace_identity_error("source_identities is missing or invalid")
    try:
        actual_identities = _sorted_source_identities(actual_identities)
    except ValueError as error:
        raise _workspace_identity_error(str(error)) from error
    if len(actual_identities) != len(expected["source_identities"]):
        raise _workspace_identity_error("source_identities length differs")
    for expected_identity, actual_identity in zip(
        expected["source_identities"], actual_identities
    ):
        try:
            validate_source_identity(expected_identity, actual_identity)
        except ValueError as error:
            raise _workspace_identity_error(str(error)) from error


def _initialize_resolve_workspace(
    workspace_root: Path, expected_manifest: dict, workspace: VideoWorkspace | None = None
) -> None:
    """Atomically initialize or validate a Resolve workspace's root manifest."""
    workspace_root.mkdir(parents=True, exist_ok=True)
    lock_path = workspace_root / ".workspace-init.lock"
    with exclusive_file_lock(
        lock_path,
        timeout=10.0,
        timeout_error=RuntimeError,
        timeout_message=(
            f"Timed out waiting to initialize Resolve workspace: {workspace_root}"
        ),
    ):
        manifest_path = workspace_root / "workspace.json"
        if manifest_path.exists():
            _validate_workspace_manifest(manifest_path, expected_manifest)
        else:
            write_manifest(manifest_path, expected_manifest)
        if workspace is not None:
            workspace.ensure_tree()


def source_workspace(video_path: Path) -> VideoWorkspace:
    """Resolve and initialize the collision-safe analysis workspace for one clip."""
    video_path = Path(video_path)
    identity = source_identity(video_path)
    key = source_key(identity)
    root = resolve_ai_edits_root() / f"{safe_label(video_path.stem)}--{key}"
    workspace = VideoWorkspace(video_path, root)
    _initialize_resolve_workspace(
        root,
        {
            "schema_version": 1,
            "workspace_type": "source",
            "source_key": key,
            "source_identity": identity,
        },
        workspace,
    )
    return workspace


def resolve_shared_edit_workspace(project_name: str, identities: list[dict]) -> Path:
    """Resolve and initialize the shared workspace for a set of source clips."""
    initial_project_name = safe_label(project_name)
    source_identities = _sorted_source_identities(identities)
    key = source_set_key(source_identities)
    root = resolve_ai_edits_root() / f"{initial_project_name}--{key}"
    _initialize_resolve_workspace(
        root,
        {
            "schema_version": 1,
            "workspace_type": "shared_edit",
            "source_set_key": key,
            "source_identities": source_identities,
            "initial_project_name": initial_project_name,
        },
    )
    return root


def safe_label(value: str) -> str:
    label = re.sub(r'[<>:"/\\|?*]', '-', value)
    label = re.sub(r'-{2,}', '-', label)
    label = re.sub(r'\s{2,}', ' ', label)
    return label.strip(' .')


_TRANSIENT_PERMISSION_RETRY_ATTEMPTS = 10
_TRANSIENT_PERMISSION_RETRY_DELAY_SECONDS = 0.01


def _retry_transient_permission_error(action, *, attempts: int = _TRANSIENT_PERMISSION_RETRY_ATTEMPTS):
    """Retry `action` a small, bounded number of times on `PermissionError`
    before letting it propagate.

    Absorbs a narrow Windows-only race between `write_manifest`'s atomic
    `os.replace()` and a concurrent plain reader opening the same path
    (`read_manifest`, or any other `Path.open()` on a manifest): Windows can
    transiently deny one side with `WinError 5` ("Access is denied") when
    both land in the same instant, even though the rename itself is
    otherwise atomic and a reader that does get in never sees partial
    content. This never happens on POSIX, where `rename()` has no such
    transient state. Callers throughout this codebase already write
    manifests frequently and concurrently (heartbeats, progress events,
    artifact claims); a handful of ~10ms retries is cheap insurance against
    this one specific, already-observed Windows contention mode rather than
    a real, persistent permission problem -- which still surfaces (after
    this short grace period) as the original `PermissionError`.
    """
    last_error: PermissionError | None = None
    for attempt in range(attempts):
        try:
            return action()
        except PermissionError as error:
            last_error = error
            if attempt + 1 >= attempts:
                raise
            time.sleep(_TRANSIENT_PERMISSION_RETRY_DELAY_SECONDS)
    raise last_error  # pragma: no cover - loop above always returns or raises


def write_manifest(path: Path, payload: dict) -> None:
    """Atomically replace a UTF-8 JSON manifest without exposing partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
        ) as temporary:
            temp_path = Path(temporary.name)
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
        _retry_transient_permission_error(lambda: os.replace(temp_path, path))
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def read_manifest(path: Path) -> dict:
    path = Path(path)

    def _read() -> dict:
        with path.open(encoding="utf-8") as manifest_file:
            return json.load(manifest_file)

    payload = _retry_transient_permission_error(_read)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must contain a JSON object: {path}")
    return payload


def validate_artifact_claim(workspace_root: Path, artifact: ArtifactRef) -> dict:
    """Load and validate the immutable allocation claim for ``artifact``."""

    workspace_root = Path(workspace_root).resolve()
    claim_path = (
        workspace_root
        / ".artifact-id-claims"
        / f"{artifact.artifact_id:03d}"
        / ARTIFACT_CLAIM_FILENAME
    )
    if claim_path.resolve() != claim_path:
        raise ValueError(f"Artifact claim path is a link or junction: {claim_path}")
    if not claim_path.is_file():
        raise ValueError(f"Artifact claim is missing: {claim_path}")

    claim = read_manifest(claim_path)
    if (
        type(claim.get("schema_version")) is not int
        or claim["schema_version"] != ARTIFACT_CLAIM_SCHEMA_VERSION
    ):
        raise ValueError(f"Artifact claim schema is invalid: {claim_path}")
    if (
        type(claim.get("artifact_id")) is not int
        or claim["artifact_id"] != artifact.artifact_id
    ):
        raise ValueError(f"Artifact claim ID does not match artifact: {artifact.path}")
    if claim.get("artifact_type") != artifact.kind.value:
        raise ValueError(f"Artifact claim type does not match artifact: {artifact.path}")
    if claim.get("artifact_path") != str(artifact.path.resolve()):
        raise ValueError(f"Artifact claim path does not match artifact: {artifact.path}")
    return claim


def source_identity(video_path: Path) -> dict:
    path = Path(video_path).resolve()
    digest = hashlib.sha1()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "filename": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha1": digest.hexdigest(),
    }


def validate_source_identity(expected: dict, actual: dict) -> None:
    # "path" is intentionally excluded: it is the absolute filesystem path
    # recorded at artifact-creation time and is not portable across clones,
    # moved repos, or worktrees. Content identity is filename + size + sha1.
    differing_keys = [
        key for key in ("filename", "size_bytes", "sha1")
        if expected.get(key) != actual.get(key)
    ]
    if differing_keys:
        raise ValueError("Source identity mismatch: " + ", ".join(differing_keys))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("resolve", "select-combined", "create-xml"):
        command = commands.add_parser(name)
        command.add_argument("video")
        command.add_argument("--output-root", type=Path)
        if name == "select-combined":
            command.add_argument("--run", type=int)
        if name == "create-xml":
            command.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    root = resolve_output_root(args.video, args.output_root)
    if args.command == "resolve":
        print(root)
        return 0

    workspace = VideoWorkspace(Path(args.video), root)
    workspace.ensure_tree()
    if args.command == "select-combined":
        artifact = workspace.select_artifact(ArtifactKind.COMBINED, args.run)
        print((artifact.path / "combined_analysis.json").resolve())
        return 0

    artifact = workspace.reserve_artifact(ArtifactKind.XML, args.name)
    print(artifact.path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
