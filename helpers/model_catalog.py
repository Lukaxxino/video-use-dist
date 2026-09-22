"""Typed model catalogue loaded from explicit environment configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from helpers.env_config import get_env


class ModelRole(str, Enum):
    AUDIO = "audio"
    VISION = "vision"
    EDITOR = "editor"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    role: ModelRole
    backend: str
    model: str
    url: str = ""
    command: str = ""


@dataclass(frozen=True)
class ModelCatalog:
    audio: tuple[ModelSpec, ...]
    vision: tuple[ModelSpec, ...]
    editor: tuple[ModelSpec, ...]

    def for_role(self, role: ModelRole) -> tuple[ModelSpec, ...]:
        return getattr(self, role.value)


_ID_PATTERN = re.compile(r"[a-z0-9_]+\Z")
_CONNECTION_FIELDS_BY_BACKEND = {
    "whisper": ("URL",),
    "nemo": ("URL",),
    "parakeet": ("URL",),
    "canary": ("URL",),
    "ollama": ("URL",),
    "claude_cli": ("COMMAND",),
    "litellm": ("URL",),
}


def load_model_catalog(getter: Callable[[str], str | None] = get_env) -> ModelCatalog:
    """Load the explicitly configured, role-scoped model catalogue."""
    entries_by_role: dict[ModelRole, tuple[ModelSpec, ...]] = {}
    assigned_roles: dict[str, ModelRole] = {}

    for role in ModelRole:
        ids = _configured_ids(role, getter)
        for model_id in ids:
            if model_id in assigned_roles:
                other_role = assigned_roles[model_id]
                if other_role is role:
                    raise ValueError(f"Model ID '{model_id}' is duplicated in the {role.value} role")
                raise ValueError(f"Model ID '{model_id}' is configured for multiple roles")
            assigned_roles[model_id] = role
        entries_by_role[role] = tuple()

    for role in ModelRole:
        entries_by_role[role] = tuple(
            _load_spec(model_id, role, getter)
            for model_id in _configured_ids(role, getter)
        )

    return ModelCatalog(
        audio=entries_by_role[ModelRole.AUDIO],
        vision=entries_by_role[ModelRole.VISION],
        editor=entries_by_role[ModelRole.EDITOR],
    )


def _configured_ids(role: ModelRole, getter: Callable[[str], str | None]) -> tuple[str, ...]:
    key = f"AI_{role.value.upper()}_MODELS"
    raw_ids = getter(key) or ""
    ids = tuple(item.strip().lower() for item in raw_ids.split(",") if item.strip())
    if not ids:
        raise ValueError(f"{key} must configure at least one {role.value} model")
    for model_id in ids:
        if not _ID_PATTERN.fullmatch(model_id):
            raise ValueError(f"invalid model ID '{model_id}'; expected [a-z0-9_]+")
    return ids


def _load_spec(
    model_id: str,
    role: ModelRole,
    getter: Callable[[str], str | None],
) -> ModelSpec:
    prefix = f"MODEL_{model_id.upper()}_"
    values = {field: (getter(prefix + field) or "").strip() for field in ("LABEL", "BACKEND", "MODEL", "URL", "COMMAND")}
    for field in ("LABEL", "BACKEND", "MODEL"):
        if not values[field]:
            raise ValueError(f"{prefix}{field} is required")
    for field in _CONNECTION_FIELDS_BY_BACKEND.get(values["BACKEND"], ()):
        if not values[field]:
            raise ValueError(f"{prefix}{field} is required for backend '{values['BACKEND']}'")
    return ModelSpec(
        id=model_id,
        label=values["LABEL"],
        role=role,
        backend=values["BACKEND"],
        model=values["MODEL"],
        url=values["URL"],
        command=values["COMMAND"],
    )
