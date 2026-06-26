"""Design-of-Experiments matrix engine.

Expands a single :class:`~geaptimes.schemas.ExperimentConfig` into a grid of variants by taking the
cross-product of the sweep ``axes`` declared under ``doe.axes`` (dotted config path -> list of
values). Each variant is a fully re-validated config, so an invalid override (e.g. a negative
horizon) fails loudly through Pydantic rather than producing a silently-broken run.

The engine is pure and offline: it never touches the cloud. The downstream runner pairs each
:class:`DOEPoint` with every enabled model to form the run grid.
"""

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any

from geaptimes.schemas import ExperimentConfig


@dataclass(frozen=True)
class DOEPoint:
    """One point in the experiment grid: a re-validated config plus the overrides that made it."""

    config: ExperimentConfig
    overrides: dict[str, Any] = field(default_factory=dict)


def expand(cfg: ExperimentConfig) -> list[DOEPoint]:
    """Expand ``cfg`` into one :class:`DOEPoint` per cross-product combination of ``doe.axes``.

    Empty axes yield a single point with the base config (overrides ``{}``). The input ``cfg`` is
    never mutated; each variant is produced from a deep copy and re-validated.
    """
    axes = cfg.doe.axes
    # Variants don't re-sweep: drop the doe block so re-validation restores the empty default.
    base = cfg.model_dump()
    base.pop("doe", None)

    if not axes:
        return [DOEPoint(config=ExperimentConfig.model_validate(copy.deepcopy(base)), overrides={})]

    keys = list(axes.keys())
    value_lists = [axes[key] for key in keys]
    points: list[DOEPoint] = []
    for combo in itertools.product(*value_lists):
        overrides = dict(zip(keys, combo, strict=True))
        variant = copy.deepcopy(base)
        for path, value in overrides.items():
            _set_path(variant, path, value)
        points.append(
            DOEPoint(config=ExperimentConfig.model_validate(variant), overrides=overrides)
        )
    return points


def point_slug(overrides: dict[str, Any]) -> str:
    """Short, run-name-safe token for a point's overrides (``base`` when there are none)."""
    if not overrides:
        return "base"
    parts = [f"{path.split('.')[-1]}{_sanitize(value)}" for path, value in overrides.items()]
    return "-".join(parts)


def _set_path(data: dict[str, Any], dotted: str, value: Any) -> None:  # noqa: ANN401 - config value
    """Set ``data[a][b]...=value`` for a dotted path ``"a.b..."`` over nested dicts."""
    parts = dotted.split(".")
    node: Any = data
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            msg = f"DOE axis path {dotted!r} does not resolve to a config field"
            raise ValueError(msg)
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        msg = f"DOE axis path {dotted!r} does not resolve to a config field"
        raise ValueError(msg)
    node[parts[-1]] = value


def _sanitize(value: Any) -> str:  # noqa: ANN401 - config value
    """Lowercase, alphanumeric-only rendering of a sweep value for run names."""
    return "".join(ch for ch in str(value).lower() if ch.isalnum())
