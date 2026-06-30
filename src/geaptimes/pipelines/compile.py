"""Compile the comparison pipeline to a KFP v2 pipeline spec (pure — no cloud calls).

``compile_pipeline`` turns a typed :class:`~geaptimes.schemas.ExperimentConfig` into the IR YAML
that ``aiplatform.PipelineJob`` consumes. It performs no authentication, BigQuery, or Vertex calls,
so it is part of the offline gate; the live submission lives in :mod:`geaptimes.pipelines.submit`.
"""

from typing import TYPE_CHECKING

from geaptimes.pipelines.pipeline import build_pipeline
from geaptimes.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    from geaptimes.schemas import ExperimentConfig

logger = get_logger(__name__)


def compile_pipeline(
    cfg: "ExperimentConfig",
    out_path: "str | Path",
    *,
    reused_endpoint: str = "",
    serving_fingerprint: str = "",
) -> str:
    """Compile the comparison pipeline for *cfg* to *out_path* (KFP IR YAML); return the path.

    ``reused_endpoint`` / ``serving_fingerprint`` are compile-time inputs for the warm-endpoint
    reuse branch, resolved live at submit time and threaded to :func:`build_pipeline` (they are not
    pipeline parameters). Both default empty -- the normal fresh-deploy path.
    """
    from kfp import compiler  # noqa: PLC0415 - keep kfp import lazy for fast offline imports

    pipeline_func = build_pipeline(
        cfg, reused_endpoint=reused_endpoint, serving_fingerprint=serving_fingerprint
    )
    out = str(out_path)
    compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=out)
    logger.info("compiled comparison pipeline to %s", out)
    return out
