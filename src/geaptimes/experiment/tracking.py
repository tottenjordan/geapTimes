"""Vertex AI experiment tracking + GCS artifact logging.

:class:`ExperimentTracker` wraps the ``aiplatform`` experiment API (``init`` / ``start_run`` /
``log_params`` / ``log_metrics`` / ``end_run``) behind a small, uniform surface, and writes run
artifacts (the resolved config + the prediction frame) to
``gs://<bucket>/experiments/<experiment>/<run>/``.

Both the Vertex SDK and the artifact sink are injected seams: the default sink uploads to GCS via
``google.cloud.storage``, but tests pass a recording sink and a fake ``aiplatform`` module so the
whole tracker exercises offline with no cloud calls.
"""

import json
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from geaptimes.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from geaptimes.schemas import ExperimentConfig

logger = get_logger(__name__)

# Sink signature: (gs_uri, text_payload) -> None.
ArtifactSink = Callable[[str, str], None]

_GCS_PREFIX = "gs://"


class ExperimentTracker:
    """Log params, metrics, and artifacts for a single experiment to Vertex AI + GCS."""

    def __init__(
        self,
        cfg: "ExperimentConfig",
        *,
        aiplatform: Any = None,  # noqa: ANN401 - external SDK module, injected for tests
        artifact_sink: ArtifactSink | None = None,
    ) -> None:
        self.cfg = cfg
        self._aiplatform = aiplatform
        self._artifact_sink = artifact_sink
        self._storage_client: Any = None

    # -- seams ------------------------------------------------------------------------------
    def _resolve_aiplatform(self) -> Any:  # noqa: ANN401 - external SDK module
        if self._aiplatform is None:
            from google.cloud import aiplatform  # noqa: PLC0415 - lazy cloud import

            self._aiplatform = aiplatform
        return self._aiplatform

    def _resolve_sink(self) -> ArtifactSink:
        return self._artifact_sink or self._default_sink

    def _default_sink(self, uri: str, text: str) -> None:
        """Upload ``text`` to a ``gs://`` URI via a lazily-built storage client."""
        if not uri.startswith(_GCS_PREFIX):
            msg = f"artifact URI must start with {_GCS_PREFIX!r}: {uri}"
            raise ValueError(msg)
        if self._storage_client is None:
            from google.cloud import storage  # noqa: PLC0415 - lazy cloud import

            self._storage_client = storage.Client(project=self.cfg.project.id)
        bucket_name, _, blob_path = uri[len(_GCS_PREFIX) :].partition("/")
        blob = self._storage_client.bucket(bucket_name).blob(blob_path)
        blob.upload_from_string(text)

    # -- run lifecycle ----------------------------------------------------------------------
    @contextmanager
    def run(self, run_name: str) -> Iterator["ExperimentTracker"]:
        """Open a Vertex ``ExperimentRun`` named ``run_name`` for the lifetime of the block."""
        aip = self._resolve_aiplatform()
        aip.init(
            project=self.cfg.project.id,
            location=self.cfg.project.region,
            experiment=self.cfg.execution.experiment_name,
            staging_bucket=f"{_GCS_PREFIX}{self.cfg.project.gcs_bucket}",
        )
        logger.info("starting experiment run %s", run_name)
        aip.start_run(run_name)
        try:
            yield self
        finally:
            aip.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        """Log run parameters (coerced to Vertex-supported primitives)."""
        self._resolve_aiplatform().log_params(_coerce_params(params))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log numeric run metrics, dropping non-finite values.

        Vertex's ExperimentRun metadata service rejects ``NaN``/``Infinity`` outright (400
        INVALID_ARGUMENT). A metric can legitimately be non-finite -- e.g. a point-only backend
        (AutoML under ``minimize-rmse``) emits no quantiles, so ``quantile_loss`` is ``NaN`` -- so
        we drop non-finite entries here rather than let the whole run fail. The value is still
        honestly reported downstream (the comparison table renders ``NaN`` and ``rank_backends`` is
        NaN-safe); this only governs what Vertex is asked to persist.
        """
        finite = {key: float(val) for key, val in metrics.items() if math.isfinite(val)}
        dropped = [key for key in metrics if key not in finite]
        if dropped:
            logger.warning(
                "dropping non-finite metric(s) from Vertex ExperimentRun (unsupported by the "
                "metadata service): %s",
                ", ".join(sorted(dropped)),
            )
        self._resolve_aiplatform().log_metrics(finite)

    def artifact_base(self, run_name: str) -> str:
        """The ``gs://`` prefix under which this run's artifacts are written."""
        project = self.cfg.project
        return (
            f"{_GCS_PREFIX}{project.gcs_bucket}/experiments/"
            f"{self.cfg.execution.experiment_name}/{run_name}"
        )

    def log_artifacts(
        self, run_name: str, *, config: "ExperimentConfig", predictions: "pd.DataFrame"
    ) -> str:
        """Write the resolved config (JSON) and predictions (CSV); return the artifact base URI."""
        base = self.artifact_base(run_name)
        sink = self._resolve_sink()
        sink(f"{base}/config.json", config.model_dump_json(indent=2))
        sink(f"{base}/predictions.csv", predictions.to_csv(index=False))
        logger.info("wrote run artifacts to %s", base)
        return base


def _coerce_params(params: dict[str, Any]) -> dict[str, Any]:
    """Coerce arbitrary param values to Vertex-supported primitives (str/int/float/bool)."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        out[key] = value if isinstance(value, str | int | float | bool) else json.dumps(value)
    return out
