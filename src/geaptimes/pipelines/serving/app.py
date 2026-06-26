"""Flask app implementing the Vertex AI custom-container prediction contract.

Vertex sends prediction requests to ``AIP_PREDICT_ROUTE`` and health checks to ``AIP_HEALTH_ROUTE``
on ``AIP_HTTP_PORT`` (defaults ``/predict``, ``/health``, ``8080``). The request body is
``{"instances": [...]}`` and the response is ``{"predictions": [...]}``. The model loads lazily on
the first prediction, so health checks succeed immediately during cold start.

Run in the container with gunicorn::

    gunicorn -b 0.0.0.0:$AIP_HTTP_PORT geaptimes.pipelines.serving.app:app
"""

import os
from typing import TYPE_CHECKING

from flask import Flask, jsonify, request

from geaptimes.pipelines.serving.predictor import Predictor, TimesFMPredictor

if TYPE_CHECKING:  # pragma: no cover - typing only
    from werkzeug.wrappers import Response


def create_app(predictor: Predictor | None = None) -> Flask:
    """Build the Flask app, wiring the Vertex health + predict routes (predictor injectable)."""
    predictor = predictor or TimesFMPredictor.from_env()
    app = Flask(__name__)
    health_route = os.environ.get("AIP_HEALTH_ROUTE", "/health")
    predict_route = os.environ.get("AIP_PREDICT_ROUTE", "/predict")

    @app.get(health_route)
    def health() -> tuple[str, int]:
        return "", 200

    @app.post(predict_route)
    def predict() -> "Response":
        body = request.get_json(force=True, silent=True) or {}
        instances = body.get("instances", [])
        predictions = predictor.predict(instances)
        return jsonify({"predictions": predictions})

    return app


# Module-level WSGI callable for gunicorn (model loads lazily on first request).
app = create_app()


def main() -> None:  # pragma: no cover - container entrypoint
    """Run the dev server (gunicorn is preferred in the container)."""
    port = int(os.environ.get("AIP_HTTP_PORT", "8080"))
    app.run(host="0.0.0.0", port=port)  # noqa: S104 - bind all interfaces inside the container


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    main()
