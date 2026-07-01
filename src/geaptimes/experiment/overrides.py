"""Config overrides for the AutoML backend, shared by the runner CLI and the pipeline submitter.

DOE-style transforms (``model_dump`` -> patch -> ``model_validate``, never in-place mutation) that
flip the AutoML backend on or off for a one-off run without editing YAML. They depend only on
:mod:`geaptimes.schemas`, so the in-process ``run_experiment`` CLI can reuse them without pulling in
the pipeline/``kfp`` stack that :mod:`geaptimes.pipelines.submit` imports.
"""

from geaptimes.schemas import ExperimentConfig


def with_automl_enabled(cfg: ExperimentConfig) -> ExperimentConfig:
    """Return a re-validated copy of *cfg* with the AutoML backend enabled.

    Flips ``enabled: true`` on the existing AutoML model, or appends a default AutoML model if the
    config has none. The input config is never mutated.
    """
    data = cfg.model_dump()
    models = data["models"]
    for model in models:
        if model["params"]["type"] == "automl":
            model["enabled"] = True
            break
    else:
        models.append({"name": "automl", "enabled": True, "params": {"type": "automl"}})
    return ExperimentConfig.model_validate(data)


def with_automl_disabled(cfg: ExperimentConfig) -> ExperimentConfig:
    """Return a re-validated copy of *cfg* with the AutoML backend disabled.

    The mirror of :func:`with_automl_enabled`: flips ``enabled: false`` on every AutoML model so a
    cheap run skips the long, billable AutoML training. A config with no AutoML model is returned
    unchanged. The input config is never mutated.
    """
    data = cfg.model_dump()
    for model in data["models"]:
        if model["params"]["type"] == "automl":
            model["enabled"] = False
    return ExperimentConfig.model_validate(data)
