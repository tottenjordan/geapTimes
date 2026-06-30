"""Offline tests: the comparison pipeline compiles to a KFP spec with the expected graph shape.

These assert the *structure* of the compiled IR (which component executors are present per config)
without any cloud calls — the compile-time branching on enabled backends, serving mode, and
teardown is what these lock down.
"""

from pathlib import Path

import yaml

from geaptimes.pipelines.compile import compile_pipeline
from geaptimes.schemas import ExperimentConfig

BASE = {
    "project": {"id": "proj", "gcs_bucket": "bkt", "bq_dataset": "ds", "region": "us-central1"},
    "execution": {"experiment_name": "exp"},
}

ALL_THREE = [
    {"name": "timesfm", "params": {"type": "timesfm"}},
    {"name": "bqml_arima_xreg", "params": {"type": "bqml"}},
    {"name": "automl", "params": {"type": "automl"}},
]


def _cfg(models: list[dict], serving: dict | None = None) -> ExperimentConfig:
    over: dict = {**BASE, "models": models}
    if serving is not None:
        over["pipeline"] = {"serving": serving}
    return ExperimentConfig.model_validate(over)


def _executors(cfg: ExperimentConfig, tmp_path: Path) -> set[str]:
    out = compile_pipeline(cfg, tmp_path / "pipeline.yaml")
    spec = yaml.safe_load(Path(out).read_text(encoding="utf-8"))
    return set(spec["deploymentSpec"]["executors"].keys())


def _display_names(cfg: ExperimentConfig, tmp_path: Path) -> set[str]:
    """All task display names (``taskInfo.name``) across the root + nested (ExitHandler) DAGs."""
    out = compile_pipeline(cfg, tmp_path / "pipeline.yaml")
    spec = yaml.safe_load(Path(out).read_text(encoding="utf-8"))
    names: set[str] = set()

    def collect(dag: dict) -> None:
        for task in dag.get("tasks", {}).values():
            names.add(task.get("taskInfo", {}).get("name", ""))

    collect(spec["root"]["dag"])
    for comp in spec.get("components", {}).values():
        if "dag" in comp:
            collect(comp["dag"])
    return names


def test_endpoint_mode_transient_full_graph(tmp_path: Path) -> None:
    execs = _executors(_cfg(ALL_THREE), tmp_path)
    # self-bootstrapping data-prep front steps
    assert "exec-ensure-source" in execs
    assert "exec-ensure-prepped" in execs
    assert "exec-build-tables" in execs
    # hybrid GCPC serving: importer -> ModelUploadOp -> EndpointCreateOp + ModelDeployOp
    assert "exec-importer" in execs
    assert "exec-model-upload" in execs
    assert "exec-endpoint-create" in execs
    assert "exec-model-deploy" in execs
    # endpoint_predict stays custom (online predict + our frame mapping)
    assert "exec-endpoint-predict" in execs
    assert "exec-compare-backends" in execs
    # transient endpoint mode -> teardown wired as the ExitHandler exit task
    assert "exec-teardown-serving" in execs
    # batch path not present in endpoint mode
    assert "exec-batch-predict" not in execs
    # two in-process backends (bqml + automl) -> one train + one infer executor each
    assert len({e for e in execs if e.startswith("exec-train-backend")}) == 2
    assert len({e for e in execs if e.startswith("exec-infer-backend")}) == 2
    # one shared score-and-track per backend: bqml + automl + timesfm = 3
    assert len({e for e in execs if e.startswith("exec-score-and-track")}) == 3


def test_batch_mode_uses_batch_predict_and_no_endpoint(tmp_path: Path) -> None:
    execs = _executors(_cfg(ALL_THREE, serving={"mode": "batch"}), tmp_path)
    assert "exec-batch-predict" in execs
    # batch still uploads the model (importer -> ModelUploadOp) but creates/deploys no endpoint
    assert "exec-model-upload" in execs
    assert "exec-endpoint-create" not in execs
    assert "exec-model-deploy" not in execs
    assert "exec-endpoint-predict" not in execs
    # batch creates no endpoint, so there is nothing to tear down
    assert "exec-teardown-serving" not in execs


def test_keep_deployed_skips_teardown(tmp_path: Path) -> None:
    serving = {"mode": "endpoint", "keep_deployed": True}
    execs = _executors(_cfg(ALL_THREE, serving=serving), tmp_path)
    assert "exec-model-deploy" in execs
    assert "exec-endpoint-predict" in execs
    assert "exec-teardown-serving" not in execs


def test_reuse_branch_skips_serving_lifecycle(tmp_path: Path) -> None:
    # A warm endpoint resolved at submit time -> predict straight against it, skipping the whole
    # register -> create -> deploy lifecycle, and emit no teardown (we did not create it).
    out = compile_pipeline(
        _cfg(ALL_THREE),
        tmp_path / "pipeline.yaml",
        reused_endpoint="projects/x/locations/us-central1/endpoints/123",
        serving_fingerprint="feedface",
    )
    spec = yaml.safe_load(Path(out).read_text(encoding="utf-8"))
    execs = set(spec["deploymentSpec"]["executors"].keys())
    assert "exec-endpoint-predict-by-name" in execs
    assert "exec-endpoint-predict" not in execs  # the fresh-deploy predict is gone
    assert "exec-importer" not in execs
    assert "exec-model-upload" not in execs
    assert "exec-endpoint-create" not in execs
    assert "exec-model-deploy" not in execs
    assert "exec-teardown-serving" not in execs  # reuse => no teardown
    # the in-process backends + one shared scorer per backend are unchanged
    assert len({e for e in execs if e.startswith("exec-score-and-track")}) == 3


def test_fresh_path_stamps_serving_fingerprint_label(tmp_path: Path) -> None:
    # With a serving fingerprint supplied (reuse mode), the deployed model + endpoint carry the
    # fingerprint label so a later run can discover and reuse them.
    out = compile_pipeline(
        _cfg(ALL_THREE),
        tmp_path / "pipeline.yaml",
        serving_fingerprint="feedface",
    )
    assert "feedface" in Path(out).read_text(encoding="utf-8")


def test_timesfm_disabled_skips_serving(tmp_path: Path) -> None:
    models = [
        {"name": "timesfm", "enabled": False, "params": {"type": "timesfm"}},
        {"name": "bqml_arima_xreg", "params": {"type": "bqml"}},
    ]
    execs = _executors(_cfg(models), tmp_path)
    assert "exec-train-backend" in execs
    assert "exec-infer-backend" in execs
    assert "exec-score-and-track" in execs
    # no TimesFM => no serving lifecycle at all
    assert "exec-importer" not in execs
    assert "exec-model-upload" not in execs
    assert "exec-endpoint-create" not in execs
    assert "exec-model-deploy" not in execs
    assert "exec-endpoint-predict" not in execs
    assert "exec-batch-predict" not in execs
    assert "exec-teardown-serving" not in execs


def test_task_display_names_are_readable(tmp_path: Path) -> None:
    names = _display_names(_cfg(ALL_THREE), tmp_path)
    # Each in-process backend's train/infer/score tasks are disambiguated by model name (otherwise
    # they render as "train-backend"/"train-backend-2" etc. in the console).
    assert {
        "train:bqml_arima_xreg",
        "infer:bqml_arima_xreg",
        "score:bqml_arima_xreg",
        "train:automl",
        "infer:automl",
        "score:automl",
        "score:timesfm",
    } <= names
    assert {
        "ensure-source",
        "ensure-prepped",
        "build-tables",
        "timesfm:container-spec",
        "register-timesfm",
        "create-endpoint",
        "deploy-endpoint",
        "timesfm:endpoint-predict",
        "compare-backends",
        "teardown-serving",
    } <= names


def test_batch_mode_display_name(tmp_path: Path) -> None:
    names = _display_names(_cfg(ALL_THREE, serving={"mode": "batch"}), tmp_path)
    assert "timesfm:batch-predict" in names
    assert "timesfm:endpoint-predict" not in names


def _tasks_by_name(cfg: ExperimentConfig, tmp_path: Path) -> dict[str, dict]:
    """Every task across the root + nested (ExitHandler) DAGs, keyed by task name."""
    out = compile_pipeline(cfg, tmp_path / "pipeline.yaml")
    spec = yaml.safe_load(Path(out).read_text(encoding="utf-8"))
    tasks: dict[str, dict] = dict(spec["root"]["dag"]["tasks"])
    for comp in spec.get("components", {}).values():
        if "dag" in comp:
            tasks.update(comp["dag"]["tasks"])
    return tasks


def test_table_artifacts_wire_data_lineage_into_every_backend(tmp_path: Path) -> None:
    """The table Dataset artifacts form one source->prepped->{train,infer,timesfm} lineage chain.

    The self-bootstrapping front steps own the source/prepped artifacts: ensure-prepped derives its
    ``prepped`` from ensure-source's ``source`` and feeds it into build-tables AND the TimesFM
    serving branch; build-tables derives train/infer from prepped. The served TimesFM model reads
    its context from prepped at predict time, so the prepped edge lands on endpoint-predict (the
    GCPC upload/deploy ops are infra and have no data input) -- closing the once edge-less branch.

    Task ids here are the component-derived keys: the GCPC tasks are ``model-upload`` /
    ``endpoint-create`` / ``model-deploy`` (their readable display names are asserted separately).
    """
    tasks = _tasks_by_name(_cfg(ALL_THREE), tmp_path)

    def artifact_source(task: str, key: str) -> str:
        src = tasks[task]["inputs"]["artifacts"][key]["taskOutputArtifact"]
        return f"{src['producerTask']}.{src['outputArtifactKey']}"

    # data lineage
    assert artifact_source("ensure-prepped", "source") == "ensure-source.source"
    assert artifact_source("build-tables", "prepped") == "ensure-prepped.prepped"
    assert artifact_source("train-backend", "train") == "build-tables.train"
    assert artifact_source("infer-backend", "infer") == "build-tables.infer"
    assert artifact_source("endpoint-predict", "prepped") == "ensure-prepped.prepped"
    assert "ensure-source" in tasks["ensure-prepped"]["dependentTasks"]
    # GCPC serving lifecycle: importer -> model-upload; {endpoint-create, model-upload} -> deploy
    assert "importer" in tasks["model-upload"]["dependentTasks"]
    assert artifact_source("model-deploy", "model") == "model-upload.model"
    assert artifact_source("model-deploy", "endpoint") == "endpoint-create.endpoint"
    # custom endpoint-predict consumes the VertexEndpoint artifact and runs after the deploy
    assert artifact_source("endpoint-predict", "endpoint") == "endpoint-create.endpoint"
    assert "model-deploy" in tasks["endpoint-predict"]["dependentTasks"]


def _enable_cache(tasks: dict[str, dict], task: str) -> bool:
    return bool(tasks[task].get("cachingOptions", {}).get("enableCache"))


def test_serving_lifecycle_tasks_are_never_cached(tmp_path: Path) -> None:
    """The ephemeral endpoint lifecycle must re-run every submit, not cache across runs.

    A cached endpoint-create would hand back an endpoint a prior run's ExitHandler teardown already
    deleted (and a cached teardown would strand a live endpoint). The deterministic producers stay
    cacheable. Caching set False renders an *empty* ``cachingOptions`` (KFP omits false booleans);
    set True renders ``cachingOptions.enableCache``. Runtime semantics close the gap: the empty
    lifecycle options fall back to the always-off Vertex pipeline-level default (submit_pipeline
    submits ``enable_caching=False``), while the producers' explicit True survives and caches.
    """
    tasks = _tasks_by_name(_cfg(ALL_THREE), tmp_path)
    for task in ("endpoint-create", "model-deploy", "endpoint-predict", "teardown-serving"):
        assert not _enable_cache(tasks, task), f"{task} must not be cached"
    # deterministic producers remain cacheable
    for task in ("build-tables", "model-upload", "importer", "train-backend"):
        assert _enable_cache(tasks, task), f"{task} should stay cacheable"


def test_no_cache_config_makes_producers_uncacheable(tmp_path: Path) -> None:
    """With caching disabled in config, producers carry no explicit ``enableCache`` either.

    ``--no-cache`` (``with_caching_disabled``) sets ``pipeline.enable_caching=False``, so producers
    compile to an empty ``cachingOptions`` just like the lifecycle -- the whole graph then relies on
    the off pipeline-level default and re-runs from scratch. (Producers are cacheable only when the
    flag is on; see ``test_serving_lifecycle_tasks_are_never_cached``.)
    """
    cfg = ExperimentConfig.model_validate(
        {**BASE, "models": ALL_THREE, "pipeline": {"enable_caching": False}}
    )
    tasks = _tasks_by_name(cfg, tmp_path)
    # importer included: a KFP importer defaults to cacheable, so it must honor the off flag too.
    for task in ("build-tables", "model-upload", "importer", "train-backend", "endpoint-create"):
        assert not _enable_cache(tasks, task), f"{task} must be uncacheable when caching is off"


def test_richer_artifacts_metrics_and_markdown(tmp_path: Path) -> None:
    """score-and-track exposes a Metrics output; compare-backends a Markdown ranking (4A.4)."""
    out = compile_pipeline(_cfg(ALL_THREE), tmp_path / "pipeline.yaml")
    spec = yaml.safe_load(Path(out).read_text(encoding="utf-8"))
    components = spec["components"]

    def output_schemas(comp_key: str) -> set[str]:
        arts = components[comp_key]["outputDefinitions"]["artifacts"]
        return {a["artifactType"]["schemaTitle"] for a in arts.values()}

    assert "system.Metrics" in output_schemas("comp-score-and-track")
    assert "system.Markdown" in output_schemas("comp-compare-backends")


def test_component_machine_type_sizes_custom_executors(tmp_path: Path) -> None:
    """The configured component machine type becomes CPU/memory limits on the custom pods (4A.8).

    GCPC ops run in Google-managed images and are deliberately left unsized -- their machine is
    GCPC-controlled -- so only the custom component executors carry our resource limits.
    """
    out = compile_pipeline(_cfg(ALL_THREE), tmp_path / "pipeline.yaml")
    spec = yaml.safe_load(Path(out).read_text(encoding="utf-8"))
    execs = spec["deploymentSpec"]["executors"]
    # default e2-standard-2 -> 2 vCPU / 8 GB on a custom component pod
    res = execs["exec-build-tables"]["container"]["resources"]
    assert res["resourceCpuLimit"] == "2"
    assert res["resourceMemoryLimit"] == "8G"
    # GCPC serving ops keep their managed sizing (we set no resource limits on them)
    assert "resources" not in execs["exec-model-upload"]["container"]


def test_pipeline_name_and_config_param(tmp_path: Path) -> None:
    out = compile_pipeline(_cfg(ALL_THREE), tmp_path / "p.yaml")
    spec = yaml.safe_load(Path(out).read_text(encoding="utf-8"))
    assert spec["pipelineInfo"]["name"] == "geaptimes-comparison-top25-h14-t14-v14"
    params = spec["root"]["inputDefinitions"]["parameters"]
    assert "config_json" in params
