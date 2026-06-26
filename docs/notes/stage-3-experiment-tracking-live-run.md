# Stage 3: experiment tracking — first live run + quirks

Captured 2026-06-26 during Stage 3. The first live execution of the Stage 2 forecasters, driven by
the `geaptimes.experiment` harness (DOE → runner → Vertex AI Experiments + GCS artifacts). Records
the outcomes and two non-obvious issues found by running for real.

## Live run outcome (`hybrid-vertex`, experiment `citibike-daily-baseline`)

Backtest over the held-out 14-day **TEST** window (25 series, 308 aligned points). Default DOE
(empty axes → one `base` point); enabled models TimesFM + BQML; AutoML wired but disabled.

| model | run name | MAE | RMSE |
|-------|----------|-----|------|
| TimesFM 2.5 (zero-shot, univariate, CPU) | `timesfm-base-20260626-013624` | 85.51 | 113.65 |
| BQML `ARIMA_PLUS_XREG` | `bqml-arima-xreg-base-20260626-013703` | **77.72** | **101.72** |

BQML with weather/calendar XReg beat zero-shot univariate TimesFM on this backtest (lower MAE/RMSE).
Not a definitive comparison — TimesFM was univariate and zero-shot; the standardized eval suite
(MAPE, quantile loss, per-series) lands in Stage 5.

Artifacts (resolved config + predictions) written per run under
`gs://geaptimes-hybrid-vertex/experiments/citibike-daily-baseline/<run>/{config.json,predictions.csv}`.
Params + metrics logged to each Vertex `ExperimentRun`. The BQML model
`hybrid-vertex.geaptimes.bqml_arima_xreg__top25_h14_t14_v14` carries `solution=geaptimes`.

## Bug found + fixed: BQML CREATE MODEL rejects resource labels in OPTIONS

`CREATE MODEL ... OPTIONS(labels=[("solution","geaptimes")])` fails with
`400 Expected type ARRAY<STRING>; found ARRAY<STRUCT<STRING, STRING>>; failed to set 'labels'`.
In BQML the `labels` model OPTION is **not** the table-style resource-label list — that key collides
with training-label semantics (typed `ARRAY<STRING>`). Table/view/dataset DDL accept
`labels=[("k","v")]`; BQML `CREATE MODEL` does not.

**Fix** (`models/bqml.py`): dropped `labels` from the CREATE MODEL OPTIONS; apply the
`solution=geaptimes` resource label to the model *after* creation via the BigQuery API
(`client.get_model` → set `.labels` → `client.update_model(model, ["labels"])`), through an injected
`model_labeler` seam so offline tests stay cloud-free. (`ALTER MODEL SET OPTIONS(labels=...)` is
undocumented for models — not relied on.)

## Environment quirk: pyOpenSSL + urllib3 breaks the Python GCS multipart upload

On this corp workstation, the Python `google-cloud-storage` multipart upload fails
nondeterministically with `ValueError: Context has already been used to create a Connection, it
cannot be mutated again` (urllib3's pyOpenSSL shim mutating `verify_mode` on an already-used
`SSLContext`). It is **not** a geapTimes code defect — Vertex experiment logging (init/start_run/
log_params/log_metrics) and all BigQuery/BQML calls succeed every time; only the GCS upload flakes,
and often only on the 2nd upload after intervening cloud calls.

Mitigations tried:
- `pyopenssl.extract_from_urllib3()` — helps once but `aiplatform`'s import re-injects pyOpenSSL.
- Blocking injection (`inject_into_urllib3 = noop`) — **breaks** the corp cert loader
  (`SSLContext has no attribute '_ctx'` in `use_certificate`); pyOpenSSL is required here.
- Fresh storage client per upload — still flakes.

**What works:** route artifact writes through the `gcloud storage cp` CLI (its own bundled TLS),
injected as a custom `artifact_sink` into `ExperimentTracker` — Vertex tracking stays on the Python
client. This is a *workstation* mitigation kept in the run harness, **not** in the package: the
package's default Python GCS sink is correct and works in normal environments. If the demo notebook
is run on an affected workstation, pass a `gcloud storage cp` sink the same way.

Reusable harness pattern (artifact sink):

```python
import subprocess, tempfile
def gcloud_sink(uri: str, text: str) -> None:
    suffix = ".json" if uri.endswith(".json") else ".csv"
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as fh:
        fh.write(text); local = fh.name
    subprocess.run(["gcloud", "storage", "cp", local, uri], check=True, capture_output=True)

tracker = ExperimentTracker(cfg, artifact_sink=gcloud_sink)
run_experiment(cfg, tracker=tracker)
```
