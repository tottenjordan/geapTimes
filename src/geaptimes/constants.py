"""Project-wide constants for geapTimes."""

SOLUTION = "geaptimes"

# Every label-capable GCP asset (BQ datasets/tables, GCS buckets, jobs, Vertex resources, ...)
# must carry this label. See CODE_STANDARDS.md.
RESOURCE_LABELS: dict[str, str] = {"solution": SOLUTION}


def bq_labels_option() -> str:
    """Render RESOURCE_LABELS as a BigQuery DDL ``OPTIONS(labels=[...])`` clause."""
    pairs = ", ".join(f'("{key}", "{value}")' for key, value in RESOURCE_LABELS.items())
    return f"OPTIONS(labels=[{pairs}])"


def bq_table_options(*, description: str | None = None) -> str:
    """Render a BigQuery DDL ``OPTIONS(...)``: required labels plus an optional description.

    The description carries a table's :func:`~geaptimes.naming.config_fingerprint_hash`, written at
    ``CREATE OR REPLACE`` time so the self-bootstrapping data-prep steps can read it back and skip a
    rebuild when the table is already present and current. With no description this is identical to
    :func:`bq_labels_option`.
    """
    pairs = ", ".join(f'("{key}", "{value}")' for key, value in RESOURCE_LABELS.items())
    options = [f"labels=[{pairs}]"]
    if description is not None:
        options.append(f'description="{description}"')
    return f"OPTIONS({', '.join(options)})"
