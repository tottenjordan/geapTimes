"""Project-wide constants for geapTimes."""

SOLUTION = "geaptimes"

# Every label-capable GCP asset (BQ datasets/tables, GCS buckets, jobs, Vertex resources, ...)
# must carry this label. See CODE_STANDARDS.md.
RESOURCE_LABELS: dict[str, str] = {"solution": SOLUTION}


def bq_labels_option() -> str:
    """Render RESOURCE_LABELS as a BigQuery DDL ``OPTIONS(labels=[...])`` clause."""
    pairs = ", ".join(f'("{key}", "{value}")' for key, value in RESOURCE_LABELS.items())
    return f"OPTIONS(labels=[{pairs}])"
