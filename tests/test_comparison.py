"""Offline tests for the shared cross-backend ranking + Markdown rendering."""

from geaptimes.experiment.comparison import (
    RANKING_METRICS,
    Comparison,
    rank_backends,
    render_ranking_markdown,
)


def _metrics(  # noqa: PLR0913 - mirrors the full metric set (4 ranking + pmae/prmse/n_points)
    mae: float,
    rmse: float,
    smape: float,
    ql: float,
    n_points: float = 308.0,
    pmae: float = 0.25,
    prmse: float = 0.30,
) -> dict[str, float]:
    return {
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "quantile_loss": ql,
        "pmae": pmae,
        "prmse": prmse,
        "n_points": n_points,
    }


def test_rank_backends_orders_by_rmse_then_mae() -> None:
    results = [
        ("timesfm", _metrics(80.0, 110.0, 20.0, 9.0)),
        ("bqml_arima_xreg", _metrics(77.0, 101.0, 18.0, 8.0)),
        ("automl", _metrics(70.0, 101.0, 17.0, 7.0)),  # ties bqml on rmse, wins on mae
    ]
    comp = rank_backends(results)
    assert comp.winner == "automl"
    assert [r["model"] for r in comp.ranking] == ["automl", "bqml_arima_xreg", "timesfm"]
    # every row carries the full metric set + display metrics, uniform shape regardless of source
    assert set(comp.ranking[0]) == {"model", *RANKING_METRICS, "pmae", "prmse", "n_points"}
    assert not comp.warnings  # equal n_points ⇒ no comparability warning


def test_rank_backends_missing_metric_defaults_to_inf() -> None:
    comp = rank_backends([("partial", {"mae": 1.0, "rmse": 2.0})])
    assert comp.ranking[0]["smape"] == float("inf")
    assert comp.ranking[0]["quantile_loss"] == float("inf")


def test_rank_backends_empty_has_no_winner() -> None:
    comp = rank_backends([])
    assert comp.winner == ""
    assert comp.ranking == []
    assert comp.warnings == []


def test_rank_backends_nan_metric_sorts_last() -> None:
    results = [
        ("nan_rmse", _metrics(10.0, float("nan"), 5.0, 1.0)),
        ("clean", _metrics(80.0, 110.0, 20.0, 9.0)),
    ]
    comp = rank_backends(results)
    # a present-but-NaN rmse must sink the model, not corrupt the sort order
    assert comp.winner == "clean"
    assert [r["model"] for r in comp.ranking] == ["clean", "nan_rmse"]


def test_rank_backends_warns_on_differing_point_counts() -> None:
    results = [
        ("timesfm", _metrics(80.0, 110.0, 20.0, 9.0, n_points=308.0)),
        ("automl", _metrics(70.0, 101.0, 17.0, 7.0, n_points=150.0)),
    ]
    comp = rank_backends(results)
    assert len(comp.warnings) == 1
    assert "differing point counts" in comp.warnings[0]
    assert "timesfm=308" in comp.warnings[0]
    assert "automl=150" in comp.warnings[0]


def test_rank_backends_no_parity_warning_ignores_zero_and_missing_counts() -> None:
    # a zero/missing n_points is not a genuine differing-denominator signal
    results = [
        ("timesfm", _metrics(80.0, 110.0, 20.0, 9.0, n_points=308.0)),
        ("empty", {"mae": 1.0, "rmse": 2.0, "smape": 3.0, "quantile_loss": 4.0, "n_points": 0.0}),
        ("nocount", {"mae": 1.0, "rmse": 2.0, "smape": 3.0, "quantile_loss": 4.0}),
    ]
    comp = rank_backends(results)
    assert comp.warnings == []


def test_render_ranking_markdown_flags_winner_and_shows_all_metrics() -> None:
    comp = Comparison(
        ranking=[
            {"model": "bqml_arima_xreg", **_metrics(77.0, 101.0, 18.0, 8.0)},
            {"model": "timesfm", **_metrics(80.0, 110.0, 20.0, 9.0)},
        ],
        winner="bqml_arima_xreg",
    )
    md = render_ranking_markdown(comp)
    lines = md.splitlines()
    assert lines[0] == (
        "| rank | model | mae | rmse | smape | quantile_loss | pmae | prmse | n_points |"
    )
    assert lines[1] == "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    assert lines[2].startswith("| 1 | bqml_arima_xreg (winner) |")
    assert "timesfm" in lines[3]
    assert "(winner)" not in lines[3]
    assert "101.0000" in lines[2]  # 4-decimal formatting for error metrics
    assert "0.2500" in lines[2]  # pmae rendered as a 4-decimal rate, not an integer
    assert lines[2].endswith("| 308 |")  # n_points as an integer, not 308.0000


def test_render_ranking_markdown_prints_warnings_under_table() -> None:
    comp = Comparison(
        ranking=[{"model": "timesfm", **_metrics(80.0, 110.0, 20.0, 9.0)}],
        winner="timesfm",
        warnings=["backends scored over differing point counts (timesfm=308, automl=150)"],
    )
    md = render_ranking_markdown(comp)
    assert "⚠️" in md
    assert "differing point counts" in md
    # the warning appears after the table body, not inside it
    assert md.index("timesfm") < md.index("differing point counts")


def test_render_ranking_markdown_renders_nan_metric_cell() -> None:
    comp = Comparison(
        ranking=[{"model": "broken", **_metrics(10.0, float("nan"), 5.0, 1.0)}],
        winner="broken",
    )
    md = render_ranking_markdown(comp)
    assert "nan" in md
