"""Offline tests for the shared cross-backend ranking + Markdown rendering."""

from geaptimes.experiment.comparison import (
    RANKING_METRICS,
    Comparison,
    rank_backends,
    render_ranking_markdown,
)


def _metrics(mae: float, rmse: float, smape: float, ql: float) -> dict[str, float]:
    return {"mae": mae, "rmse": rmse, "smape": smape, "quantile_loss": ql}


def test_rank_backends_orders_by_rmse_then_mae() -> None:
    results = [
        ("timesfm", _metrics(80.0, 110.0, 20.0, 9.0)),
        ("bqml_arima_xreg", _metrics(77.0, 101.0, 18.0, 8.0)),
        ("automl", _metrics(70.0, 101.0, 17.0, 7.0)),  # ties bqml on rmse, wins on mae
    ]
    comp = rank_backends(results)
    assert comp.winner == "automl"
    assert [r["model"] for r in comp.ranking] == ["automl", "bqml_arima_xreg", "timesfm"]
    # every row carries the full metric set, uniform shape regardless of source
    assert set(comp.ranking[0]) == {"model", *RANKING_METRICS}


def test_rank_backends_missing_metric_defaults_to_inf() -> None:
    comp = rank_backends([("partial", {"mae": 1.0, "rmse": 2.0})])
    assert comp.ranking[0]["smape"] == float("inf")
    assert comp.ranking[0]["quantile_loss"] == float("inf")


def test_rank_backends_empty_has_no_winner() -> None:
    comp = rank_backends([])
    assert comp.winner == ""
    assert comp.ranking == []


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
    assert lines[0] == "| rank | model | mae | rmse | smape | quantile_loss |"
    assert lines[1] == "| --- | --- | --- | --- | --- | --- |"
    assert lines[2].startswith("| 1 | bqml_arima_xreg (winner) |")
    assert "timesfm" in lines[3]
    assert "(winner)" not in lines[3]
    assert "101.0000" in lines[2]  # 4-decimal formatting
