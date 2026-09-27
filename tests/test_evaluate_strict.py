"""Unit tests for the ``normalize_strict`` evaluation harness helpers.

These are pure functions over an already-scored DataFrame: no model loading,
no gazetteer, no I/O. They summarize the production path
(``cali_address.service.normalize_strict``), which is stricter than
``AddressNormalizer.normalize_batch`` and is what actually produces
``estado == "OK"``.

Coordinates are secondary: correctness here is judged by cadastral identity
(``manzana`` / ``numero_predial_nacional`` against ground truth) and by
text agreement with IDESC's own normalization, never by distance.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cali_address.evaluate import motivo_breakdown, strict_summary  # noqa: E402


def _row(
    dataset="dsA",
    estado="OK",
    has_gt=True,
    correct_manzana=None,
    correct_predial=None,
    raw_top1_correct_manzana=None,
    text_agree_idesc=None,
    dist_m=None,
):
    return {
        "dataset": dataset,
        "estado": estado,
        "has_gt": has_gt,
        "correct_manzana": correct_manzana,
        "correct_predial": correct_predial,
        "raw_top1_correct_manzana": raw_top1_correct_manzana,
        "text_agree_idesc": text_agree_idesc,
        "dist_m": dist_m,
    }


def _get(summary: pd.DataFrame, dataset: str, estado: str) -> pd.Series:
    matches = summary[(summary["dataset"] == dataset) & (summary["estado"] == estado)]
    assert len(matches) == 1, f"expected exactly one row for ({dataset}, {estado}), got {len(matches)}"
    return matches.iloc[0]


# ---------------------------------------------------------------------------
# strict_summary: manzana / predial precision
# ---------------------------------------------------------------------------
def test_manzana_precision_is_share_of_correct_among_ok_rows_with_gt():
    rows = [
        _row(estado="OK", has_gt=True, correct_manzana=True, correct_predial=True),
        _row(estado="OK", has_gt=True, correct_manzana=True, correct_predial=False),
        _row(estado="OK", has_gt=True, correct_manzana=False, correct_predial=False),
        _row(estado="SIN_MATCH", has_gt=True, correct_manzana=np.nan, correct_predial=np.nan),
    ]
    summary = strict_summary(pd.DataFrame(rows))
    ok = _get(summary, "dsA", "OK")
    assert ok["manzana_precision"] == pytest.approx(2 / 3)
    assert ok["predial_precision"] == pytest.approx(1 / 3)


def test_manzana_precision_is_nan_for_non_ok_estado():
    rows = [
        _row(estado="OK", has_gt=True, correct_manzana=True, correct_predial=True),
        _row(estado="SIN_MATCH", has_gt=True, correct_manzana=np.nan, correct_predial=np.nan),
    ]
    summary = strict_summary(pd.DataFrame(rows))
    sin_match = _get(summary, "dsA", "SIN_MATCH")
    assert pd.isna(sin_match["manzana_precision"])
    assert pd.isna(sin_match["predial_precision"])


# ---------------------------------------------------------------------------
# strict_summary: rows with a hidden predial / manzana (shared addresses)
# ---------------------------------------------------------------------------
def _shared_row(predial, manzana, correct_manzana, correct_predial, **kw):
    row = _row(estado="OK", has_gt=True, correct_manzana=correct_manzana,
               correct_predial=correct_predial, **kw)
    row["numero_predial_nacional"] = predial
    row["manzana"] = manzana
    return row


def test_hidden_predial_and_manzana_are_counted_separately_not_as_wrong():
    rows = [
        _shared_row("P1", "M1", True, True),
        _shared_row(None, "M2", True, np.nan),     # shared: predial hidden
        _shared_row(None, None, np.nan, np.nan),   # shared across manzanas
        _shared_row("P4", "M4", False, False),
    ]
    ok = _get(strict_summary(pd.DataFrame(rows)), "dsA", "OK")
    assert ok["n"] == 4
    assert ok["n_ok_sin_predial"] == 2
    assert ok["n_ok_sin_manzana"] == 1
    assert ok["manzana_precision"] == pytest.approx(2 / 3)
    assert ok["predial_precision"] == pytest.approx(1 / 2)


def test_all_hidden_predial_group_has_nan_precision_without_raising():
    rows = [_shared_row(None, None, np.nan, np.nan) for _ in range(3)]
    ok = _get(strict_summary(pd.DataFrame(rows)), "dsA", "OK")
    assert ok["n_ok_sin_predial"] == 3
    assert ok["n_ok_sin_manzana"] == 3
    assert pd.isna(ok["manzana_precision"])
    assert pd.isna(ok["predial_precision"])


def test_hidden_counts_treat_nan_and_blank_as_missing():
    rows = [
        _shared_row(np.nan, "M", np.nan, np.nan),
        _shared_row("", "", np.nan, np.nan),
        _shared_row("P", "M", True, True),
    ]
    ok = _get(strict_summary(pd.DataFrame(rows)), "dsA", "OK")
    assert ok["n_ok_sin_predial"] == 2
    assert ok["n_ok_sin_manzana"] == 1


def test_hidden_counts_are_zero_when_columns_absent_and_for_non_ok():
    rows = [_row(estado="OK", correct_manzana=True, correct_predial=True),
            _row(estado="SIN_MATCH")]
    summary = strict_summary(pd.DataFrame(rows))
    assert _get(summary, "dsA", "OK")["n_ok_sin_predial"] == 0
    assert _get(summary, "dsA", "SIN_MATCH")["n_ok_sin_manzana"] == 0


def test_nivel_precision_breakdown_counts_only_ok_rows():
    from cali_address.evaluate import nivel_precision_breakdown
    df = pd.DataFrame({
        "estado": ["OK", "OK", "OK", "SIN_MATCH", "OK"],
        "nivel_precision": ["predio", "predio", "manzana", "predio", None],
    })
    out = nivel_precision_breakdown(df)
    counts = dict(zip(out["nivel_precision"], out["n"]))
    assert counts == {"predio": 2, "manzana": 1, "(vacio)": 1}


def test_nivel_precision_breakdown_on_empty_or_missing_column():
    from cali_address.evaluate import nivel_precision_breakdown
    assert nivel_precision_breakdown(pd.DataFrame()).empty
    assert nivel_precision_breakdown(pd.DataFrame({"estado": ["OK"]})).empty


# ---------------------------------------------------------------------------
# strict_summary: lost_correct
# ---------------------------------------------------------------------------
def test_lost_correct_counts_only_non_ok_rows_with_raw_top1_correct_manzana():
    rows = [
        _row(estado="OK", has_gt=True, correct_manzana=True, raw_top1_correct_manzana=True),
        _row(estado="SIN_MATCH", has_gt=True, correct_manzana=np.nan, raw_top1_correct_manzana=True),
        _row(estado="SIN_MATCH", has_gt=True, correct_manzana=np.nan, raw_top1_correct_manzana=False),
        _row(estado="NO_PARSEABLE", has_gt=True, correct_manzana=np.nan, raw_top1_correct_manzana=np.nan),
    ]
    summary = strict_summary(pd.DataFrame(rows))
    ok = _get(summary, "dsA", "OK")
    sin_match = _get(summary, "dsA", "SIN_MATCH")
    no_parseable = _get(summary, "dsA", "NO_PARSEABLE")
    assert ok["lost_correct"] == 0
    assert sin_match["lost_correct"] == 1
    assert sin_match["n_with_gt"] == 2
    assert sin_match["lost_correct_share"] == pytest.approx(0.5)
    assert no_parseable["lost_correct"] == 0


# ---------------------------------------------------------------------------
# strict_summary: edge cases
# ---------------------------------------------------------------------------
def test_strict_summary_on_empty_frame_returns_empty_result_without_raising():
    columns = [
        "dataset", "estado", "has_gt", "correct_manzana", "correct_predial",
        "raw_top1_correct_manzana", "text_agree_idesc", "dist_m",
    ]
    result = strict_summary(pd.DataFrame(columns=columns))
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_strict_summary_on_totally_empty_frame_does_not_raise():
    result = strict_summary(pd.DataFrame())
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_dataset_with_no_ground_truth_has_nan_precision_but_counts_rows():
    rows = [
        _row(dataset="dsB", estado="OK", has_gt=False, correct_manzana=np.nan, correct_predial=np.nan)
        for _ in range(3)
    ]
    summary = strict_summary(pd.DataFrame(rows))
    ok = _get(summary, "dsB", "OK")
    assert ok["n"] == 3
    assert ok["n_with_gt"] == 0
    assert pd.isna(ok["manzana_precision"])
    assert pd.isna(ok["predial_precision"])
    assert pd.isna(ok["lost_correct_share"])


def test_pooled_all_row_equals_union_of_datasets():
    rows_a = [
        _row(dataset="dsA", estado="OK", has_gt=True, correct_manzana=True, correct_predial=True),
        _row(dataset="dsA", estado="OK", has_gt=True, correct_manzana=False, correct_predial=False),
    ]
    rows_b = [
        _row(dataset="dsB", estado="OK", has_gt=True, correct_manzana=True, correct_predial=False),
    ]
    summary = strict_summary(pd.DataFrame(rows_a + rows_b))
    pooled = _get(summary, "ALL", "OK")
    assert pooled["n"] == 3
    assert pooled["n_with_gt"] == 3
    assert pooled["manzana_precision"] == pytest.approx(2 / 3)
    assert pooled["predial_precision"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# strict_summary: idesc_text_agreement (coordinate-free)
# ---------------------------------------------------------------------------
def test_idesc_text_agreement_is_share_among_ok_rows_with_an_idesc_answer():
    rows = [
        _row(estado="OK", text_agree_idesc=True),
        _row(estado="OK", text_agree_idesc=False),
        _row(estado="OK", text_agree_idesc=True),
        _row(estado="OK", text_agree_idesc=np.nan),  # no IDESC answer: excluded from denominator
    ]
    summary = strict_summary(pd.DataFrame(rows))
    ok = _get(summary, "dsA", "OK")
    assert ok["idesc_text_agreement"] == pytest.approx(2 / 3)


def test_idesc_text_agreement_is_nan_when_no_idesc_answers_present():
    rows = [_row(estado="OK", text_agree_idesc=np.nan) for _ in range(2)]
    summary = strict_summary(pd.DataFrame(rows))
    ok = _get(summary, "dsA", "OK")
    assert pd.isna(ok["idesc_text_agreement"])


def test_idesc_text_agreement_is_nan_for_non_ok_estado():
    rows = [_row(estado="SIN_MATCH", has_gt=True, text_agree_idesc=True)]
    summary = strict_summary(pd.DataFrame(rows))
    sin_match = _get(summary, "dsA", "SIN_MATCH")
    assert pd.isna(sin_match["idesc_text_agreement"])


def test_strict_summary_no_longer_reports_distance_buckets():
    rows = [_row(estado="OK", has_gt=True, correct_manzana=True, correct_predial=True, dist_m=10.0)]
    summary = strict_summary(pd.DataFrame(rows))
    for forbidden in ("within_50m", "within_100m", "within_500m", "median_dist_m"):
        assert forbidden not in summary.columns


# ---------------------------------------------------------------------------
# motivo_breakdown: prefix extraction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "motivo, expected_prefix",
    [
        ("confianza 0.61 < umbral 0.73", "confianza"),
        ("letra de via E != -", "letra de via"),
        ("candidatos fuera de barrio ...", "candidatos fuera de"),
        ("", "(vacio)"),
    ],
)
def test_motivo_breakdown_extracts_expected_prefix(motivo, expected_prefix):
    df = pd.DataFrame([{"motivo": motivo, "raw_top1_correct_manzana": np.nan}])
    result = motivo_breakdown(df)
    assert len(result) == 1
    assert result.iloc[0]["motivo_prefix"] == expected_prefix
    assert result.iloc[0]["n"] == 1


def test_motivo_breakdown_crosses_prefix_with_raw_top1_correct_manzana():
    rows = [
        {"motivo": "confianza 0.61 < umbral 0.73", "raw_top1_correct_manzana": True},
        {"motivo": "confianza 0.50 < umbral 0.73", "raw_top1_correct_manzana": False},
        {"motivo": "confianza 0.40 < umbral 0.73", "raw_top1_correct_manzana": np.nan},
    ]
    result = motivo_breakdown(pd.DataFrame(rows))
    row = result[result["motivo_prefix"] == "confianza"].iloc[0]
    assert row["n"] == 3
    assert row["raw_top1_correct_true"] == 1
    assert row["raw_top1_correct_false"] == 1
    assert row["raw_top1_correct_nan"] == 1


def test_motivo_breakdown_on_empty_frame_returns_empty_without_raising():
    result = motivo_breakdown(pd.DataFrame(columns=["motivo", "raw_top1_correct_manzana"]))
    assert isinstance(result, pd.DataFrame)
    assert result.empty

    result_totally_empty = motivo_breakdown(pd.DataFrame())
    assert isinstance(result_totally_empty, pd.DataFrame)
    assert result_totally_empty.empty
