"""Jede Zahl aus Texten, Hilfen und README ist hier belegt (gemessen am 2026-09-22, Toleranzen fangen Rundung ab). `analyse()` ist deterministisch (et.fit intern mit seed=0); die Aufwands- und
Bias-Varianz-Experimente sind Mittel über feste Seeds - keine Zufallsstreuung zwischen Testläufen."""

import functools

import numpy as np
import pytest

import et_algorithm as et
import et_constants as C
import et_evaluation as ev
import et_scenario as S

PRESET = {"standard": "🌲 Extra Trees (Standard)", "boot": "🎲 Mit Bootstrap", "cheap": "⚡ Viele billige Bäume", "all": "🌳 Random Forest (mtry = alle)", "reg": "📈 Regression"}


@functools.lru_cache(maxsize=None)
def _preset(key):
    p = C.PRESETS[PRESET[key]]
    return ev.analyse(p["task"], p["criterion"], p["leaf"], p["n_trees"], p["mtry"], p["bootstrap"], p["n"], p["n_noise"], p["label_noise"], p["seed"])


def _help(key, *needles):
    text = C.PRESET_HELP[PRESET[key]]
    for n in needles:
        assert n in text, (key, n)


# --- Preset-Hilfen --------------------------------------------------------------------------------------------------------------------------------

def test_standard_preset():
    a = _preset("standard")
    assert not a.forest.bootstrap and a.forest.mtry == 3 and a.verdict == "cheaper_similar"
    assert (a.test["error"], a.exhaustive_test["error"]) == pytest.approx((0.1667, 0.1639), abs=0.0005)
    assert a.checked == 29346 and a.would_check == 795714
    _help("standard", "16,7 %", "16,4 %", "3,7 %")


def test_bootstrap_preset():
    a = _preset("boot")
    assert a.forest.bootstrap and a.test["error"] == pytest.approx(0.1611, abs=0.0005)
    assert a.oob["error"] == pytest.approx(0.2345, abs=0.0005) and a.oob["n"] == 840
    _help("boot", "23,5 %", "16,1 %")


def test_cheap_trees_preset():
    a = _preset("cheap")
    assert len(a.forest.trees) == 60 and not a.forest.bootstrap
    assert a.test["error"] == pytest.approx(0.1611, abs=0.0005) and a.checked == 59754
    ratio = a.checked / _preset("standard").would_check                                             # gegen Random Forests Aufwand bei 30 Bäumen
    assert ratio == pytest.approx(0.0751, abs=0.0005)
    _help("cheap", "60 statt 30", "7,5 %", "16,1 %", "16,7 %", "16,4 %")


def test_all_features_preset():
    a = _preset("all")
    assert a.forest.mtry == 11 and (a.test["error"], a.exhaustive_test["error"]) == pytest.approx((0.1778, 0.1667), abs=0.0005)
    _help("all", "mtry = 11", "17,8 %", "16,7 %")


def test_regression_preset():
    a = _preset("reg")
    assert a.task == "reg" and not a.forest.bootstrap and a.forest.mtry == 3
    assert (a.test["error"], a.exhaustive_test["error"]) == pytest.approx((13.4555, 11.6427), abs=0.005)
    ratio = a.checked / a.would_check
    assert ratio == pytest.approx(0.0791, abs=0.0005)
    _help("reg", "13,5 min", "11,6 min", "7,9 %")


def test_every_preset_is_a_valid_setting():
    for name, p in C.PRESETS.items():
        assert p["task"] in C.TASKS and p["criterion"] in C.CRITERIA[p["task"]] and C.LEAF_MIN <= p["leaf"] <= C.LEAF_MAX and C.N_TREES_MIN <= p["n_trees"] <= C.N_TREES_MAX
        d = C.N_BASE + p["n_noise"]
        assert 1 <= p["mtry"] <= d and isinstance(p["bootstrap"], bool) and C.N_MIN <= p["n"] <= C.N_MAX and 0 <= p["fx"] < d and 0 <= p["fy"] < d and name in C.PRESET_HELP


# --- Aufwands-Formeln (exakt) -----------------------------------------------------------------------------------------------------------------------

def test_split_count_formulas_on_the_default_forest():
    a = _preset("standard")
    forest = a.forest
    manual_checked = sum(et.splits_checked_extra(t, forest.mtry) for t in forest.trees)
    manual_would = sum(et.splits_checked_exhaustive(t, forest.mtry) for t in forest.trees)
    assert manual_checked == a.checked == et.forest_splits(forest, "extra")
    assert manual_would == a.would_check == et.forest_splits(forest, "exhaustive")
    assert a.checked < a.would_check / 20                                                            # mindestens 20-fach weniger auf dem Standarddatensatz


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------------

def test_effort_curve_numbers_class():
    er = ev.effort_rows("class", None, 1, 3, C.DEFAULT_N, C.DEFAULT_NOISE)
    assert er["rf_error"] == pytest.approx(0.1361, abs=0.001)
    tests = [r["test"] for r in er["rows"]]
    assert tests == pytest.approx([0.1796, 0.1583, 0.1574, 0.1528, 0.1509], abs=0.002)
    assert er["rows"][-1]["test"] > er["rf_error"]                                                     # auch bei 150 Bäumen erreicht Extra Trees Random Forest nicht
    assert er["rows"][0]["ratio"] < er["rows"][-1]["ratio"] < 1.0                                      # Aufwand wächst mit der Baumzahl, bleibt aber unter Random Forests Kosten


def test_effort_curve_numbers_reg():
    er = ev.effort_rows("reg", None, 1, 3, C.DEFAULT_N, C.DEFAULT_NOISE)
    assert er["rf_error"] == pytest.approx(11.26, abs=0.05)
    tests = [r["test"] for r in er["rows"]]
    assert tests == pytest.approx([14.87, 13.25, 12.75, 12.62, 12.45], abs=0.05)
    assert er["rows"][-1]["test"] > er["rf_error"]


def test_bias_variance_numbers():
    bv = ev.bias_variance_rows("class", None, 1, C.DEFAULT_N_TREES, 3, C.DEFAULT_N, C.DEFAULT_NOISE)
    assert bv["rf"]["bias2"] == pytest.approx(0.1058, abs=0.0005) and bv["rf"]["variance"] == pytest.approx(0.0112, abs=0.0005)
    assert bv["extra"]["bias2"] == pytest.approx(0.1263, abs=0.0005) and bv["extra"]["variance"] == pytest.approx(0.0069, abs=0.0005)
    var_cut = 1 - bv["extra"]["variance"] / bv["rf"]["variance"]
    bias_up = bv["extra"]["bias2"] / bv["rf"]["bias2"] - 1
    assert var_cut == pytest.approx(0.382, abs=0.003) and bias_up == pytest.approx(0.194, abs=0.003)

    bvr = ev.bias_variance_rows("reg", None, 1, C.DEFAULT_N_TREES, 3, C.DEFAULT_N, C.DEFAULT_NOISE)
    assert bvr["rf"]["bias2"] == pytest.approx(113.72, abs=0.1) and bvr["rf"]["variance"] == pytest.approx(16.09, abs=0.1)
    assert bvr["extra"]["bias2"] == pytest.approx(162.77, abs=0.1) and bvr["extra"]["variance"] == pytest.approx(13.07, abs=0.1)
    var_cut_r = 1 - bvr["extra"]["variance"] / bvr["rf"]["variance"]
    bias_up_r = bvr["extra"]["bias2"] / bvr["rf"]["bias2"] - 1
    assert var_cut_r == pytest.approx(0.188, abs=0.003) and bias_up_r == pytest.approx(0.431, abs=0.003)


# --- Erzeuger (geteilt mit cart-demo/bagging-demo/random-forest-demo) ---------------------------------------------------------------------------------

def test_generator_matches_cart_demo_conventions():
    ds = S.generate_dataset(500, 3, 0, 7)
    assert ds.X.shape == (500, 11) and ds.names[:2] == ("Distanz", "Ladegewicht")
    Xtr, ytr, Xte, yte = S.split(ds, "class")
    assert len(Xtr) == 350 and len(Xte) == 150


def test_default_mtry_heuristic():
    assert ev.default_mtry("class", 11) == 3 and ev.default_mtry("reg", 11) == 3
