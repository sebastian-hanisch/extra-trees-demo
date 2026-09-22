"""Extra Trees gegen unabhängige Referenzen: Invarianten der Zufallsschwelle (liegt im Wertebereich, Blattgröße), scikit-learns ExtraTreesClassifier/Regressor über ein Fehlerband (andere Zufallsquelle,
keine exakte Übereinstimmung erwartet), der Zähler geprüfter (Merkmal, Schwelle)-Paare gegen eine direkte Instrumentierung. Der Baumkern ohne `extra` ist in cart-demo/random-forest-demo geprüft."""

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

import et_algorithm as et
import et_scenario as S
import et_tree as T


def _continuous(n=400, d=6, seed=0, task="class", noise=1.6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    signal = X[:, 0] + 0.7 * np.sin(2 * X[:, 1]) + 0.5 * (X[:, 2] > 0.3) * X[:, 3]
    y = (signal + rng.normal(0, noise, n) > 0).astype(float) if task == "class" else signal * 3 + 10 + rng.normal(0, 1.0, n)
    return X, y


# --- Invarianten der Zufallsschwelle --------------------------------------------------------------------------------------------------------------------

def _walk_thresholds(tree, X):
    """Für jeden inneren Knoten: liegt die Schwelle im Wertebereich der Zeilen, die diesen Knoten tatsächlich erreichen?"""
    stack = [(0, np.arange(len(X)))]
    while stack:
        t, idx = stack.pop()
        f = tree.feature[t]
        if f < 0:
            continue
        v = X[idx, f]
        thr = tree.threshold[t]
        assert v.min() <= thr <= v.max(), (t, v.min(), thr, v.max())
        go_left = v <= thr
        stack.append((tree.left[t], idx[go_left]))
        stack.append((tree.right[t], idx[~go_left]))


@pytest.mark.parametrize("task", ["class", "reg"])
@pytest.mark.parametrize("mtry", [None, 3])
def test_random_thresholds_stay_within_the_data_range_at_every_node(task, mtry):
    X, y = _continuous(400, 6, 1, task)
    tree = T.grow(X, y, task, None, None, 3, mtry=mtry, seed=5, extra=True)
    _walk_thresholds(tree, X)
    assert tree.n_leaves > 1


def test_leaf_sizes_respect_min_leaf_everywhere():
    X, y = _continuous(500, 5, 2, "class")
    for leaf in (1, 5, 15):
        tree = T.grow(X, y, "class", None, None, leaf, mtry=3, seed=0, extra=True)
        assert (tree.n[tree.feature < 0] >= leaf).all()


def test_extra_trees_grows_a_different_tree_from_a_different_seed_on_the_same_data():
    X, y = _continuous(300, 5, 0, "class")
    t1 = T.grow(X, y, "class", None, None, 5, mtry=3, seed=1, extra=True)
    t2 = T.grow(X, y, "class", None, None, 5, mtry=3, seed=2, extra=True)
    assert not np.array_equal(t1.feature, t2.feature) or not np.allclose(t1.threshold, t2.threshold, equal_nan=True)


def test_random_split_never_beats_the_best_split_in_expected_gain_but_can_be_worse():
    """Kein Widerspruch, nur eine Erwartung: die beste unter mtry zufälligen Schwellen ist im Mittel nicht besser als die gesuchte beste Schwelle (best_split) auf denselben Kandidaten."""
    X, y = _continuous(400, 4, 0, "class")
    rng = np.random.default_rng(9)
    gains_random, gains_best = [], []
    for trial in range(30):
        r = T.random_split(X, y, "gini", 5, np.arange(4), np.random.default_rng(trial))
        b = T.best_split(X, y, "gini", 5, np.arange(4))
        if r is not None:
            gains_random.append(r[2])
        if b is not None:
            gains_best.append(b[2])
    assert np.mean(gains_random) <= np.mean(gains_best) + 1e-9


# --- Zähler geprüfter Paare: Formel gegen direkte Instrumentierung -----------------------------------------------------------------------------------

def test_evaluated_split_count_formula_matches_a_direct_instrumentation():
    """Zählt während des Wachsens tatsächlich, wie oft random_split für ein Kandidatenmerkmal einen Gain berechnet (auch wenn der Split am Ende ungültig ist) - muss exakt mtry * n_splits(tree) ergeben,
    solange kein Kandidat wegen eines im Knoten konstanten Merkmals übersprungen wird (bei stetigen Rauschdaten praktisch nie)."""
    X, y = _continuous(300, 5, 3, "class")
    mtry = 3
    calls = {"n": 0}
    orig = T.node_impurity

    # random_split nachgebaut, nur um jeden geprüften Kandidaten zu zählen (auch ungültige) - die Logik ist wortgleich zu T.random_split.
    def counting_random_split(Xn, yn, criterion, min_leaf, candidates, rng):
        if len(yn) < 2 * min_leaf or len(yn) < 2:
            return None
        parent = orig(yn, criterion)
        n = len(yn)
        best = None
        for f in candidates:
            calls["n"] += 1
            v = Xn[:, f]
            lo, hi = float(v.min()), float(v.max())
            if hi <= lo:
                continue
            thr = float(rng.uniform(lo, hi))
            go_left = v <= thr
            nl = int(go_left.sum())
            nr = n - nl
            if nl < min_leaf or nr < min_leaf:
                continue
            gain = parent - (nl * orig(yn[go_left], criterion) + nr * orig(yn[~go_left], criterion)) / n
            if best is None or gain > best[2] + 1e-12:
                best = (int(f), thr, gain)
        return best

    real_random_split = T.random_split
    T.random_split = counting_random_split
    try:
        tree = T.grow(X, y, "class", None, None, 1, mtry=mtry, seed=0, extra=True)
    finally:
        T.random_split = real_random_split
    assert calls["n"] == mtry * T.n_splits(tree)
    assert et.splits_checked_extra(tree, mtry) == calls["n"]


def test_extra_trees_checks_far_fewer_splits_than_exhaustive_search():
    ds = S.generate_dataset(600, 3, 0, 7)
    Xtr, ytr, _, _ = S.split(ds, "class")
    forest = et.fit(Xtr, ytr, "class", None, 1, 10, mtry=3, extra=True, bootstrap=False, seed=0)
    checked = et.forest_splits(forest, "extra")
    would_have = et.forest_splits(forest, "exhaustive")
    assert checked < would_have / 5                                                                   # deutlich weniger, nicht nur ein bisschen
    assert checked == 3 * sum(T.n_splits(t) for t in forest.trees)


# --- Kreuzprobe mit scikit-learn (Rang-/Fehlerbänder, andere Zufallsquelle) ----------------------------------------------------------------------------

@pytest.mark.parametrize("task,cls", [("class", ExtraTreesClassifier), ("reg", ExtraTreesRegressor)])
def test_forest_error_is_close_to_scikit_learns_extra_trees(task, cls):
    ds = S.generate_dataset(800, 3, 0, 7)
    Xtr, ytr, Xte, yte = S.split(ds, task)
    d = Xtr.shape[1]
    mtry = int(round(np.sqrt(d))) if task == "class" else max(1, d // 3)
    forest = et.fit(Xtr, ytr, task, None, 5, 80, mtry=mtry, extra=True, bootstrap=False, seed=0)
    ref = cls(n_estimators=80, max_features=mtry, min_samples_leaf=5, bootstrap=False, random_state=0).fit(Xtr, ytr)
    v = et.predict_value(forest, Xte)
    pred = (v > 0.5).astype(int) if task == "class" else v
    if task == "class":
        our_err = 1.0 - T.accuracy(yte, pred)
        ref_err = 1.0 - ref.score(Xte, yte)
    else:
        our_err = T.rmse(yte, pred)
        ref_err = float(np.sqrt(np.mean((ref.predict(Xte) - yte) ** 2)))
    assert our_err < 2.0 * ref_err + 0.05


# --- Bootstrap-Regler -------------------------------------------------------------------------------------------------------------------------------

def test_without_bootstrap_every_tree_sees_every_row_and_oob_is_empty():
    X, y = _continuous(150, 4, 0, "class")
    forest = et.fit(X, y, "class", None, 1, 8, mtry=3, extra=True, bootstrap=False, seed=0)
    assert forest.in_bag.all()
    assert np.all(np.isnan(et.oob_predict(forest, X)))


def test_with_bootstrap_oob_behaves_like_random_forest_demo():
    X, y = _continuous(400, 4, 0, "class")
    forest = et.fit(X, y, "class", None, 1, 30, mtry=3, extra=True, bootstrap=True, seed=4)
    share_in_bag = forest.in_bag.mean()
    assert share_in_bag == pytest.approx(1 - 1 / np.e, abs=0.02)
    assert np.mean(~np.isnan(et.oob_predict(forest, X))) > 0.9


def test_permutation_importance_is_zero_without_bootstrap_and_meaningful_with_it():
    X, y = _continuous(500, 4, 0, "class")
    no_boot = et.fit(X, y, "class", None, 3, 20, mtry=3, extra=True, bootstrap=False, seed=0)
    assert np.all(et.permutation_importance(no_boot, X, y, seed=1) == 0.0)
    with_boot = et.fit(X, y, "class", None, 3, 30, mtry=3, extra=True, bootstrap=True, seed=0)
    imp = et.permutation_importance(with_boot, X, y, seed=1)
    assert imp[0] > 0.02                                                                                # das stärkste echte Merkmal


# --- extra=False reproduziert random-forest-demo/cart-demo exakt (Rückwärtskompatibilität des Kerns) --------------------------------------------------

def test_extra_false_with_mtry_equals_the_random_forest_kernel():
    X, y = _continuous(300, 5, 1, "class")
    a = T.grow(X, y, "class", None, None, 5, mtry=3, seed=7, extra=False)
    b = T.grow(X, y, "class", None, None, 5, mtry=3, seed=7)                                            # extra=False ist der Default
    assert np.array_equal(a.feature, b.feature) and np.allclose(a.threshold, b.threshold, equal_nan=True)


def test_extra_false_without_mtry_equals_the_plain_cart_kernel():
    X, y = _continuous(300, 5, 2, "reg")
    full = T.grow(X, y, "reg", None, None, 1)
    same = T.grow(X, y, "reg", None, None, 1, mtry=None, seed=99, extra=False)
    assert np.array_equal(full.feature, same.feature) and np.allclose(full.threshold, same.threshold, equal_nan=True)


# --- Grenzfälle --------------------------------------------------------------------------------------------------------------------------------------

def test_a_constant_candidate_feature_is_skipped_without_crashing():
    X = np.column_stack([np.full(100, 3.0), np.random.default_rng(0).normal(size=100)])
    y = (X[:, 1] > 0).astype(float)
    tree = T.grow(X, y, "class", None, None, 1, mtry=2, seed=0, extra=True)
    assert tree.n_leaves > 1 and 0 not in set(tree.feature[tree.feature >= 0])


def test_a_single_tree_forest_equals_that_tree():
    X, y = _continuous(120, 4, 0, "reg")
    forest = et.fit(X, y, "reg", None, 1, 1, mtry=2, extra=True, bootstrap=False, seed=0)
    assert np.allclose(et.predict_value(forest, X), T.predict_value(forest.trees[0], X))
