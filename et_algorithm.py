"""Extra Trees: wie Random Forest (random-forest-demo) B Bäume mit `mtry` zufälligen Merkmalen je Split, aber jeder Split zieht pro Kandidatenmerkmal nur EINE zufällige Schwelle statt sie zu suchen
(Geurts, Ernst, Wehenkel 2006). Standardmäßig zusätzlich OHNE Bootstrap (jeder Baum sieht alle Trainingszeilen; die Zufallsschwellen allein sorgen für unterschiedliche Bäume) - Bootstrap ist ein Regler.
Der Baumkern (mit `mtry` UND `extra`) steht in `et_tree.py`."""

from dataclasses import dataclass

import numpy as np

import et_tree as T


@dataclass(frozen=True)
class Forest:
    trees: tuple                 # Tupel von T.Tree
    in_bag: np.ndarray            # (B, n) bool - ohne Bootstrap: überall True (jeder Baum sieht jede Zeile)
    task: str
    n_train: int
    mtry: int
    seed: int
    n_features: int
    bootstrap: bool
    extra: bool


def bootstrap_indices(n, seed):
    """n Zeilennummern mit Zurücklegen (eine Bootstrap-Stichprobe)."""
    return np.random.default_rng(seed).integers(0, n, n)


def _bootstrap_seed(seed, b):
    return seed * 1_000_003 + b


def _feature_seed(seed, b):
    """Eigener Zufalls-Strang für Merkmalswahl UND Zufallsschwellen je Baum, unabhängig vom Bootstrap-Seed desselben Baums."""
    return seed * 1_000_003 + b + 500_000_000


def fit(X, y, task, criterion=None, min_leaf=1, n_trees=30, mtry=None, extra=True, bootstrap=False, seed=0):
    """Wächst n_trees Bäume. `bootstrap=True` wie in bagging-demo/random-forest-demo (Bootstrap-Stichprobe je Baum, Out-of-Bag verfügbar); `bootstrap=False` (Geurts' Standard): jeder Baum sieht alle
    n Trainingszeilen - die Vielfalt kommt dann allein aus `mtry` und den Zufallsschwellen (`extra=True`)."""
    n, d = len(y), X.shape[1]
    mtry_eff = d if mtry is None else min(int(mtry), d)
    trees, in_bag = [], np.zeros((n_trees, n), dtype=bool)
    for b in range(n_trees):
        if bootstrap:
            idx = bootstrap_indices(n, _bootstrap_seed(seed, b))
        else:
            idx = np.arange(n)
        trees.append(T.grow(X[idx], y[idx], task, criterion, None, min_leaf, mtry=mtry_eff, seed=_feature_seed(seed, b), extra=extra))
        in_bag[b, idx] = True
    return Forest(tuple(trees), in_bag, task, n, mtry_eff, seed, d, bootstrap, extra)


def root_candidates(forest, b):
    """Welche `mtry` Merkmale der Baum b an der Wurzel zur Wahl hatte (der erste Zufallszug seines Merkmals-Strangs - die Wurzel ist immer der erste Knoten, den `grow` bearbeitet)."""
    if forest.mtry >= forest.n_features:
        return np.arange(forest.n_features)
    return np.sort(np.random.default_rng(_feature_seed(forest.seed, b)).choice(forest.n_features, forest.mtry, replace=False))


def _tree_values(forest, X, upto=None):
    trees = forest.trees[:upto] if upto else forest.trees
    return np.array([T.predict_value(t, X) for t in trees])


def predict_value(forest, X, upto=None):
    return _tree_values(forest, X, upto).mean(axis=0)


def predict(forest, X, upto=None):
    v = predict_value(forest, X, upto)
    return (v > 0.5).astype(int) if forest.task == "class" else v


def oob_predict(forest, X):
    """Für jede Trainingszeile der Mittelwert nur der Bäume, die sie nicht gesehen haben. Ohne Bootstrap ist das für jede Zeile leer (NaN überall - jeder Baum sah jede Zeile)."""
    vals = _tree_values(forest, X)
    oob = ~forest.in_bag
    count = oob.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, (vals * oob).sum(axis=0) / np.maximum(count, 1), np.nan)
    return np.where(count > 0, mean, np.nan)


def scores_from_values(task, y, v):
    """Gütemaße aus Blattwerten `v` gegen `y`, wie in cart-demo/bagging-demo/random-forest-demo. NaN-Zeilen werden ausgeschlossen."""
    ok = ~np.isnan(v)
    y, v = np.asarray(y)[ok], v[ok]
    if task == "class":
        pred = (v > 0.5).astype(int)
        return {"error": 1.0 - T.accuracy(y, pred), "accuracy": T.accuracy(y, pred), "auc": T.auc(y, v), "logloss": T.log_loss(y, v), "n": int(ok.sum())}
    return {"error": T.rmse(y, v), "rmse": T.rmse(y, v), "mae": T.mae(y, v), "r2": T.r2(y, v), "n": int(ok.sum())}


# --- Wald-Kennzahlen (Korrelation, Wurzeln) -------------------------------------------------------------------------------------------------------

def root_shares(forest):
    feats = [int(t.feature[0]) for t in forest.trees]
    vals, counts = np.unique(feats, return_counts=True)
    order = np.argsort(-counts)
    return [(int(vals[i]), int(counts[i])) for i in order]


def tree_correlation(forest, X):
    """Mittlere paarweise Ähnlichkeit der Einzelbaum-Vorhersagen auf X. None bei nur einem Baum."""
    vals = _tree_values(forest, X)
    B = len(vals)
    if B < 2:
        return None
    if forest.task == "reg":
        c = np.corrcoef(vals)
        iu = np.triu_indices(B, k=1)
        return float(np.nanmean(c[iu]))
    preds = (vals > 0.5).astype(int)
    iu = np.triu_indices(B, k=1)
    agree = np.array([np.mean(preds[i] == preds[j]) for i, j in zip(*iu)])
    return float(agree.mean())


def importances_mean(forest):
    imps = np.array([T.importances(t) for t in forest.trees])
    return imps.mean(axis=0)


# --- Permutationswichtigkeit auf Out-of-Bag-Zeilen (braucht Bootstrap) ---------------------------------------------------------------------------------

def _point_error(task, y, v):
    if task == "class":
        return 1.0 - T.accuracy(y, (v > 0.5).astype(int))
    return T.rmse(y, v)


def permutation_importance(forest, X, y, seed=0, min_oob=10):
    """Wie in random-forest-demo: je Baum der Fehler auf seinen Out-of-Bag-Zeilen, dann je Merkmal gemischt und der Anstieg gemessen. Braucht `forest.bootstrap = True` - ohne Bootstrap gibt es keine
    Out-of-Bag-Zeilen und die Funktion liefert lauter 0 (kein Baum liefert einen Beitrag)."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    task = forest.task
    per_tree = []
    for b, tree in enumerate(forest.trees):
        oob_idx = np.nonzero(~forest.in_bag[b])[0]
        if len(oob_idx) < min_oob:
            continue
        Xo, yo = X[oob_idx], y[oob_idx]
        base_err = _point_error(task, yo, T.predict_value(tree, Xo))
        deltas = np.empty(d)
        for j in range(d):
            Xp = Xo.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            deltas[j] = _point_error(task, yo, T.predict_value(tree, Xp)) - base_err
        per_tree.append(deltas)
    return np.mean(per_tree, axis=0) if per_tree else np.zeros(d)


def normalize_importance(imp):
    clipped = np.clip(imp, 0.0, None)
    s = clipped.sum()
    return clipped / s if s > 0 else clipped


# --- Aufwand: geprüfte (Merkmal, Schwelle)-Paare --------------------------------------------------------------------------------------------------------

def splits_checked_extra(tree, mtry):
    """Extra Trees: pro innerem Knoten genau `mtry` geprüfte Paare - ein Zufallsschnitt je Kandidatenmerkmal, unabhängig von der Knotengröße."""
    return int(mtry) * T.n_splits(tree)


def splits_checked_exhaustive(tree, mtry):
    """CART/Bagging/Random Forest: pro innerem Knoten mit m Zeilen werden für jedes der `mtry` Kandidatenmerkmale alle m - 1 Schwellen geprüft (keine Gleichstände bei stetigen Merkmalen)."""
    inner = tree.internal_nodes()
    return int(mtry) * int((tree.n[inner] - 1).sum())


def forest_splits(forest, kind):
    """Summe der geprüften Paare über alle Bäume des Walds; `kind` = 'extra' oder 'exhaustive' (Letzteres: wie viele Prüfungen derselbe Baumaufbau mit erschöpfender Suche gebraucht hätte)."""
    fn = splits_checked_extra if kind == "extra" else splits_checked_exhaustive
    return int(sum(fn(t, forest.mtry) for t in forest.trees))
