"""Messungen an Extra Trees: Testfehler gegen die Zahl der Bäume verglichen mit Random Forest bei gleicher Merkmalsteilmenge, der Aufwand (geprüfte Paare) und die Bias-Varianz-Zerlegung."""

import numpy as np
from dataclasses import dataclass

import et_algorithm as et
import et_constants as C
import et_scenario as S
import et_tree as T

SIX = (C.DEFAULT_SEED,) + C.SWEEP_SEEDS
THREE = SIX[:3]                          # für den Aufwands-Sweep: voll gewachsene Regressionsbäume sind teuer (siehe cart-demo) - drei Datensätze genügen für den Trend


def error_of(forest, X, y, upto=None):
    return et.scores_from_values(forest.task, y, et.predict_value(forest, X, upto))["error"]


def baseline_error(ds, task):
    _, ytr, _, yte = S.split(ds, task)
    if task == "class":
        return float(np.mean(yte != int(ytr.mean() > 0.5)))
    return float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))


def default_mtry(task, d):
    return max(1, int(round(np.sqrt(d)))) if task == "class" else max(1, d // 3)


@dataclass
class Analysis:
    ds: object
    task: str
    criterion: str
    leaf: int
    n_trees: int
    forest: object                 # Extra-Trees-Wald
    exhaustive: object              # derselbe Aufbau (Bootstrap-Flag, mtry, Bäume), aber mit erschöpfender Suche (= Random Forest)
    single: object
    train: dict
    test: dict
    single_test: dict
    exhaustive_test: dict
    oob: dict
    baseline: float
    verdict: str
    imp: np.ndarray
    root_shares: list
    correlation: float
    exhaustive_correlation: float
    checked: int
    would_check: int


def analyse(task, criterion, leaf, n_trees, mtry, bootstrap, n, n_noise, label_noise, seed):
    ds = S.generate_dataset(n, n_noise, label_noise if task == "class" else 0, seed)
    criterion = criterion if criterion in C.CRITERIA[task] else C.DEFAULT_CRITERION[task]
    Xtr, ytr, Xte, yte = S.split(ds, task)
    d = Xtr.shape[1]
    mtry = min(int(mtry), d)
    forest = et.fit(Xtr, ytr, task, criterion, leaf, n_trees, mtry, extra=True, bootstrap=bootstrap, seed=0)
    exhaustive = et.fit(Xtr, ytr, task, criterion, leaf, n_trees, mtry, extra=False, bootstrap=bootstrap, seed=0)          # derselbe Bootstrap/mtry, nur die Schwellensuche unterscheidet sich
    single = forest.trees[0]
    train = et.scores_from_values(task, ytr, et.predict_value(forest, Xtr))
    test = et.scores_from_values(task, yte, et.predict_value(forest, Xte))
    single_test = et.scores_from_values(task, yte, T.predict_value(single, Xte))
    exhaustive_test = et.scores_from_values(task, yte, et.predict_value(exhaustive, Xte))
    oob = et.scores_from_values(task, ytr, et.oob_predict(forest, Xtr)) if bootstrap else {"error": float("nan"), "n": 0}
    baseline = baseline_error(ds, task)
    imp = et.importances_mean(forest)
    root_shares = et.root_shares(forest)
    correlation = et.tree_correlation(forest, Xte)
    exhaustive_correlation = et.tree_correlation(exhaustive, Xte)
    checked = et.forest_splits(forest, "extra")
    would_check = et.forest_splits(forest, "exhaustive")
    a = Analysis(ds, task, criterion, leaf, n_trees, forest, exhaustive, single, train, test, single_test, exhaustive_test, oob, baseline, "", imp, root_shares, correlation, exhaustive_correlation, checked, would_check)
    a.verdict = verdict(a)
    return a


def verdict(a):
    """'stump' (ein Baum), 'worse' (schlechter als ein Einzelbaum), 'better' (spürbar besser als die erschöpfende Suche bei gleichem mtry - selten, siehe Messwerte), 'cheaper_similar'
    (kein besserer, aber kein spürbar schlechterer Testfehler - der Aufwand sinkt trotzdem stark), sonst 'worse_but_cheaper' (schlechter, aber ein Bruchteil des Aufwands)."""
    if a.n_trees <= 1:
        return "stump"
    if a.test["error"] > a.single_test["error"] * 1.01:
        return "worse"
    if a.test["error"] < a.exhaustive_test["error"] * 0.99:
        return "better"
    if a.test["error"] < a.exhaustive_test["error"] * 1.03:
        return "cheaper_similar"
    return "worse_but_cheaper"


# --- Testfehler gegen die Zahl der Bäume: Extra Trees gegen Random Forest (fester Aufwand) --------------------------------------------------------------

EFFORT_TREE_GRID = (10, 30, 60, 100, 150)


def effort_rows(task, criterion, leaf, mtry, n, n_noise, seeds=THREE, tree_grid=EFFORT_TREE_GRID, rf_trees=30):
    """Mittel über sechs Datensätze: Random-Forest-Testfehler und -Aufwand bei `rf_trees` Bäumen (erschöpfende Suche) als fester Vergleichswert, dagegen Extra-Trees-Testfehler und -Aufwand über eine wachsende Baumzahl."""
    rf_errs, rf_checked = [], []
    et_errs = {k: [] for k in tree_grid}
    checked = {k: [] for k in tree_grid}
    would = {k: [] for k in tree_grid}
    for sd in seeds:
        ds = S.generate_dataset(n, n_noise, 0, sd)
        Xtr, ytr, Xte, yte = S.split(ds, task)
        rf = et.fit(Xtr, ytr, task, criterion, leaf, rf_trees, mtry, extra=False, bootstrap=False, seed=0)
        rf_errs.append(error_of(rf, Xte, yte))
        rf_checked.append(et.forest_splits(rf, "exhaustive"))
        for k in tree_grid:
            f = et.fit(Xtr, ytr, task, criterion, leaf, k, mtry, extra=True, bootstrap=False, seed=0)
            et_errs[k].append(error_of(f, Xte, yte))
            checked[k].append(et.forest_splits(f, "extra"))
            would[k].append(et.forest_splits(f, "exhaustive"))
    rf_checked_mean = float(np.mean(rf_checked))
    rows = [{"n_trees": k, "test": float(np.mean(et_errs[k])), "checked": float(np.mean(checked[k])), "ratio": float(np.mean(checked[k])) / rf_checked_mean} for k in tree_grid]
    return {"rf_error": float(np.mean(rf_errs)), "rf_checked": rf_checked_mean, "rows": rows}


# --- Bias-Varianz: Extra Trees gegen Random Forest --------------------------------------------------------------------------------------------------------

def bias_variance_rows(task, criterion, leaf, n_trees, mtry, n, n_noise, seeds=THREE):
    """Fixe Testpunkte (Seed 7); je Trainingsseed ein Random-Forest- und ein Extra-Trees-Wald (gleiches mtry, kein Bootstrap), beide auf denselben Testpunkten ausgewertet."""
    ds0 = S.generate_dataset(n, n_noise, 0, C.DEFAULT_SEED)
    Xeval = ds0.X[ds0.test]
    yeval = (ds0.y_true if task == "class" else ds0.y_reg)[ds0.test]
    rf_preds, et_preds = [], []
    for sd in seeds:
        ds = S.generate_dataset(n, n_noise, 0, sd)
        Xtr, ytr, _, _ = S.split(ds, task)
        rf_preds.append(et.predict_value(et.fit(Xtr, ytr, task, criterion, leaf, n_trees, mtry, extra=False, bootstrap=False, seed=0), Xeval))
        et_preds.append(et.predict_value(et.fit(Xtr, ytr, task, criterion, leaf, n_trees, mtry, extra=True, bootstrap=False, seed=0), Xeval))

    def decompose(preds):
        preds = np.array(preds)
        mean = preds.mean(axis=0)
        bias2 = float(np.mean((mean - yeval) ** 2))
        var = float(np.mean(preds.var(axis=0)))
        return {"bias2": bias2, "variance": var, "total": bias2 + var}

    return {"rf": decompose(rf_preds), "extra": decompose(et_preds)}
