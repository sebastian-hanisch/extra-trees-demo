"""Extra Trees - zufällige Schwellen statt Split-Suche - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Extra Trees - und lässt stattdessen das Beispiel wachsen.
Viertes und letztes Stück des Bagging-Asts der Baumbasierten Linie der "Konzepte"-Reihe: der Nachfolger von Random Forest (random-forest-demo). Random Forest sucht an jedem Split noch die
BESTE Schwelle unter den mtry Kandidatenmerkmalen; Extra Trees zieht pro Kandidat nur EINE zufällige Schwelle - mehr Bias, weniger Varianz, ein Bruchteil des Rechenaufwands.
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import et_algorithm as et
import et_constants as C
import et_evaluation as ev
from et_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from et_visualization import (
    build_bias_variance,
    build_effort_curve,
    build_effort_ratio,
    build_importance,
    build_map,
    build_tree,
    feature_label,
    value_text,
)

st.set_page_config(page_title="Extra Trees – Sebastian Hanisch", layout="wide")

VERDICT_TEXT = {
    "stump": "ℹ️ Nur ein Baum: kein Wald, kein Vergleich möglich.",
    "worse": "⚠️ **Schlechter als ein Einzelbaum** - mtry, Bäume oder Blattgröße sind hier zu klein.",
    "better": "✅ **Besser als die erschöpfende Suche bei gleichem mtry** - selten, aber möglich (siehe README).",
    "cheaper_similar": "➖ **Kaum schlechter als die erschöpfende Suche** - bei einem Bruchteil des Aufwands.",
    "worse_but_cheaper": "ℹ️ **Schlechter als die erschöpfende Suche, aber ein Bruchteil des Aufwands** - mehr Bias, weniger Varianz, hier gewinnt der Bias.",
}


def _err(task, x):
    return f"{x:.1%}" if task == "class" else f"{x:.1f} min"


@st.cache_resource(show_spinner=False, max_entries=16)
def _analysis(*params):
    return ev.analyse(*params)


@st.cache_data(show_spinner=False, max_entries=6)
def _effort_rows(task, criterion, leaf, mtry, n, n_noise):
    return ev.effort_rows(task, criterion, leaf, mtry, n, n_noise)


@st.cache_data(show_spinner=False, max_entries=6)
def _bias_variance(task, criterion, leaf, n_trees, mtry, n, n_noise):
    return ev.bias_variance_rows(task, criterion, leaf, n_trees, mtry, n, n_noise)


st.title("🌳🎰 Extra Trees – zufällige Schwellen statt Split-Suche")
st.markdown(
    """
Random Forest (random-forest-demo) entkoppelt Bäume, indem jeder Split nur eine zufällige Teilmenge von **mtry** Merkmalen zur Wahl hat - **unter diesen** sucht er aber weiterhin die **beste** Schwelle,
über alle Werte im Knoten. **Extra Trees** (*Extremely Randomized Trees*, Geurts, Ernst, Wehenkel 2006) geht einen Schritt weiter: für jedes Kandidatenmerkmal wird **nur eine** Schwelle zufällig gezogen
(gleichverteilt zwischen dem kleinsten und größten Wert im Knoten) - keine Suche, nur ein Wurf. Das macht jeden Split schlechter (**mehr Bias**), aber die Bäume unterscheiden sich noch stärker
(**weniger Varianz**) - und vor allem: **massiv weniger Rechenaufwand** (ein geprüftes Paar statt bis zu tausenden Schwellen je Merkmal). Standardmäßig verzichtet Extra Trees zusätzlich auf Bootstrap-Stichproben -
jeder Baum sieht alle Trainingszeilen, die Zufallsschwellen allein sorgen für Vielfalt.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - viertes und letztes Stück des Bagging-Asts der Baumbasierten Linie der \"Konzepte\"-Reihe, Nachfolger von Random Forest - **ein** Verfahren an einem wachsenden Beispiel. "
    "Das Verfahren geht auf Geurts, Ernst und Wehenkel (2006) zurück; alle Lieferungen, Merkmale und Zahlen dieser Demo sind erzeugt und gemessen - keine echten Daten. Der Baumkern (`et_tree.py`) ist aus cart-demo/random-forest-demo übernommen und um die Zufallsschwelle erweitert; scikit-learn kommt nur in den Tests als Gegenprobe vor."
)
st.caption(
    "**Bezug zu OR:** viel weniger Rechenaufwand je Baum heißt, dass sich bei knapper Rechenzeit (etwa in einer Schleife der Tourenplanung, die viele Vorhersagen pro Sekunde braucht) ein größerer Wald oder häufigeres Neu-Trainieren ausgeht - eine Abwägung zwischen Modellgüte und Rechenbudget, nicht nur zwischen Modellen."
)

with st.expander("So funktioniert Extra Trees", expanded=True):
    st.markdown(
        """
1. **Wie Random Forest:** an jedem Knoten wird zuerst eine zufällige Teilmenge von **mtry** der d Merkmale gezogen.
2. **Der Unterschied:** für jedes dieser mtry Merkmale wird **eine einzige** Schwelle zufällig gezogen (gleichverteilt zwischen Minimum und Maximum dieses Merkmals im Knoten) - keine Suche über alle Werte. Der beste der mtry Zufalls-Splits gewinnt.
3. **Aufwand:** ein Knoten mit m Zeilen braucht bei erschöpfender Suche bis zu mtry × (m − 1) geprüfte Schwellen; Extra Trees braucht immer nur **mtry** - unabhängig von m.
4. **Kein Bootstrap (Standard):** jeder Baum sieht alle n Trainingszeilen; die Zufallsschwellen (und bei mtry < d die Merkmalsauswahl) reichen für unterschiedliche Bäume. Bootstrap ist ein Regler - zusammen ergibt das eine zweite, unabhängige Quelle der Vielfalt (und Out-of-Bag).
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()
if st.session_state.get(KEPT["criterion_select"], "gini") not in ("gini", "entropy"):
    st.session_state[KEPT["criterion_select"]] = "gini"

with st.sidebar:
    st.header("⚙️ Einstellungen")
    task = st.selectbox("Aufgabe", C.TASKS, key="task_select", format_func=lambda k: C.TASK_LABELS[k])
    if task == "class":
        seed_widget("criterion_select")
        crit = st.selectbox("Split-Kriterium", C.CRITERIA["class"], key="criterion_select", format_func=lambda k: C.CRITERION_LABELS[k])
        st.session_state[KEPT["criterion_select"]] = crit
    else:
        crit = "variance"
        st.caption("Split-Kriterium: Varianz - bei einem Zahlenziel gibt es keine Wahl.")
    leaf = st.slider("Mindestgröße eines Blatts", *bounds("leaf_slider"), key="leaf_slider")
    n_trees = st.slider("Zahl der Bäume", *bounds("n_trees_slider"), key="n_trees_slider")
    n_noise = st.slider("Rauschmerkmale", *bounds("n_noise_slider"), key="n_noise_slider")
    d_now = C.N_BASE + int(n_noise)
    mtry_hi = min(bounds("mtry_slider")[1], d_now)
    if st.session_state["mtry_slider"] > mtry_hi:
        st.session_state["mtry_slider"] = mtry_hi
    mtry = st.slider(f"mtry (Merkmale je Split, von {d_now})", 1, mtry_hi, key="mtry_slider",
                     help=f"Wie viele der {d_now} Merkmale an jedem Split zur Wahl stehen - jedes bekommt genau eine zufällige Schwelle. {d_now} = alle Merkmale, aber immer noch mit Zufallsschwelle statt Suche.")
    bootstrap = st.checkbox("Bootstrap-Stichproben", key="bootstrap_check", help="Aus: jeder Baum sieht alle Trainingszeilen (Geurts' Standard, auch scikit-learns Voreinstellung). An: wie in bagging-demo/random-forest-demo - macht Out-of-Bag verfügbar.")
    if task == "class":
        seed_widget("label_noise_slider")
        label_noise = st.slider("Falsche Etiketten im Training [%]", *bounds("label_noise_slider"), key="label_noise_slider")
        st.session_state[KEPT["label_noise_slider"]] = label_noise
    else:
        label_noise = int(st.session_state.get(KEPT["label_noise_slider"], C.DEFAULT_LABEL_NOISE))
        st.caption("Falsche Etiketten gibt es nur bei der Klassifikation.")
    st.markdown("**Daten**")
    n = st.slider("Lieferungen", *bounds("n_slider"), key="n_slider", step=100)
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Daten generieren", width="stretch", on_click=randomize_seed)

base_params = (task, crit, int(leaf), int(n_trees), int(mtry), bool(bootstrap))
data_params = (int(n), int(n_noise), int(label_noise), int(seed))
with st.spinner("Rechne ..."):
    a = _analysis(*base_params, *data_params)
ds = a.ds
Xtr, ytr, Xte, yte = ds.X[ds.train], ds.y(task)[ds.train], ds.X[ds.test], (ds.y_true if task == "class" else ds.y_reg)[ds.test]
names = ds.names
n_feat = len(names)
n_test = len(ds.test)

with st.sidebar:
    st.markdown("**Ansicht**")
    for key, default in (("map_x_select", C.DEFAULT_MAP[0]), ("map_y_select", C.DEFAULT_MAP[1])):
        if st.session_state[key] >= n_feat:
            st.session_state[key] = default
    fx = st.selectbox("Karte: waagerecht", range(n_feat), key="map_x_select", format_func=lambda f: feature_label(names, f))
    fy = st.selectbox("Karte: senkrecht", range(n_feat), key="map_y_select", format_func=lambda f: feature_label(names, f))
    if st.session_state.get("sample_slider", 0) > n_test - 1:
        st.session_state["sample_slider"] = 0
    sample_idx = st.slider("Testlieferung", 0, n_test - 1, 0, key="sample_slider")
sync_query_params({"task_select": task, "criterion_select": crit if task == "class" else st.session_state.get(KEPT["criterion_select"], "gini"), "leaf_slider": int(leaf), "n_trees_slider": int(n_trees),
                   "mtry_slider": int(mtry), "bootstrap_check": bool(bootstrap), "n_slider": int(n), "n_noise_slider": int(n_noise), "label_noise_slider": int(label_noise), "seed_input": int(seed),
                   "map_x_select": int(fx), "map_y_select": int(fy)})

view_key = (base_params, data_params)
if st.session_state.get("et_owner") != view_key:
    st.session_state["et_owner"] = view_key
    st.session_state["et_step"] = int(n_trees)

# --- Extra Trees in Aktion ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Extra Trees in Aktion")
st.caption("Der Wald wächst Baum für Baum; links der zuletzt hinzugekommene Einzelbaum, rechts das Wald-Mittel über zwei Merkmale.")
if n_trees > 1:
    step_col, play_col = st.columns([5, 2])
    with step_col:
        step = st.slider("Bäume im Mittel", 1, int(n_trees), key="et_step")
    with play_col:
        auto_play = st.button("▶️ Abspielen", width="stretch")
else:
    step, auto_play = 1, False
    st.info("ℹ️ Nur ein Baum eingestellt - kein Mittel zu bilden. Mehr Bäume in der Seitenleiste zeigen den Effekt.")
view_slot = st.empty()
sample_x = Xte[sample_idx]
sample_y = yte[sample_idx]


def _render(current):
    sub = et.Forest(a.forest.trees[:current], a.forest.in_bag[:current], task, a.forest.n_train, a.forest.mtry, a.forest.seed, a.forest.n_features, a.forest.bootstrap, a.forest.extra)
    latest = a.forest.trees[current - 1]
    with view_slot.container():
        c1, c2 = st.columns([2, 3])
        with c1:
            st.plotly_chart(build_tree(latest, task, (float(ytr.min()), float(ytr.max()))), width="stretch", key=f"tree_chart_{current}")
            st.caption(f"Baum {current}: Wurzel = **{names[latest.feature[0]]}** (zufällige Schwelle {latest.threshold[0]:.3g}), {latest.n_leaves} Blätter.")
        with c2:
            st.plotly_chart(build_map(sub, ds, fx, fy, upto=None, sample=sample_x), width="stretch", key=f"map_chart_{current}")
        pred_here = et.predict_value(sub, sample_x.reshape(1, -1))[0]
        truth = ("zu spät" if sample_y == 1 else "pünktlich") if task == "class" else f"{sample_y:.0f} min"
        st.markdown(f"**Testlieferung {sample_idx}:** Mittel der ersten {current} Bäume = **{value_text(task, pred_here)}**; tatsächlich: **{truth}**.")


if auto_play:
    frames = sorted(set(np.unique(np.round(np.linspace(1, int(n_trees), min(12, int(n_trees)))).astype(int))))
    for kk in frames:
        _render(kk)
        time.sleep(min(0.9, 6.0 / len(frames)))
    step = int(n_trees)
else:
    _render(step)

st.markdown("---")

# --- Was der Wald gelernt hat -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 📐 Was der Wald gelernt hat – und wie gut er auf neuen Lieferungen ist")
st.caption("Vergleichsmaßstäbe: **ein Einzelbaum** und **dieselbe Konfiguration mit erschöpfender Schwellensuche** (= Random Forest bzw. cart-demo, je nach mtry) statt Zufallsschwellen.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("mtry / Bäume", f"{int(mtry)} / {int(n_trees)}", help=f"Merkmale je Split von {n_feat} insgesamt, und Zahl der Bäume.")
m2.metric("Testfehler", _err(task, a.test["error"]), delta=f"1 Baum: {_err(task, a.single_test['error'])}", delta_color="off")
if bootstrap:
    m3.metric("Out-of-Bag-Fehler", _err(task, a.oob["error"]), delta=f"Raten: {_err(task, a.baseline)}", delta_color="off")
else:
    m3.metric("Raten (Fehler ohne Wald)", _err(task, a.baseline), help="Ohne Bootstrap gibt es keine Out-of-Bag-Schätzung - jeder Baum sieht jede Zeile.")
gap = a.test["error"] - a.exhaustive_test["error"]
gap_text = f"{gap:+.1%} ggü. hier" if task == "class" else f"{gap:+.1f} min ggü. hier"
m4.metric("Erschöpfende Suche", _err(task, a.exhaustive_test["error"]), delta=gap_text, delta_color="inverse", help="Testfehler mit derselben Konfiguration (Bootstrap, mtry, Bäume), aber bester statt zufälliger Schwelle.")
st.markdown(VERDICT_TEXT[a.verdict])
ratio = a.checked / a.would_check if a.would_check else 0.0
st.caption(f"Geprüfte (Merkmal, Schwelle)-Paare: **{a.checked:,}** statt **{a.would_check:,}** bei erschöpfender Suche (**{ratio:.1%}** des Aufwands).".replace(",", "."))

st.markdown("**Wichtigkeit der Merkmale (Mittel über die Bäume)**")
st.plotly_chart(build_importance(names, a.imp), width="stretch", key="importance_chart")
st.caption("Gini-Wichtigkeit wie in cart-demo/random-forest-demo, hier über die Extra-Trees-Bäume gemittelt. Orange = Rauschmerkmale.")

st.markdown("---")

# --- Experimente -----------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Testfehler gegen die Zahl der Bäume: Extra Trees gegen Random Forest")
if st.button("Testfehler und Aufwand über mehrere Baumzahlen messen (dauert einen Moment)", key="effort_start"):
    st.session_state["effort_on"] = True
if st.session_state.get("effort_on"):
    with st.spinner("Wachse Random-Forest- und Extra-Trees-Wälder auf drei Datensätzen ..."):
        er = _effort_rows(task, crit, int(leaf), int(mtry), int(n), int(n_noise))
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(build_effort_curve(er["rows"], task, er["rf_error"], int(n_trees)), width="stretch", key="effort_chart")
        st.plotly_chart(build_effort_ratio(er["rows"]), width="stretch", key="effort_ratio_chart")
    c2.table({"Bäume": [r["n_trees"] for r in er["rows"]], "Testfehler": [_err(task, r["test"]) for r in er["rows"]], "Aufwand ggü. RF": [f"{r['ratio']:.1%}" for r in er["rows"]]})
    last = er["rows"][-1]
    st.caption(f"Mittel über drei Datensätze, mtry = {int(mtry)}. Random Forest (30 Bäume, erschöpfende Suche): Testfehler {_err(task, er['rf_error'])}. Extra Trees nähert sich mit mehr Bäumen an "
               f"({_err(task, er['rows'][0]['test'])} bei {er['rows'][0]['n_trees']} Bäumen, {_err(task, last['test'])} bei {last['n_trees']} Bäumen), erreicht Random Forest aber auch bei {last['n_trees']} Bäumen "
               f"(nur {last['ratio']:.0%} von dessen Aufwand) nicht ganz. Mehr billige Bäume schließen die Lücke - hier - nicht vollständig, aber jeder Punkt auf der Kurve kostet einen Bruchteil dessen, was Random Forest kostet.")

st.markdown("---")

st.subheader("🔬 Bias-Varianz-Zerlegung: Extra Trees gegen Random Forest")
if st.button("Bias und Varianz über drei Datensätze messen (dauert einen Moment)", key="bv_start"):
    st.session_state["bv_on"] = True
if st.session_state.get("bv_on"):
    with st.spinner("Wachse Random-Forest- und Extra-Trees-Wälder auf drei Datensätzen ..."):
        bv = _bias_variance(task, crit, int(leaf), int(n_trees), int(mtry), int(n), int(n_noise))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_bias_variance(bv), width="stretch", key="bv_chart")
    c2.table({"": ["Bias²", "Varianz", "Gesamt"], "Random Forest": [f"{bv['rf']['bias2']:.4g}", f"{bv['rf']['variance']:.4g}", f"{bv['rf']['total']:.4g}"],
              "Extra Trees": [f"{bv['extra']['bias2']:.4g}", f"{bv['extra']['variance']:.4g}", f"{bv['extra']['total']:.4g}"]})
    var_cut = 1 - bv["extra"]["variance"] / bv["rf"]["variance"]
    bias_up = bv["extra"]["bias2"] / bv["rf"]["bias2"] - 1
    st.caption(f"Drei Datensätze mit denselben Regeln (nur der Seed ändert sich), Random Forest und Extra Trees auf denselben festen Testpunkten (Seed {C.DEFAULT_SEED}). Extra Trees senkt die **Varianz** um **{var_cut:.0%}**, "
               f"der **Bias²** steigt um **{bias_up:.0%}** - genau der erwartete Tausch. Auf diesem Datensatz überwiegt der zusätzliche Bias: der Gesamtfehler ist am Ende höher als bei Random Forest.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Aufwand sinkt, die Genauigkeit auch** | Eine zufällige statt der besten Schwelle ist fast immer ein schlechterer Split (Messwert oben) - Extra Trees tauscht Genauigkeit gegen Geschwindigkeit, nicht umgekehrt. Auf diesem Datensatz gewinnt Random Forest bei der reinen Vorhersagegüte. | mehr Bäume (hilft nur teilweise, siehe Experiment), größeres mtry |
| **Mehr billige Bäume sind kein Ersatz für bessere Splits** | Selbst bei einem Bruchteil des Aufwands von Random Forest bleibt eine Lücke, die auch fünfmal so viele Bäume nicht schließen (gemessen oben) - die zusätzliche Varianzsenkung sättigt, der Bias bleibt. | Random Forest, wenn Genauigkeit vor Geschwindigkeit geht |
| **Ohne Bootstrap gibt es kein Out-of-Bag** | Der Standardfall (kein Bootstrap) braucht eigene Testdaten für eine ehrliche Fehlerschätzung - die eingebaute OOB-Schätzung aus bagging-demo/random-forest-demo entfällt, außer man schaltet Bootstrap zusätzlich ein. | Bootstrap-Regler |
| **Die Zufallsschwelle ignoriert die Datenverteilung** | Gleichverteilt zwischen Minimum und Maximum trifft bei schiefen oder mehrgipfligen Merkmalen selten eine wirklich informative Stelle - die besten Splits liegen oft nicht in der Mitte des Wertebereichs. | erschöpfende Suche (Random Forest) für Merkmale mit bekannter Struktur |
| **Randomisierung hat Grenzen** | Bei sehr wenigen informativen Merkmalen unter vielen Rauschmerkmalen trifft die zufällige Merkmalsteilmenge (mtry) oft gar kein echtes Merkmal - dasselbe Problem wie bei Random Forest, hier zusätzlich verschärft durch die zufällige Schwelle. | größeres mtry, mehr Bäume |
"""
)
st.caption("Damit endet der Bagging-Ast der Baumbasierten Linie (CART → Bagging → Random Forest → Extra Trees). Der Boosting-Ast (AdaBoost → Gradient Boosting → XGBoost/LightGBM/CatBoost) beginnt mit dem nächsten Stück.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Extra Trees.** Wie Random Forest: $B$ Bäume, an jedem Knoten $t$ eine zufällige Teilmenge $M_t\subset\{1,\dots,d\}$, $|M_t|=\texttt{mtry}$. Für jedes $j\in M_t$ wird **eine** Schwelle
$s_j \sim \mathrm{Uniform}(\min_{i\in t} x_{ij},\, \max_{i\in t} x_{ij})$ gezogen (statt aller Kandidaten zwischen den sortierten Werten); gewählt wird $j^* = \arg\max_{j\in M_t} \Delta(j, s_j)$ mit demselben
Gain $\Delta$ wie in cart-demo. Ohne Bootstrap sieht jeder Baum alle $n$ Trainingszeilen.

**Aufwand.** Erschöpfende Suche (cart-demo/Random Forest) prüft an einem Knoten mit $m$ Zeilen bis zu $\texttt{mtry}\times(m-1)$ Schwellen; Extra Trees prüft immer genau $\texttt{mtry}$ - unabhängig von $m$. Über einen
ganzen Baum mit $L-1$ inneren Knoten ($L$ Blätter): $\texttt{mtry}\times(L-1)$ statt $\texttt{mtry}\times\sum_t (m_t-1)$.

**Bias-Varianz.** Wie in bagging-demo/random-forest-demo, aber mit einer dritten Zufallsquelle (der Schwelle selbst): die erwartete Vorhersage $\mathbb E[\hat f]$ verschiebt sich stärker vom besten erreichbaren Split weg
(höherer Bias, weil $s_j$ selten die optimale Schwelle trifft), während die Streuung zwischen den Bäumen wächst - das senkt die Korrelation $\rho$ zwischen Baumpaaren und damit (siehe bagging-demo/random-forest-demo-Formel) die Varianz des Mittels zusätzlich.

Implementiert in `et_tree.py` (Baumkern mit `mtry` und `random_split`), `et_algorithm.py` (Bootstrap-Regler, Mitteln, Aufwands-Zähler), `et_evaluation.py` (Analyse, Aufwands-Sweep, Bias-Varianz).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
