"""Plotly-Darstellungen: einzelner Baum, Karte des Wald-Mittels, Testfehler-gegen-Bäume- und Bias-Varianz-Kurven (Extra Trees gegen Random Forest). Alle Achsen sind gesperrt (Touch-Scrollen)."""

import numpy as np
import plotly.graph_objects as go

import et_algorithm as et
import et_constants as C

CLASS_SCALE = [[0.0, "#2ca02c"], [0.5, "#f2e394"], [1.0, "#d62728"]]        # pünktlich (grün) -> zu spät (rot)
REG_SCALE = "Viridis"


def lock_axes(fig, height=None, **layout):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=height, dragmode=False, **layout)
    return fig


def feature_label(names, f):
    unit = dict(C.FEATURES).get(names[f], "")
    return f"{names[f]} [{unit}]" if unit else names[f]


def value_text(task, v):
    return f"{v:.0%} zu spät" if task == "class" else f"{v:.0f} min"


def _scale(task):
    return CLASS_SCALE if task == "class" else REG_SCALE


# --- Ein einzelner Baum (klein) -------------------------------------------------------------------------------------------------------------------------

def tree_layout(tree):
    x = np.zeros(tree.n_nodes)
    counter = 0
    stack = [(0, False)]
    while stack:
        t, done = stack.pop()
        if tree.feature[t] < 0:
            x[t] = counter
            counter += 1
        elif done:
            x[t] = (x[tree.left[t]] + x[tree.right[t]]) / 2.0
        else:
            stack += [(t, True), (int(tree.right[t]), False), (int(tree.left[t]), False)]
    return x, -tree.depth.astype(float)


def build_tree(tree, task, y_range, height=280):
    x, y = tree_layout(tree)
    inner = tree.feature >= 0
    fig = go.Figure()
    ex, ey = [], []
    for t in np.nonzero(inner)[0]:
        for c in (tree.left[t], tree.right[t]):
            ex += [x[t], x[c], None]
            ey += [y[t], y[c], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#9aa0a6", width=1), hoverinfo="skip", showlegend=False))
    vmin, vmax = (0.0, 1.0) if task == "class" else y_range
    size = 6 + 10 * np.sqrt(tree.n / tree.n_total)
    color = np.where(inner, np.nan, tree.value)
    fig.add_trace(go.Scatter(x=x[~inner], y=y[~inner], mode="markers", marker=dict(size=size[~inner], color=color[~inner], colorscale=_scale(task), cmin=vmin, cmax=vmax, line=dict(color="#111111", width=1)),
                             hovertext=[f"Blatt: {value_text(task, tree.value[t])}, n={tree.n[t]}" for t in np.nonzero(~inner)[0]], hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter(x=x[inner], y=y[inner], mode="markers", marker=dict(size=size[inner], color="#ffffff", line=dict(color="#555555", width=1)),
                             hovertext=[f"Split {tree.feature[t]}" for t in np.nonzero(inner)[0]], hoverinfo="text", showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return lock_axes(fig, height, plot_bgcolor="rgba(0,0,0,0)")


# --- Karte des Wald-Mittels ---------------------------------------------------------------------------------------------------------------------------

def build_map(forest, ds, fx, fy, upto=None, sample=None, height=430):
    task = forest.task
    Xtr = ds.X[ds.train]
    ytr = ds.y(task)[ds.train]
    med = np.median(Xtr, axis=0)
    gx = np.round(np.linspace(Xtr[:, fx].min(), Xtr[:, fx].max(), 60), 4)
    gy = np.round(np.linspace(Xtr[:, fy].min(), Xtr[:, fy].max(), 60), 4)
    XX, YY = np.meshgrid(gx, gy)
    grid = np.tile(med, (XX.size, 1))
    grid[:, fx], grid[:, fy] = XX.ravel(), YY.ravel()
    z = np.round(et.predict_value(forest, grid, upto), 3).reshape(XX.shape)
    vmin, vmax = (0.0, 1.0) if task == "class" else (float(np.min(ytr)), float(np.max(ytr)))
    scale = _scale(task)
    fig = go.Figure(go.Heatmap(x=gx, y=gy, z=z, colorscale=scale, zmin=vmin, zmax=vmax, opacity=0.55, showscale=False, hovertemplate="%{z:.2f}<extra></extra>"))
    if task == "class":
        fig.add_trace(go.Contour(x=gx, y=gy, z=z, contours=dict(start=0.5, end=0.5, size=1, coloring="none"), line=dict(color="#111111", width=2), showscale=False, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=np.round(Xtr[:, fx], 4), y=np.round(Xtr[:, fy], 4), mode="markers", marker=dict(size=5, color=ytr, colorscale=scale, cmin=vmin, cmax=vmax, line=dict(color="#333333", width=0.5)),
                             hovertemplate="%{x:.3g} / %{y:.3g}<extra></extra>", showlegend=False))
    if sample is not None:
        fig.add_trace(go.Scatter(x=[sample[fx]], y=[sample[fy]], mode="markers", marker=dict(symbol="star", size=16, color="#ffffff", line=dict(color="#111111", width=2)), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(title=feature_label(ds.names, fx))
    fig.update_yaxes(title=feature_label(ds.names, fy))
    return lock_axes(fig, height)


# --- Wichtigkeit ------------------------------------------------------------------------------------------------------------------------------------

def build_importance(names, imp, height=330):
    order = np.argsort(-imp, kind="stable")
    colors = ["#ff7f0e" if f >= C.N_BASE else "#1f77b4" for f in order]
    fig = go.Figure(go.Bar(x=imp[order], y=[names[f] for f in order], orientation="h", marker_color=colors, text=[f"{imp[f]:.1%}" for f in order], textposition="outside", cliponaxis=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Wichtigkeit (Mittel über die Bäume)", tickformat=".0%", rangemode="tozero")
    return lock_axes(fig, height, showlegend=False).update_layout(margin=dict(l=10, r=60, t=30, b=10))


# --- Testfehler gegen die Zahl der Bäume: Extra Trees gegen Random Forest --------------------------------------------------------------------------------

def _error_axis(fig, task):
    fig.update_yaxes(title="Fehlerquote" if task == "class" else "RMSE [min]", rangemode="tozero", **({"tickformat": ".0%"} if task == "class" else {}))


def build_effort_curve(rows, task, rf_error, current_k, height=380):
    k = [r["n_trees"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=[r["test"] for r in rows], mode="lines+markers", name="Extra Trees", line=dict(color=C.COLORS["extra"])))
    fig.add_hline(y=rf_error, line=dict(color=C.COLORS["exhaustive"], dash="dash"), annotation_text="Random Forest, 30 Bäume", annotation_position="top right")
    fig.add_vline(x=current_k, line=dict(color="#111111", dash="dot"))
    fig.update_xaxes(title="Zahl der Extra-Trees-Bäume")
    _error_axis(fig, task)
    return lock_axes(fig, height, legend=dict(orientation="h", y=1.12))


def build_effort_ratio(rows, height=280):
    k = [r["n_trees"] for r in rows]
    fig = go.Figure(go.Scatter(x=k, y=[100 * r["ratio"] for r in rows], mode="lines+markers", line=dict(color="#9467bd")))
    fig.update_xaxes(title="Zahl der Extra-Trees-Bäume")
    fig.update_yaxes(title="Aufwand ggü. Random Forest [%]", rangemode="tozero")
    return lock_axes(fig, height, showlegend=False)


# --- Bias-Varianz: Extra Trees gegen Random Forest ------------------------------------------------------------------------------------------------------

def build_bias_variance(bv, height=340):
    labels = ["Random Forest", "Extra Trees"]
    bias2 = [bv["rf"]["bias2"], bv["extra"]["bias2"]]
    var = [bv["rf"]["variance"], bv["extra"]["variance"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=bias2, name="Bias²", marker_color="#1f77b4", text=[f"{v:.3g}" for v in bias2], textposition="inside"))
    fig.add_trace(go.Bar(x=labels, y=var, name="Varianz", marker_color="#d62728", text=[f"{v:.3g}" for v in var], textposition="inside"))
    fig.update_yaxes(title="mittlerer quadratischer Fehler", rangemode="tozero")
    return lock_axes(fig, height, barmode="stack", legend=dict(orientation="h", y=1.12))
