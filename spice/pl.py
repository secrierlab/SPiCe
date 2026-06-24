"""Plotting — publication-quality figures from ``adata.uns['spice']``.

Every function reads results previously stored by :mod:`spice.tl` and
returns a ``(fig, ax)`` tuple for further customisation.

Usage::

    import spice

    spice.pl.node_importance(adata, state_map={0: "EPI", 1: "MES"})
    spice.pl.edge_network(adata, state=0)
    spice.pl.auc_per_class(adata)
"""

from __future__ import annotations

import ast

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────

CB_PALETTE = [
    "#0072B2",
    "#D55E00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#E69F00",
    "#CC79A7",
    "#999999",
]

_DEFAULT_STYLE = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Arial, Helvetica, sans-serif",
    "font.weight": "normal",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
}


def _apply_style():
    for k, v in _DEFAULT_STYLE.items():
        mpl.rcParams[k] = v


def _require(adata, key, hint=""):
    s = adata.uns.get("spice", {})
    if key not in s:
        msg = f"adata.uns['spice']['{key}'] not found."
        if hint:
            msg += f" Run spice.tl.{hint} first."
        raise KeyError(msg)
    return s[key]


def _despine(ax):
    """Remove all spines except bottom and left."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ──────────────────────────────────────────────────────────────────────
# 1.  Node importance scatter
# ──────────────────────────────────────────────────────────────────────


def node_importance(
    adata,
    state_map: dict[int, str] | None = None,
    alpha: float = 0.05,
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Scatter plot of node-level feature importance p-values.

    Significant features (p < *alpha*) are shown in red, non-significant
    in grey.

    Parameters
    ----------
    adata
        Annotated data matrix (requires ``spice.tl.explain_nodes``).
    state_map
        Dict mapping integer labels to readable names (e.g.
        ``{0: "EPI", 1: "MES"}``).
    alpha
        Significance threshold.
    save
        File path to save the figure.
    figsize
        Override figure size per panel.

    Returns
    -------
    ``(fig, axes)``
    """
    _apply_style()
    pvals = _require(adata, "node_pvalues", "explain_nodes")
    qvals = _require(adata, "node_qvalues", "explain_nodes")  

    state_map = state_map or {}
    red_cmap = plt.get_cmap("Reds")

    n_states = len(pvals)
    n_cols = min(n_states, 2)
    n_rows = int(np.ceil(n_states / n_cols))
    if figsize is None:
        figsize = (6.5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for idx, (label, row) in enumerate(pvals.iterrows()):
        ax = axes[idx]
        p_arr = row.values.astype(float)
        x = np.arange(len(p_arr))

        colors = []
        sizes = []
        for p in p_arr:
            if p < alpha:
                norm = mcolors.Normalize(vmin=0, vmax=alpha)
                colors.append(red_cmap(1 - norm(p)))
                sizes.append(60)
            else:
                colors.append("#bdbdbd")
                sizes.append(30)

        neg_log_p = -np.log10(p_arr + 1e-300)  # avoid log(0)
        ax.scatter(x, neg_log_p, c=colors, s=sizes, edgecolors="white", linewidths=0.4, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(pvals.columns, rotation=90, fontsize=7)
        name = state_map.get(label, str(label))
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_ylabel("$-\\log_{10}(p)$")
        ax.axhline(-np.log10(alpha), ls="--", lw=0.8, color="#999999", alpha=0.6)
        _despine(ax)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=300)
    return fig, axes


def node_importance_signed(
    adata,
    state_map: dict[int, str] | None = None,
    alpha: float = 0.05,
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
    pos_color: str = "#c0392b",
    neg_color: str = "#2c6fbb",
) -> tuple[plt.Figure, np.ndarray]:
    """Signed scatter plot of node-level feature importance p-values.

    For each node label, features are plotted at ``-log10(p)``. Significant
    features (p < *alpha*) are coloured by the direction of their mean
    effect (red positive, blue negative); non-significant features are grey.

    Parameters
    ----------
    adata
        Annotated data matrix (requires ``spice.tl.explain_nodes``).
    state_map
        Dict mapping integer labels to readable names (e.g.
        ``{0: "EPI", 1: "MES"}``).
    alpha
        Significance threshold.
    save
        File path to save the figure.
    figsize
        Override figure size per panel.
    pos_color
        Colour for significant positive effects.
    neg_color
        Colour for significant negative effects.

    Returns
    -------
    ``(fig, axes)``
    """
    _apply_style()
    pvals = _require(adata, "node_pvalues", "explain_nodes")
    imp = _require(adata, "node_importance", "explain_nodes")
    means = imp.pivot(index="label", columns="feature", values="mean")
    means = means.reindex(index=pvals.index, columns=pvals.columns)
    state_map = state_map or {}

    n_states = len(pvals)
    n_cols = min(n_states, 2)
    n_rows = int(np.ceil(n_states / n_cols))
    if figsize is None:
        figsize = (6.5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for idx, label in enumerate(pvals.index):
        ax = axes[idx]
        p_arr = pvals.loc[label].values.astype(float)
        m_arr = means.loc[label].values.astype(float)
        x = np.arange(len(p_arr))

        sig = p_arr < alpha
        colors = np.where(~sig, "#bdbdbd",
                          np.where(m_arr > 0, pos_color, neg_color))
        neg_log_p = -np.log10(p_arr + 1e-300)  # avoid log(0)

        ax.scatter(
            x, neg_log_p,
            c=colors,
            s=np.where(sig, 60, 30),
            edgecolors=np.where(sig, "black", "white"),
            linewidths=np.where(sig, 0.8, 0.4),
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(pvals.columns, rotation=90, fontsize=7)
        name = state_map.get(label, str(label))
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_ylabel(r"$-\log_{10}(p)$")
        ax.axhline(-np.log10(alpha), ls="--", lw=0.8, color="#999999", alpha=0.6)
        _despine(ax)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker="o", ls="", mfc=pos_color, mec="black",
                   ms=8, label="Positive effect"),
        plt.Line2D([0], [0], marker="o", ls="", mfc=neg_color, mec="black",
                   ms=8, label="Negative effect"),
        plt.Line2D([0], [0], marker="o", ls="", mfc="#bdbdbd", mec="white",
                   ms=8, label="n.s."),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=300)
    return fig, axes

# ──────────────────────────────────────────────────────────────────────
# 2.  Edge interaction network
# ──────────────────────────────────────────────────────────────────────


def edge_network(
    adata,
    state: int | None = None,
    state_map: dict[int, str] | None = None,
    alpha: float = 0.05,
    min_width: float = 0.8,
    max_width: float = 6.0,
    save: str | None = None,
    figsize: tuple[float, float] = (7, 7),
) -> tuple[plt.Figure, plt.Axes] | list[tuple[plt.Figure, plt.Axes]]:
    """Circular network of significant cell-type interactions.

    Edge widths are normalised to the range [*min_width*, *max_width*]
    so that the plot stays readable regardless of raw importance scale.

    Parameters
    ----------
    adata
        Annotated data matrix (requires ``spice.tl.explain_edges``).
    state
        Plot a single state label.  ``None`` plots all states, returning
        a list of (fig, ax) tuples.
    state_map
        Readable label names.
    alpha
        Significance threshold for edges.
    min_width, max_width
        Range for normalised edge widths in points.
    save
        Path prefix — ``"dir/edge"`` produces ``dir/edge_EPI.pdf`` etc.
    figsize
        Figure size.

    Returns
    -------
    ``(fig, ax)`` or list thereof.
    """
    _apply_style()
    edge_dict = _require(adata, "edge_explanations", "explain_edges")
    state_map = state_map or {}

    labels = [state] if state is not None else list(edge_dict.keys())
    outputs = []

    for lbl in labels:
        df = edge_dict[lbl].copy()
        if isinstance(df["cell_type_pair"].iloc[0], str):
            df["cell_type_pair"] = df["cell_type_pair"].apply(ast.literal_eval)
        df[["ct1", "ct2"]] = pd.DataFrame(
            df["cell_type_pair"].tolist(),
            index=df.index,
        )
        df_sig = df[df["p_value"] < alpha]

        name = state_map.get(lbl, str(lbl))
        fig, ax = plt.subplots(figsize=figsize)

        if df_sig.empty:
            ax.text(
                0.5,
                0.5,
                f"No significant edges (p < {alpha})",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="#666666",
            )
            ax.set_title(name, fontsize=14, fontweight="bold")
            ax.axis("off")
            outputs.append((fig, ax))
            continue

        G_net = nx.Graph()
        for _, row in df_sig.iterrows():
            G_net.add_edge(row["ct1"], row["ct2"], weight=row["mean_importance_target"])

        # Use shell layout for cleaner spacing with few nodes,
        # fall back to kamada_kawai for larger graphs
        if G_net.number_of_nodes() <= 8:
            pos = nx.shell_layout(G_net)
        else:
            pos = nx.kamada_kawai_layout(G_net)

        # Normalise edge widths to a sensible range
        weights = np.array([G_net[u][v]["weight"] for u, v in G_net.edges()])
        if weights.max() == weights.min():
            widths = np.full_like(weights, (min_width + max_width) / 2)
        else:
            widths = min_width + (weights - weights.min()) / (weights.max() - weights.min()) * (
                max_width - min_width
            )

        # Colour edges by importance
        edge_cmap = plt.get_cmap("YlOrRd")
        if weights.max() == weights.min():
            edge_colors = [edge_cmap(0.5)] * len(weights)
        else:
            norm = mcolors.Normalize(vmin=weights.min(), vmax=weights.max())
            edge_colors = [edge_cmap(norm(w)) for w in weights]

        nx.draw_networkx_edges(G_net, pos, ax=ax, width=widths, edge_color=edge_colors, alpha=0.7)
        nx.draw_networkx_nodes(
            G_net,
            pos,
            ax=ax,
            node_size=1200,
            node_color="white",
            linewidths=2,
            edgecolors=CB_PALETTE[0],
        )
        nx.draw_networkx_labels(G_net, pos, ax=ax, font_size=9, font_weight="bold")

        # Continuous colorbar legend
        sm = cm.ScalarMappable(
            cmap=edge_cmap,
            norm=mcolors.Normalize(vmin=weights.min(), vmax=weights.max()),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.3, aspect=15, pad=0.02)
        cbar.set_label("Edge importance", fontsize=10)
        cbar.outline.set_visible(False)

        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()

        if save:
            safe = name.replace(" ", "_").replace("/", "_")
            fig.savefig(f"{save}_{safe}.pdf", bbox_inches="tight", dpi=300)
        outputs.append((fig, ax))

    if state is not None:
        return outputs[0]
    # When plotting all states, figures are already shown — return None
    # to avoid Jupyter printing a list of (fig, ax) tuples.
    return None


# ──────────────────────────────────────────────────────────────────────
# 3.  AUC per class
# ──────────────────────────────────────────────────────────────────────


def auc_per_class(
    adata,
    state_map: dict[int, str] | None = None,
    save: str | None = None,
    figsize: tuple[float, float] = (5, 3.5),
) -> tuple[plt.Figure, plt.Axes]:
    """Strip plot of AUC per class across folds.

    Parameters
    ----------
    adata
        Annotated data matrix (requires ``spice.tl.evaluate``).
    state_map
        Readable label names.
    save
        File path to save figure.
    figsize
        Figure size.

    Returns
    -------
    ``(fig, ax)``
    """
    _apply_style()
    df = _require(adata, "auc", "evaluate").copy()
    state_map = state_map or {}

    per_class = df[~df["class"].isin(["micro", "macro"])]
    per_class = per_class.copy()
    per_class["state"] = per_class["class"].map(state_map)
    if per_class["state"].isna().any():
        per_class["state"] = per_class["state"].fillna(per_class["class"].astype(str))

    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    order = sorted(
        per_class["state"].unique(),
        key=lambda s: list(state_map.values()).index(s) if s in state_map.values() else 999,
    )
    sns.stripplot(
        data=per_class,
        x="state",
        y="AUC",
        jitter=True,
        palette=CB_PALETTE[: len(order)],
        size=8,
        ax=ax,
        order=order,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.8,
        zorder=3,
    )

    means = per_class.groupby("state")["AUC"].mean()
    for i, state in enumerate(order):
        if state in means.index:
            ax.hlines(means[state], i - 0.3, i + 0.3, colors="k", linewidth=2, zorder=4)

    ax.set_ylabel("AUC")
    ax.set_xlabel("")
    ax.set_box_aspect(1)
    _despine(ax)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=300)
    return fig, ax


# ──────────────────────────────────────────────────────────────────────
# 4.  Regression R²
# ──────────────────────────────────────────────────────────────────────


def regression_r2(
    adata,
    save: str | None = None,
    figsize: tuple[float, float] = (5, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Error-bar plot of R² across folds.

    Parameters
    ----------
    adata
        Annotated data matrix (requires ``spice.tl.evaluate`` with regression).
    save
        File path.
    figsize
        Figure size.

    Returns
    -------
    ``(fig, ax)``
    """
    _apply_style()
    df = _require(adata, "regression", "evaluate")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        df["fold"],
        df["R2"],
        "o",
        color=CB_PALETTE[0],
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=0.5,
        zorder=3,
    )
    ax.axhline(df["R2"].mean(), ls="--", color="k", lw=0.8, alpha=0.6)
    ax.set_xlabel("Fold")
    ax.set_ylabel(r"$R^2$")
    ax.set_box_aspect(1)
    _despine(ax)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=300)
    return fig, ax


# ──────────────────────────────────────────────────────────────────────
# 5.  Baseline comparison
# ──────────────────────────────────────────────────────────────────────


def baseline(
    adata,
    save: str | None = None,
    figsize: tuple[float, float] = (7, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal bar chart comparing GNN against baseline classifiers.

    Parameters
    ----------
    adata
        Annotated data matrix (requires ``spice.tl.run_baseline``).
    save
        File path.
    figsize
        Figure size.

    Returns
    -------
    ``(fig, ax)``
    """
    _apply_style()
    summary, _ = _require(adata, "baseline", "run_baseline")
    summary = summary.sort_values("mean_AUC", ascending=True)

    colors = [
        CB_PALETTE[0] if ("Neighbour" in m or "Neighbor" in m or "GNN" in m) else CB_PALETTE[7]
        for m in summary["model"]
    ]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        summary["model"],
        summary["mean_AUC"],
        xerr=summary["std"],
        color=colors,
        edgecolor="white",
        capsize=4,
        linewidth=0.8,
    )
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.text(
            row["mean_AUC"] + row["std"] + 0.005,
            i,
            f"{row['mean_AUC']:.3f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("ROC AUC")
    ax.set_xlim(0.5, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=300)
    return fig, ax


# ──────────────────────────────────────────────────────────────────────
# 6.  MERFISH-style AUC dot plot
# ──────────────────────────────────────────────────────────────────────


def auc_dotplot(
    df_auc: pd.DataFrame,
    celltype_col: str = "Cell_Type",
    auc_col: str = "AUC",
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Dot plot of AUC per cell type (standalone, does not read from adata).

    Useful for MERFISH-style per-cell-type analyses where each cell type
    is trained independently.

    Parameters
    ----------
    df_auc
        DataFrame with at least *celltype_col* and *auc_col*.
    celltype_col, auc_col
        Column names.
    save
        File path.
    figsize
        Figure size (auto-scaled by default).

    Returns
    -------
    ``(fig, ax)``
    """
    _apply_style()
    grouped = (
        df_auc.groupby(celltype_col)[auc_col]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
    )
    if figsize is None:
        figsize = (max(6, len(grouped) * 0.45), 3.5)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(grouped))
    ax.errorbar(
        x,
        grouped["mean"],
        yerr=grouped["std"],
        fmt="o",
        color=CB_PALETTE[0],
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=0.5,
        elinewidth=1,
        capsize=3,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=90, fontsize=8)
    ax.set_ylabel("AUC")
    ax.set_box_aspect(1)
    _despine(ax)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=300)
    return fig, ax


# ──────────────────────────────────────────────────────────────────────
# 7.  Multi-run comparison (feature modes, split strategies)
# ──────────────────────────────────────────────────────────────────────


def compare_runs(
    auc_dfs: dict[str, pd.DataFrame],
    state_map: dict[int, str] | None = None,
    save: str | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Error-bar plot comparing AUC across multiple runs.

    Useful for comparing feature modes or split strategies by passing a
    dictionary of AUC DataFrames (e.g. from separate
    ``adata.uns['spice']['auc']`` runs).

    Parameters
    ----------
    auc_dfs
        ``{"TME": df_tme, "Intrinsic": df_intr, ...}`` where each value
        is a DataFrame as produced by :func:`spice.tl.evaluate`.
    state_map
        Readable label names.
    save
        File path.
    figsize
        Figure size.

    Returns
    -------
    ``(fig, ax)``
    """
    _apply_style()
    state_map = state_map or {}

    records = []
    for run_name, df in auc_dfs.items():
        per = df[~df["class"].isin(["micro", "macro"])].copy()
        per["state"] = per["class"].map(state_map).fillna(per["class"].astype(str))
        per["run"] = run_name
        records.append(per)

    combined = pd.concat(records, ignore_index=True)
    grouped = combined.groupby(["state", "run"])["AUC"].agg(["mean", "std"]).reset_index()

    states = sorted(
        grouped["state"].unique(),
        key=lambda s: list(state_map.values()).index(s) if s in state_map.values() else 999,
    )
    runs = list(auc_dfs.keys())
    x_pos = np.arange(len(runs))
    offset = 0.18

    fig, ax = plt.subplots(figsize=figsize)
    for i, state in enumerate(states):
        sub = grouped[grouped["state"] == state].set_index("run")
        # reindex to guarantee ordering
        sub = sub.reindex(runs).reset_index()
        ax.errorbar(
            x_pos + (i - len(states) / 2) * offset,
            sub["mean"],
            yerr=sub["std"],
            fmt="o",
            label=state,
            color=CB_PALETTE[i % len(CB_PALETTE)],
            elinewidth=1.5,
            capsize=3,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=3,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(runs, rotation=25, ha="right")
    ax.set_ylabel("AUC")
    ax.legend(title="State", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    _despine(ax)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=300)
    return fig, ax
