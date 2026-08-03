"""Internal baseline classifier comparison routines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier


def run_baseline_core(
    fold_datasets: list,
    graph,
    k: int,
    gnn_results: list[float] | None,
    verbose: bool,
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    results: dict[str, list[float]] = {
        "RF": [], "RF + Neighbours": [],
        "MLP": [], "MLP + Neighbours": [],
    }

    # X, y and spatial coordinates come from the same adata/graph for every
    # fold and don't depend on the train/test split, so the neighbour-
    # augmented feature matrix is built once and reused — only the masks
    # (and therefore the train/test slices and fitted classifiers) differ
    # per fold.
    X, y, coords, _, _ = _extract(fold_datasets[0], graph)
    X_aug = _augment(X, coords, k)

    for fold_data in fold_datasets:
        train_mask = fold_data.train_mask.cpu().numpy()
        test_mask = fold_data.test_mask.cpu().numpy()
        nc = len(np.unique(y[train_mask]))
        Xtr, Xte = X[train_mask], X[test_mask]
        ytr, yte = y[train_mask], y[test_mask]
        Xtr_a, Xte_a = X_aug[train_mask], X_aug[test_mask]

        for name, clf, Xtr_, Xte_ in [
            ("RF", RandomForestClassifier(100, random_state=42, n_jobs=-1), Xtr, Xte),
            ("RF + Neighbours", RandomForestClassifier(100, random_state=42, n_jobs=-1), Xtr_a, Xte_a),
            ("MLP", MLPClassifier((64, 32), max_iter=500, random_state=42), Xtr, Xte),
            ("MLP + Neighbours", MLPClassifier((64, 32), max_iter=500, random_state=42), Xtr_a, Xte_a),
        ]:
            clf.fit(Xtr_, ytr)
            if nc == 2:
                a = roc_auc_score(yte, clf.predict_proba(Xte_)[:, 1])
            else:
                a = roc_auc_score(yte, clf.predict_proba(Xte_), multi_class="ovr")
            results[name].append(a)

    if gnn_results is not None:
        results["GNN"] = gnn_results

    rows = []
    for name, aucs in results.items():
        m, s = np.mean(aucs), np.std(aucs)
        if verbose:
            print(f"  {name:20s}: {m:.4f} ± {s:.4f}")
        rows.append({"model": name, "mean_AUC": m, "std": s})

    return pd.DataFrame(rows).sort_values("mean_AUC", ascending=False), results


def _extract(fold_data, graph):
    tm = fold_data.train_mask.cpu().numpy()
    te = fold_data.test_mask.cpu().numpy()
    X = fold_data.x.cpu().numpy()
    y = fold_data.y.cpu().numpy()
    nodes = list(graph.nodes())
    node_data = graph.nodes
    coords = np.array([[node_data[n]["array_row"], node_data[n]["array_col"]]
                        for n in nodes])
    return X, y, coords, tm, te


def _augment(X, coords, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(coords)
    _, idx = nbrs.kneighbors(coords)
    # Vectorised neighbour feature sum (replaces per-cell Python loop)
    nmean = X[idx[:, 1:]].sum(axis=1)
    return np.hstack([X, nmean])


