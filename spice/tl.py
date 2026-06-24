"""Tools — analysis functions that store results in ``adata.uns['spice']``.

Every function in this module modifies *adata* in place and returns ``None``
by default.  Set ``copy=True`` to operate on (and return) a copy instead.

Usage::

    import spice

    spice.tl.assign_state_labels(adata, score_key="EMT_hallmarks", n_states=4)
    spice.tl.build_graph(adata, celltype_key="celltype_minor")
    spice.tl.cross_validate(adata, num_epochs=500)
    spice.tl.evaluate(adata)
    spice.tl.explain_nodes(adata, n_explanations=100)

    # results live in adata.uns['spice']
    adata.uns['spice']['auc']
    adata.uns['spice']['node_pvalues']
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from spice._graph import (
    assign_spatial_blocks_core,
    build_graph_core,
    create_edge_index,
    filter_nan_coords,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _spice(adata: AnnData) -> dict:
    """Return (and lazily create) the ``adata.uns['spice']`` dict."""
    adata.uns.setdefault("spice", {})
    return adata.uns["spice"]


def _require(adata: AnnData, key: str, hint: str = ""):
    """Raise if *key* is missing from ``adata.uns['spice']``."""
    s = _spice(adata)
    if key not in s:
        msg = f"adata.uns['spice']['{key}'] not found."
        if hint:
            msg += f" Run spice.tl.{hint} first."
        raise KeyError(msg)
    return s[key]


# ──────────────────────────────────────────────────────────────────────
# 1.  Preprocessing
# ──────────────────────────────────────────────────────────────────────

def assign_state_labels(
    adata: AnnData,
    score_key: str = "EMT_hallmarks",
    label_key: str = "state_label",
    n_states: Literal[2, 4] = 2,
    tumour_mask_key: str | None = "subclone",
    copy: bool = False,
) -> AnnData | None:
    """Discretise a continuous cell-state score into categorical labels.

    Writes integer labels to ``adata.obs[label_key]``.  Non-tumour cells
    (NaN in *tumour_mask_key*) receive a label of ``-1`` and are excluded
    from GNN training/evaluation.

    Parameters
    ----------
    adata
        Annotated data matrix with ``adata.obs[score_key]``.
    score_key
        Column holding the continuous score.
    label_key
        Destination column name in ``adata.obs``.
    n_states
        ``2`` (median split) or ``4`` (quartiles).
    tumour_mask_key
        Column used to identify tumour cells.  ``None`` labels all cells.
    copy
        If ``True``, return a modified copy; otherwise modify in place.
    """
    adata = adata.copy() if copy else adata

    labels = np.full(adata.n_obs, -1, dtype=int)

    if tumour_mask_key is not None and tumour_mask_key in adata.obs.columns:
        mask = adata.obs[tumour_mask_key].notna().values
    else:
        mask = np.ones(adata.n_obs, dtype=bool)

    scores = adata.obs.loc[mask, score_key].values.astype(float)

    if n_states == 2:
        thr = np.nanmedian(scores)
        labels[mask] = (scores > thr).astype(int)
    elif n_states == 4:
        q25, q50, q75 = np.nanpercentile(scores, [25, 50, 75])
        state = np.zeros(len(scores), dtype=int)
        state[scores > q25] = 1
        state[scores > q50] = 2
        state[scores > q75] = 3
        labels[mask] = state
    else:
        raise ValueError(f"n_states must be 2 or 4, got {n_states}")

    adata.obs[label_key] = labels
    return adata if copy else None


# ──────────────────────────────────────────────────────────────────────
# 2.  Graph construction
# ──────────────────────────────────────────────────────────────────────

def build_graph(
    adata: AnnData,
    spatial_key: str = "spatial",
    n_neighbors: int = 10,
    celltype_key: str = "celltype_minor",
    label_key: str = "state_label",
    score_key: str | None = "EMT_hallmarks",
    sample_key: str | None = None,
    n_blocks: int = 4,
    extra_obs_keys: list[str] | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Build a spatial k-NN graph and store it in ``adata.uns['spice']``.

    Cells with NaN/Inf spatial coordinates are dropped in place.  The
    resulting graph, fitted :class:`~sklearn.preprocessing.LabelBinarizer`
    and run parameters are written to ``adata.uns['spice']``.

    Stored keys
    -----------
    ``adata.uns['spice']['graph']``
        NetworkX graph.
    ``adata.uns['spice']['label_binarizer']``
        Fitted LabelBinarizer.
    ``adata.uns['spice']['params']``
        Dict of key parameters for downstream functions.

    Parameters
    ----------
    adata
        Annotated data matrix.
    spatial_key
        Key in ``adata.obsm`` for 2-D coordinates.
    n_neighbors
        Number of nearest neighbors (excluding self).
    celltype_key
        Column in ``adata.obs`` with cell type annotations.
    label_key
        Column in ``adata.obs`` with discrete state labels.
    score_key
        Column holding a continuous state score (``None`` to skip).
    sample_key
        Column for sample/FOV identifiers (for inductive splitting).
    n_blocks
        Grid divisions for spatial block assignment.
    extra_obs_keys
        Additional ``adata.obs`` columns stored as node attributes.
    copy
        Operate on a copy rather than in place.
    """
    adata = adata.copy() if copy else adata
    extra_obs_keys = extra_obs_keys or []

    # drop NaN coords in place
    filter_nan_coords(adata, spatial_key)

    G, lb = build_graph_core(
        adata, spatial_key, n_neighbors, celltype_key,
        label_key, score_key, sample_key, extra_obs_keys,
    )
    assign_spatial_blocks_core(G, n_blocks=n_blocks)

    s = _spice(adata)
    s["graph"] = G
    s["label_binarizer"] = lb
    s["params"] = {
        "spatial_key": spatial_key,
        "n_neighbors": n_neighbors,
        "celltype_key": celltype_key,
        "label_key": label_key,
        "score_key": score_key,
        "sample_key": sample_key,
        "n_blocks": n_blocks,
    }

    return adata if copy else None


# ──────────────────────────────────────────────────────────────────────
# 3.  Feature preparation
# ──────────────────────────────────────────────────────────────────────

def prepare_features(
    adata: AnnData,
    mode: Literal["celltype", "intrinsic", "combined"] = "celltype",
    pca_key: str = "X_pca",
    n_pcs: int = 7,
    scale_pca: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """Build node features and store them in ``adata.obsm['X_spice']``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    mode
        ``"celltype"`` (one-hot), ``"intrinsic"`` (PCA) or ``"combined"``.
    pca_key
        Key in ``adata.obsm`` for PCA coordinates.
    n_pcs
        Number of principal components.
    scale_pca
        Min-max scale PCA features to [0, 1].
    copy
        Operate on a copy.
    """
    adata = adata.copy() if copy else adata
    params = _spice(adata).get("params", {})
    celltype_key = params.get("celltype_key", "celltype_minor")

    from spice._features import prepare_features as _prepare

    adata.obsm["X_spice"] = _prepare(
        adata, mode=mode, celltype_key=celltype_key,
        pca_key=pca_key, n_pcs=n_pcs, scale_pca=scale_pca,
    )
    _spice(adata)["feature_mode"] = mode
    return adata if copy else None


# ──────────────────────────────────────────────────────────────────────
# 4.  Cross-validation
# ──────────────────────────────────────────────────────────────────────

def cross_validate(
    adata: AnnData,
    feature_mode: Literal["celltype", "intrinsic", "combined"] = "celltype",
    pca_key: str = "X_pca",
    n_pcs: int = 7,
    continuous_y: bool = False,
    inductive_split: bool = True,
    num_folds: int = 5,
    hidden_dim1: int = 16,
    hidden_dim2: int = 32,
    dropout: float = 0.5,
    learning_rate: float = 0.01,
    num_epochs: int = 500,
    class_weights: bool = True,
    device: str | None = None,
    verbose: bool = True,
    return_last: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """Run k-fold cross-validation and store results in ``adata.uns['spice']``.

    Requires :func:`build_graph` to have been called first.

    Stored keys
    -----------
    ``adata.uns['spice']['folds']``
        List of PyG Data objects (one per fold).
    ``adata.uns['spice']['cv_results']``
        Dict with ``all_predicted``, ``all_true``, ``models`` and
        ``fold_performances``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    feature_mode
        Node feature strategy.
    pca_key, n_pcs
        PCA settings (used when *feature_mode* includes intrinsic).
    continuous_y
        Regression (``True``) or classification (``False``).
    inductive_split
        Spatial-block/sample split (``True``) or random node split.
    num_folds
        Number of cross-validation folds.
    hidden_dim1, hidden_dim2, dropout
        GCN architecture hyper-parameters.
    learning_rate, num_epochs
        Training hyper-parameters.
    class_weights
        Inverse-frequency class weighting for classification.
    device
        Torch device (``None`` for auto).
    verbose
        Print per-fold progress.
    return_last
        If ``True`` (default, matching original implementation), return
        the last evaluated epoch's predictions.  Set to ``False`` to
        return predictions from the best-performing epoch instead.
    copy
        Operate on a copy.
    """
    adata = adata.copy() if copy else adata
    G = _require(adata, "graph", "build_graph")
    params = _spice(adata)["params"]

    from spice.dataset import SpatialGNNDataset
    from spice.models import SpatialGCN
    from spice.train import train_model as _train

    folds = []
    for i in range(1, num_folds + 1):
        ds = SpatialGNNDataset(
            adata, G,
            feature_mode=feature_mode,
            celltype_key=params["celltype_key"],
            label_key=params["label_key"],
            score_key=params["score_key"],
            pca_key=pca_key,
            n_pcs=n_pcs,
            continuous_y=continuous_y,
            inductive_split=inductive_split,
            fold_idx=i,
            num_folds=num_folds,
        )
        folds.append(ds[0])

    all_predicted, all_true, models, fold_perfs = [], [], [], []

    for fold_idx, fold_data in enumerate(folds):
        if verbose:
            print(f"\n{'='*50}\nFold {fold_idx + 1}/{num_folds}\n{'='*50}")

        model = SpatialGCN.from_data(
            fold_data, hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2, dropout=dropout,
        )
        result = _train(
            model, fold_data, num_epochs=num_epochs,
            learning_rate=learning_rate, class_weights=class_weights,
            device=device, verbose=verbose, return_last=return_last,
        )
        fold_perfs.append(result["best_performance"])
        all_predicted.append(result["predictions"].cpu().numpy())
        all_true.append(fold_data.y[fold_data.test_mask].cpu().numpy())
        models.append(result["model"])

    if verbose:
        print(f"\nMean performance: {np.mean(fold_perfs):.4f} ± {np.std(fold_perfs):.4f}")

    s = _spice(adata)
    s["folds"] = folds
    s["cv_results"] = {
        "all_predicted": all_predicted,
        "all_true": all_true,
        "models": models,
        "fold_performances": fold_perfs,
    }
    s["feature_mode"] = feature_mode
    s["continuous_y"] = continuous_y

    return adata if copy else None


# ──────────────────────────────────────────────────────────────────────
# 5.  Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate(
    adata: AnnData,
    copy: bool = False,
) -> AnnData | None:
    """Evaluate cross-validation results and store metrics.

    Reads from ``adata.uns['spice']['cv_results']`` and writes a
    :class:`~pandas.DataFrame` to either ``adata.uns['spice']['auc']``
    (classification) or ``adata.uns['spice']['regression']`` (regression).

    Requires :func:`cross_validate` to have been called first.
    """
    adata = adata.copy() if copy else adata
    s = _spice(adata)
    cv = _require(adata, "cv_results", "cross_validate")
    continuous = s.get("continuous_y", False)

    from spice._evaluation import eval_classification_core, eval_regression_core

    if continuous:
        df = eval_regression_core(cv["all_predicted"], cv["all_true"])
        s["regression"] = df
        print(f"Mean R²: {df['R2'].mean():.4f}  |  "
              f"Mean correlation: {df['correlation'].mean():.4f}")
    else:
        n_cls = cv["all_predicted"][0].shape[1] if cv["all_predicted"][0].ndim > 1 else 2
        df = eval_classification_core(cv["all_predicted"], cv["all_true"], n_cls)
        s["auc"] = df

    return adata if copy else None


# ──────────────────────────────────────────────────────────────────────
# 6.  Explanations
# ──────────────────────────────────────────────────────────────────────

def explain_nodes(
    adata: AnnData,
    n_explanations: int = 50,
    fold_index: int | None = None,
    mask_type: str = "node",
    seed: int | None = None,
    ig_steps: int = 50,
    top_n: int = 15,
    verbose: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """Node feature importance via Integrated Gradients, pooled across folds.
 
    Each fold's attributions are standardized against that fold's own
    label-shuffled null, then the standardized nodes from every fold are pooled.
    Per (label, feature):
 
    - ``mean``              standardized mean (null-SD units); its sign says whether
                            the feature reads higher or lower than baseline. Plot this.
    - ``sign_consistency``  fraction of folds agreeing with that sign (scale-free).
    - ``p_value``           two-sided Mann-Whitney, pooled label nodes vs pooled null.
    - ``q_value``           Benjamini-Hochberg FDR across the grid.
 
    Stored in ``adata.uns['spice']``: ``node_importance`` (tidy table),
    ``node_pvalues`` / ``node_qvalues`` (matrices), ``node_attributions`` (raw IG).
    """
    adata = adata.copy() if copy else adata
    s = _spice(adata)
    cv = _require(adata, "cv_results", "cross_validate")
    folds = _require(adata, "folds", "cross_validate")
    fold_ids = range(len(folds)) if fold_index is None else [fold_index]
 
    from spice._explanations import explain_nodes_core
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests
 
    feature_names = _baseline_feature_names(s, folds)
 
    def mwu(label_vals, null_vals):
        a, b = label_vals.dropna().values, null_vals.dropna().values
        try:
            return mannwhitneyu(a, b, alternative="two-sided").pvalue
        except ValueError:          # constant feature 
            return np.nan
 
    # 1. Run each fold; standardize its nodes against its own null.
    label_z, null_z, raw = {}, [], {}
    for fi in fold_ids:
        data, model = folds[fi], cv["models"][fi]
        preds = torch.tensor(cv["all_predicted"][fi])
        pred_labels = preds.argmax(1) if preds.ndim > 1 else (preds > 0.5).long()
        test_df = pd.DataFrame({
            "Indices": torch.where(data.test_mask)[0].tolist(),
            "Label": pred_labels.numpy(),
        })
 
        attributions, null = explain_nodes_core(
            model, data, test_df, n_explanations, mask_type,
            feature_names=feature_names, seed=seed, ig_steps=ig_steps,
        )
        mu, sd = null.mean(0), null.std(0).replace(0, np.nan)   # guard for constant features
        null_z.append((null - mu) / sd)
        for lab, df in attributions.items():
            raw.setdefault(lab, []).append(df)
            if not df.empty:
                label_z.setdefault(lab, []).append((df - mu) / sd)
 
    null_all = pd.concat(null_z, ignore_index=True)
 
    # 2. Aggregate per (label, feature) over the pooled standardized nodes.
    rows = []
    for lab, z_list in label_z.items():
        Z = pd.concat(z_list, ignore_index=True)                  # pooled label nodes
        per_fold = pd.concat([z.mean(0) for z in z_list], axis=1)  # features × folds
        mean = Z.mean(0)
 
        valid = per_fold.notna()
        agree = np.sign(per_fold).eq(np.sign(mean), axis=0) & valid
        rows.append(pd.DataFrame({
            "label": lab,
            "feature": Z.columns,
            "mean": mean.values,
            "sign_consistency": (agree.sum(1) / valid.sum(1)).values,
            "p_value": [mwu(Z[f], null_all[f]) for f in Z.columns],
        }))
 
    imp = pd.concat(rows, ignore_index=True)
 
    # 3. FDR across the grid, then sort by |mean| within each label.
    p = imp["p_value"].values
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    imp["q_value"] = q
    imp = (imp.assign(a=imp["mean"].abs())
              .sort_values(["label", "a"], ascending=[True, False])
              .drop(columns="a").reset_index(drop=True))
 
    if verbose:
        cols = ["feature", "mean", "sign_consistency", "p_value", "q_value"]
        print("\nTop node features per label (standardized mean across folds):")
        for lab in imp["label"].unique():
            print(f"\n  Label {lab}")
            print(imp[imp["label"] == lab][cols].head(top_n).to_string(index=False))
 
    s["node_importance"] = imp
    s["node_pvalues"] = imp.pivot(index="label", columns="feature", values="p_value")
    s["node_qvalues"] = imp.pivot(index="label", columns="feature", values="q_value")
    s["node_attributions"] = {lab: pd.concat(dfs, ignore_index=True) for lab, dfs in raw.items()}
    return adata if copy else None


def explain_edges(
    adata: AnnData,
    n_explanations: int = 50,
    fold_index: int = 0,
    copy: bool = False,
) -> AnnData | None:
    """Compute edge-level explanations via GNNExplainer.

    Stored keys
    -----------
    ``adata.uns['spice']['edge_explanations']``
        Dict mapping each label to a DataFrame of cell-type-pair
        importances and p-values.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_explanations
        Number of nodes to explain per label.
    fold_index
        Which fold's model and data to use (0-based).
    copy
        Operate on a copy.
    """
    adata = adata.copy() if copy else adata
    s = _spice(adata)
    cv = _require(adata, "cv_results", "cross_validate")
    folds = _require(adata, "folds", "cross_validate")
    G = _require(adata, "graph", "build_graph")
    params = s["params"]

    from spice._explanations import explain_edges_core

    results = explain_edges_core(
        cv["models"][fold_index], folds[fold_index], G,
        n_explanations, params["celltype_key"],
    )

    s["edge_explanations"] = results
    return adata if copy else None


# ──────────────────────────────────────────────────────────────────────
# 7.  Baseline comparison
# ──────────────────────────────────────────────────────────────────────

def run_baseline(
    adata: AnnData,
    k: int = 10,
    verbose: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """Compare GNN against RF and MLP baselines.

    Stored keys
    -----------
    ``adata.uns['spice']['baseline']``
        Tuple of ``(summary_df, results_dict)``.

    Requires :func:`cross_validate` to have been called first.
    """
    adata = adata.copy() if copy else adata
    cv = _require(adata, "cv_results", "cross_validate")
    folds = _require(adata, "folds", "cross_validate")
    G = _require(adata, "graph", "build_graph")

    from spice._baseline import run_baseline_core

    summary, results = run_baseline_core(
        folds, G, k, cv["fold_performances"], verbose,
    )
    _spice(adata)["baseline"] = (summary, results)
    return adata if copy else None


def _baseline_feature_names(s: dict, folds: list) -> list[str]:
    """Column labels for the baseline feature matrix (= X_spice), per feature_mode."""
    mode = s.get("feature_mode", "celltype")
    lb = s.get("label_binarizer")
    celltype = list(lb.classes_) if lb is not None else []
    n_feat = folds[0].x.shape[1]

    if mode == "celltype":
        names = celltype
    elif mode == "intrinsic":
        names = [f"PC{i+1}" for i in range(n_feat)]
    elif mode == "combined":
        names = celltype + [f"PC{i+1}" for i in range(n_feat - len(celltype))]
    else:
        names = [f"feat_{i}" for i in range(n_feat)]

    if len(names) != n_feat:
        raise ValueError(
            f"Derived {len(names)} feature names but X has {n_feat} columns "
            f"(feature_mode={mode!r}). Pass feature_names explicitly."
        )
    return names


def explain_baselines(
    adata: AnnData,
    k: int | None = None,
    n_repeats: int = 10,
    feature_names: list[str] | None = None,
    top_n: int = 10,
    verbose: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """Per-label permutation feature importance for the RF + Neighbours baseline,
    averaged across folds.

    For each state label, the test set is restricted to cells of that label and
    permutation importance is computed on that subset.

    Stored keys
    -----------
    ``adata.uns['spice']['baseline_importances']``
        A ``features × labels`` DataFrame of mean permutation importance scores.

    Parameters
    ----------
    adata
        Annotated data matrix.
    k
        Number of spatial neighbours for the neighbour augmentation.
        If ``None``, falls back to ``params['n_neighbors']`` from graph-build time.
    
    feature_names
        Names for the base feature block (the ``_nbr`` columns are derived
        automatically). If ``None``, derived from ``feature_mode``
        (cell-type classes and/or PC labels).
    top_n
        How many top features to print per label when ``verbose``.
    verbose
        Print the top-``top_n`` features per label.
    copy
        Operate on a copy.
    """
    adata = adata.copy() if copy else adata
    s = _spice(adata)
    folds = _require(adata, "folds", "cross_validate")
    graph = _require(adata, "graph", "build_graph")

    if k is None:
        k = s.get("params", {}).get("n_neighbors")
        if k is None:
            raise ValueError(
                "No `k` provided and none stored in params; pass k explicitly."
            )

    if feature_names is None:
        feature_names = _baseline_feature_names(s, folds)

    from spice._explanations import explain_baselines_core

    importances = explain_baselines_core(
        fold_datasets=folds,
        graph=graph,
        k=k,
        feature_names=feature_names,
        top_n=top_n,
        verbose=verbose,
    )

    s["baseline_importances"] = importances
    return adata if copy else None
    
# ──────────────────────────────────────────────────────────────────────
# 8.  Sanitise for saving
# ──────────────────────────────────────────────────────────────────────

def sanitize(
    adata: AnnData,
    copy: bool = False,
) -> AnnData | None:
    """Make ``adata`` safe to save with :meth:`~anndata.AnnData.write_h5ad`.

    Several objects stored in ``adata.uns['spice']`` during a SPiCe run
    are not HDF5-serializable (NetworkX graphs, PyTorch models, PyG Data
    objects, sklearn estimators).  This function strips or converts them
    while keeping all scientific results (AUC tables, p-values, attribution
    DataFrames, baseline summaries and run parameters).

    Objects **removed**:

    - ``graph`` — NetworkX graph
    - ``label_binarizer`` — sklearn LabelBinarizer (cell-type classes
      are preserved as a list in ``params['celltype_classes']``)
    - ``folds`` — list of PyG Data objects
    - ``cv_results['models']`` — list of trained SpatialGCN instances
    - ``cv_results['all_predicted']`` — list of variable-length arrays
    - ``cv_results['all_true']`` — list of variable-length arrays

    Objects **kept** (already serializable or converted):

    - ``params`` — dict of scalars/strings
    - ``cv_results['fold_performances']`` — numpy array of per-fold metrics
    - ``auc`` / ``regression`` — pandas DataFrames
    - ``node_attributions`` — converted from ``{label: DataFrame}`` to a
      single concatenated DataFrame with a ``label`` column
    - ``node_pvalues`` — DataFrame
    - ``edge_explanations`` — converted from ``{label: DataFrame}`` to a
      single concatenated DataFrame with a ``label`` column
    - ``baseline`` — summary DataFrame (the per-fold dict is dropped)
    - ``feature_mode``, ``continuous_y`` — scalars

    Parameters
    ----------
    adata
        Annotated data matrix.
    copy
        Operate on a copy.
    """
    adata = adata.copy() if copy else adata
    s = adata.uns.get("spice", {})
    if not s:
        return adata if copy else None

    # Preserve cell-type classes before dropping the binarizer
    lb = s.pop("label_binarizer", None)
    if lb is not None:
        s.setdefault("params", {})["celltype_classes"] = list(lb.classes_)

    # Drop non-serializable objects
    s.pop("graph", None)
    s.pop("folds", None)

    # Strip non-serializable parts from cv_results.
    # - models: PyTorch modules
    # - all_predicted / all_true: lists of variable-length arrays that
    #   numpy cannot stack into a regular ndarray
    # fold_performances (list of floats) is kept as a numpy array.
    cv = s.get("cv_results")
    if cv is not None:
        cv.pop("models", None)
        cv.pop("all_predicted", None)
        cv.pop("all_true", None)
        if "fold_performances" in cv:
            cv["fold_performances"] = np.asarray(cv["fold_performances"])

    # Convert node_attributions: {label: DataFrame} → single DataFrame
    node_attr = s.get("node_attributions")
    if isinstance(node_attr, dict):
        parts = []
        for label, df in node_attr.items():
            df = df.copy()
            df.insert(0, "label", label)
            parts.append(df)
        if parts:
            s["node_attributions"] = pd.concat(parts, ignore_index=True)

    # Convert edge_explanations: {label: DataFrame} → single DataFrame
    edge_expl = s.get("edge_explanations")
    if isinstance(edge_expl, dict):
        parts = []
        for label, df in edge_expl.items():
            df = df.copy()
            df.insert(0, "label", label)
            parts.append(df)
        if parts:
            s["edge_explanations"] = pd.concat(parts, ignore_index=True)

    # Ensure cell_type_pair column is string-typed (older runs stored
    # tuples which h5py cannot serialise).
    edge_df = s.get("edge_explanations")
    if isinstance(edge_df, pd.DataFrame) and "cell_type_pair" in edge_df.columns:
        edge_df["cell_type_pair"] = edge_df["cell_type_pair"].astype(str)

    # Ensure the "class" column in auc DataFrame is string-typed so
    # h5py can serialise it (older runs stored integers mixed with
    # "micro"/"macro" strings, giving an object column).
    auc_df = s.get("auc")
    if isinstance(auc_df, pd.DataFrame) and "class" in auc_df.columns:
        auc_df["class"] = auc_df["class"].astype(str)

    # Flatten baseline tuple → keep only the summary DataFrame
    baseline = s.get("baseline")
    if isinstance(baseline, tuple):
        s["baseline"] = baseline[0]  # summary DataFrame

    return adata if copy else None

