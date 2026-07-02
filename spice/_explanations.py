"""Internal explanation routines (Integrated Gradients, GNNExplainer)."""

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, ttest_ind
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
# `shap` is imported lazily inside explain_baselines_core (the only place
# that uses it) since it's a heavy, optional import — this keeps
# `spice.tl.explain_nodes` / `explain_edges` free of that cost.



def explain_nodes_core(
    model, data, test_labels_index, n_explanations, mask_type,
    feature_names=None, seed=None, ig_steps=50,
):
    """IG node attributions per label + a label-shuffled null.
 
    Returns
    -------
    attributions : dict{label: DataFrame}   rows = explained nodes, cols = features
    perm_df      : DataFrame                 null attributions (shuffled targets)
    """
    from captum.attr import IntegratedGradients
    from torch_geometric.nn import to_captum_input, to_captum_model
 
    rng = np.random.default_rng(seed)
    unique_labels = test_labels_index["Label"].unique()
    inputs, fwd_args = to_captum_input(data.x, data.edge_index, mask_type, data.edge_attr)
    y = data.y.cpu().numpy()
 
    def attribute(idx, target):
        idx, target = int(idx), int(target)
        ig = IntegratedGradients(to_captum_model(model, mask_type, idx))
        attr = ig.attribute(inputs=inputs, target=target, n_steps=ig_steps,
                            additional_forward_args=fwd_args, internal_batch_size=1)
        return pd.DataFrame(attr[0].squeeze().detach().cpu().numpy()).sum(axis=0).values
 
    def sample(idx_pool):
        return rng.choice(idx_pool, min(n_explanations, len(idx_pool)), replace=False).astype(int)
 
    # per-label attributions — filter to correctly-classified FIRST, then sample
    attributions = {}
    for label in unique_labels:
        print(f"Computing node explanations for label {label}...")
        pool = test_labels_index.loc[test_labels_index["Label"] == label, "Indices"].values.astype(int)
        pool = pool[y[pool] == label]
        rows = [attribute(i, label) for i in tqdm(sample(pool), desc=f"Label {label}")]
        attributions[label] = pd.DataFrame(rows)
 
    # label-shuffled null: explain each sampled node at a permuted target
    print("Computing label-shuffled null...")
    idx = sample(test_labels_index["Indices"].values.astype(int))
    targets = rng.permutation(y[idx])
    perm_df = pd.DataFrame([attribute(i, t) for i, t in
                            tqdm(zip(idx, targets), total=len(idx), desc="Null")])
 
    if feature_names is not None:
        col_map = dict(enumerate(feature_names))
        perm_df = perm_df.rename(columns=col_map)
        attributions = {k: v.rename(columns=col_map) for k, v in attributions.items()}
 
    return attributions, perm_df



def explain_edges_core(
    model,
    data,
    graph,
    n_explanations: int,
    celltype_key: str,
) -> dict[int, pd.DataFrame]:
    from torch_geometric.explain import Explainer, GNNExplainer

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="log_probs",
        ),
    )

    # Build node-index → cell-type lookup array (avoids per-edge dict lookups)
    nodes = list(graph.nodes())
    node_celltypes = np.array([graph.nodes[n][celltype_key] for n in nodes])

    # Map edge endpoints to cell types in bulk
    ei = data.edge_index.cpu().numpy()
    src_ct = node_celltypes[ei[0]]
    dst_ct = node_celltypes[ei[1]]
    # Canonical ordering: sort the two cell types per edge
    cell_type_pairs = [
        (a, b) if a <= b else (b, a)
        for a, b in zip(src_ct.tolist(), dst_ct.tolist())
    ]

    test_indices = torch.where(data.test_mask)[0]
    test_labels = data.y[test_indices]
    results = {}

    for label_val in test_labels.unique():
        label_val = label_val.item()
        print(f"Edge explanations for label {label_val}...")

        label_idx = test_indices[(test_labels == label_val).nonzero(as_tuple=True)[0]]
        n_sample = min(n_explanations, len(label_idx))
        selected = label_idx[torch.randint(len(label_idx), (n_sample,))]
        perm = test_indices[torch.randperm(len(test_indices))[:n_sample]]

        combined = torch.cat([perm, selected])
        categories = ["permutation"] * len(perm) + ["target"] * len(selected)

        edge_masks = []
        for idx in tqdm(combined, desc=f"Label {label_val}"):
            explanation = explainer(
                x=data.x, edge_index=data.edge_index,
                index=idx.item(), edge_attr=data.edge_attr,
            )
            edge_masks.append(explanation.edge_mask.cpu().detach().numpy())

        importances = {"permutation": [], "target": []}
        for mask, cat in zip(edge_masks, categories):
            pair_imp = {}
            for pair, imp in zip(cell_type_pairs, mask):
                pair_imp.setdefault(pair, []).append(imp)
            importances[cat].append({p: np.mean(v) for p, v in pair_imp.items()})

        all_pairs = set()
        for group in importances.values():
            for expl in group:
                all_pairs.update(expl.keys())

        records = []
        for pair in all_pairs:
            perm_vals = [e.get(pair, 0) for e in importances["permutation"]]
            target_vals = [e.get(pair, 0) for e in importances["target"]]
            _, p = ttest_ind(target_vals, perm_vals, alternative="greater")
            records.append({
                "cell_type_pair": str(pair),
                "mean_importance_target": np.mean(target_vals),
                "mean_importance_permutation": np.mean(perm_vals),
                "p_value": p,
            })
        results[label_val] = pd.DataFrame(records)

    return results

def _feature_names(n_base, augmented, base_names=None):
    base = list(base_names) if base_names is not None else [f"feat_{i}" for i in range(n_base)]
    return base + [f"{b}_nbr" for b in base] if augmented else base

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

def explain_baselines_core(
    fold_datasets: list,
    graph,
    k: int,
    feature_names: list[str] | None = None,
    top_n: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Per-label SHAP feature importance for the RF + Neighbours baseline,
    averaged across folds. Values are signed mean SHAP contributions toward
    each label (features × labels): positive pushes cells toward that label,
    negative away.
    """
    import shap  # lazy: heavy optional dependency, only needed here

    importances = []

    # X, y and spatial coordinates don't depend on the fold split (same
    # adata/graph every time), so the neighbour-augmented feature matrix is
    # built once and reused — only the train/test masks vary per fold.
    X, y, coords, _, _ = _extract(fold_datasets[0], graph)
    X_aug = _augment(X, coords, k)
    names = _feature_names(X.shape[1], True, feature_names)

    for fold_data in fold_datasets:
        train_mask = fold_data.train_mask.cpu().numpy()
        test_mask = fold_data.test_mask.cpu().numpy()

        Xtr, Xte = X_aug[train_mask], X_aug[test_mask]
        ytr, yte = y[train_mask], y[test_mask]

        clf = RandomForestClassifier(1, random_state=42, n_jobs=-1).fit(Xtr, ytr)

        sv = shap.TreeExplainer(clf).shap_values(Xte)
        if isinstance(sv, list):                 
            sv = np.stack(sv, axis=-1)           

        per_label = {}
        for ci, cls in enumerate(clf.classes_):  # class axis follows clf.classes_
            sub = yte == cls
            if sub.sum() < 2:
                per_label[cls] = pd.Series(np.nan, index=names)
                continue
            per_label[cls] = pd.Series(sv[sub, :, ci].mean(0), index=names)

        importances.append(pd.DataFrame(per_label))

    imp_summary = pd.concat(importances).groupby(level=0).mean()  # features × labels

    if verbose:
        print("\n  Top features — RF + Neighbours (signed mean SHAP)")
        for cls in imp_summary.columns:
            order = imp_summary[cls].abs().sort_values(ascending=False).index
            top = imp_summary[cls].reindex(order).head(top_n)
            print(f"\n    Label {cls}")
            print(top.to_string())

    return imp_summary