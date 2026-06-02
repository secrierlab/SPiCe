"""Internal explanation routines (Integrated Gradients, GNNExplainer)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, ttest_ind
from tqdm import tqdm


def explain_nodes_core(
    model,
    data,
    test_labels_index: pd.DataFrame,
    n_explanations: int,
    mask_type: str,
    feature_names: list[str] | None,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    from captum.attr import IntegratedGradients
    from torch_geometric.nn import to_captum_input, to_captum_model

    unique_labels = test_labels_index["Label"].unique()
    attributions = {}

    # Pre-compute shared captum inputs (edge_index etc. are the same for
    # every target node — only the index argument changes).
    inputs, fwd_args = to_captum_input(
        data.x, data.edge_index, mask_type, data.edge_attr,
    )

    for label in unique_labels:
        print(f"Computing node explanations for label {label}...")
        indices = test_labels_index.loc[
            test_labels_index["Label"] == label, "Indices"
        ].values
        sampled = np.random.choice(indices, min(n_explanations, len(indices)), replace=False)

        rows = []
        for idx in tqdm(sampled, desc=f"Label {label}"):
            idx = int(idx)
            if int(data.y[idx]) != label:
                continue
            captum_model = to_captum_model(model, mask_type, idx)
            ig = IntegratedGradients(captum_model)
            attr = ig.attribute(
                inputs=inputs, target=int(data.y[idx]),
                additional_forward_args=fwd_args, internal_batch_size=1,
            )
            attr_np = attr[0].squeeze().detach().cpu().numpy()
            rows.append(pd.DataFrame(attr_np).mean(axis=0).values)
        attributions[label] = pd.DataFrame(rows)

    # permutation baseline
    print("Computing permutation baseline...")
    all_idx = test_labels_index["Indices"].values
    perm_sampled = np.random.choice(all_idx, min(n_explanations, len(all_idx)), replace=False)
    perm_rows = []
    for idx in tqdm(perm_sampled, desc="Permutation"):
        idx = int(idx)
        captum_model = to_captum_model(model, mask_type, idx)
        ig = IntegratedGradients(captum_model)
        attr = ig.attribute(
            inputs=inputs, target=int(data.y[idx]),
            additional_forward_args=fwd_args, internal_batch_size=1,
        )
        attr_np = attr[0].squeeze().detach().cpu().numpy()
        perm_rows.append(pd.DataFrame(attr_np).mean(axis=0).values)
    perm_df = pd.DataFrame(perm_rows)

    if feature_names is not None:
        col_map = {i: name for i, name in enumerate(feature_names)}
        perm_df = perm_df.rename(columns=col_map)
        for label in attributions:
            attributions[label] = attributions[label].rename(columns=col_map)

    ref_cols = attributions[unique_labels[0]].columns
    p_values = pd.DataFrame(index=unique_labels, columns=ref_cols, dtype=float)
    for label in unique_labels:
        for feat in ref_cols:
            _, p = mannwhitneyu(
                attributions[label][feat], perm_df[feat], alternative="greater",
            )
            p_values.loc[label, feat] = p

    return attributions, p_values


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
