"""Internal graph construction utilities."""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import torch
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelBinarizer


def build_graph_core(
    adata: AnnData,
    spatial_key: str,
    n_neighbors: int,
    celltype_key: str,
    label_key: str,
    score_key: str | None,
    sample_key: str | None,
    extra_obs_keys: list[str],
) -> tuple[nx.Graph, LabelBinarizer]:
    """Build a spatial k-NN graph.  See :func:`spice.tl.build_graph`."""
    coords = get_spatial_coords(adata, spatial_key)

    # One-hot encode cell types
    lb = LabelBinarizer()
    one_hot = lb.fit_transform(adata.obs[celltype_key].values)

    # kNN
    # NOTE: match the original implementation — NearestNeighbors with
    # n_neighbors=k returns k results including self, and the loop
    # range(1, k) yields k-1 actual neighbors per node.
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    G = nx.Graph()
    epsilon = 1e-6

    for i in range(adata.n_obs):
        attrs = {
            "cell_id": adata.obs_names[i],
            "array_row": float(coords[i, 0]),
            "array_col": float(coords[i, 1]),
            celltype_key: adata.obs[celltype_key].values[i],
            "one_hot_Cell_Type": one_hot[i],
            "labels": int(adata.obs[label_key].values[i]) if label_key in adata.obs else -1,
        }
        if score_key and score_key in adata.obs.columns:
            val = adata.obs[score_key].values[i]
            attrs[score_key] = float(val) if not np.isnan(val) else np.nan
        if sample_key and sample_key in adata.obs.columns:
            attrs["sample"] = adata.obs[sample_key].values[i]
        for key in extra_obs_keys:
            if key in adata.obs.columns:
                attrs[key] = adata.obs[key].values[i]
        G.add_node(i, **attrs)

    for i in range(adata.n_obs):
        for j in range(1, n_neighbors):  # skip self (index 0)
            neighbor_idx = indices[i, j]
            dist = distances[i, j]
            G.add_edge(i, neighbor_idx, weight=1.0 / (dist + epsilon))

    return G, lb


def assign_spatial_blocks_core(G: nx.Graph, n_blocks: int = 4) -> None:
    """Assign spatial block indices to labeled graph nodes."""
    labeled = [n for n in G.nodes() if G.nodes[n].get("labels", -1) != -1]
    coords = np.array([[G.nodes[n]["array_row"], G.nodes[n]["array_col"]] for n in labeled])

    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    bins = [np.linspace(mn, mx, n_blocks + 1) for mn, mx in zip(min_vals, max_vals)]

    row_idx = np.clip(np.digitize(coords[:, 0], bins[0]) - 1, 0, n_blocks - 1)
    col_idx = np.clip(np.digitize(coords[:, 1], bins[1]) - 1, 0, n_blocks - 1)
    blocks = row_idx * n_blocks + col_idx

    for idx, node in enumerate(labeled):
        G.nodes[node]["sample"] = int(blocks[idx])
    for node in G.nodes():
        if "sample" not in G.nodes[node]:
            G.nodes[node]["sample"] = np.nan


def create_edge_index(
    G: nx.Graph,
    edge_lengths: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Convert a NetworkX graph to PyG-style edge index + optional weights."""
    adj = nx.to_scipy_sparse_array(G, nodelist=list(G.nodes())).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64))
    col = torch.from_numpy(adj.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
    if edge_lengths:
        return edge_index, torch.from_numpy(adj.data).to(torch.float)
    return edge_index, None


def filter_nan_coords(adata: AnnData, spatial_key: str) -> int:
    """Drop cells with NaN/Inf spatial coordinates in place.

    Returns the number of cells removed.
    """
    coords = get_spatial_coords(adata, spatial_key)
    valid = np.isfinite(coords).all(axis=1)
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        warnings.warn(
            f"Dropped {n_dropped} cells with NaN/Inf spatial coordinates "
            f"({n_dropped / len(valid) * 100:.1f}% of total)."
        )
        adata._inplace_subset_obs(valid)
    return n_dropped


def get_spatial_coords(adata: AnnData, spatial_key: str = "spatial") -> np.ndarray:
    """Extract 2-D spatial coordinates from an AnnData object."""
    if spatial_key in adata.obsm:
        return np.asarray(adata.obsm[spatial_key])[:, :2]
    for row_key, x_key in [("array_row", "x")]:
        if row_key in adata.obs.columns:
            return adata.obs[["array_row", "array_col"]].values.astype(float)
        if x_key in adata.obs.columns:
            return adata.obs[["x", "y"]].values.astype(float)
    raise KeyError(
        f"Could not find spatial coordinates. Provide adata.obsm['{spatial_key}'] "
        "or adata.obs columns 'array_row'/'array_col' (or 'x'/'y')."
    )
