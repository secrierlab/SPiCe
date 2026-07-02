"""PyTorch Geometric dataset construction from AnnData and NetworkX graphs."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from anndata import AnnData
from sklearn.model_selection import KFold
from torch_geometric.data import Data, InMemoryDataset

from spice._graph import create_edge_index
from spice._features import prepare_features


def _build_shared_data(
    adata: AnnData,
    graph,
    *,
    feature_mode: str,
    celltype_key: str,
    label_key: str,
    score_key: str,
    pca_key: str,
    n_pcs: int,
    continuous_y: bool,
    edge_weights: bool,
) -> dict:
    """Build the parts of the PyG ``Data`` object that do not depend on the
    cross-validation fold split.

    Node features, labels, the edge index/weights, cell IDs and sample
    assignments are a pure function of *adata* and *graph* — they are the
    same for every fold. :class:`SpatialGNNDataset` used to recompute all of
    this from scratch for each of the ``num_folds`` folds (identical work
    repeated ``num_folds`` times). Computing it once here and passing the
    result to every fold via the ``_shared`` constructor argument removes
    that redundancy without changing any values: only the train/test masks
    (which do depend on the fold) are still computed per fold.
    """
    x = prepare_features(
        adata, mode=feature_mode, celltype_key=celltype_key,
        pca_key=pca_key, n_pcs=n_pcs,
    )
    data_x = torch.from_numpy(x)  # x is already float32

    nodes = list(graph.nodes())
    node_data = graph.nodes

    if continuous_y:
        y_raw = np.array(
            [node_data[n].get(score_key, np.nan) for n in nodes],
            dtype=np.float64,
        )
        np.round(y_raw, 5, out=y_raw)
        y_raw[np.isnan(y_raw)] = -1
        data_y = torch.from_numpy(y_raw).float()
    else:
        y_raw = np.array(
            [node_data[n].get("labels", -1) for n in nodes],
            dtype=np.float64,
        )
        y_raw[np.isnan(y_raw)] = -1
        data_y = torch.from_numpy(y_raw.astype(np.int64))

    edge_index, edge_attr = create_edge_index(graph, edge_lengths=edge_weights)

    cell_id = np.array([node_data[n].get("cell_id", n) for n in nodes])
    samples = np.array([node_data[n].get("sample", np.nan) for n in nodes])

    return {
        "data_x": data_x,
        "data_y": data_y,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "cell_id": cell_id,
        "samples": samples,
        "num_nodes": graph.number_of_nodes(),
    }


class SpatialGNNDataset(InMemoryDataset):
    """PyG dataset for GNN-based spatial cell-state prediction.

    Wraps an AnnData object and its corresponding spatial graph into a
    single :class:`~torch_geometric.data.Data` object suitable for node
    classification (discrete labels) or regression (continuous scores).

    Parameters
    ----------
    adata
        Annotated data matrix.
    graph
        Spatial k-NN graph built by :func:`~spice.graph.build_graph`.
    feature_mode
        Feature strategy passed to :func:`~spice.preprocessing.prepare_features`.
    celltype_key
        Column in ``adata.obs`` with cell type annotations.
    label_key
        Column in ``adata.obs`` holding discrete state labels.
    score_key
        Column in ``adata.obs`` holding continuous state scores (used when
        ``continuous_y=True``).
    sample_key
        Node attribute in *graph* identifying biological samples/FOVs.
    pca_key
        Key in ``adata.obsm`` for PCA features.
    n_pcs
        Number of PCA components to use.
    continuous_y
        If ``True``, treat the task as regression rather than classification.
    inductive_split
        If ``True``, split by sample/block (inductive). Otherwise, use
        random node-level splitting (transductive).
    fold_idx
        1-based fold index for cross-validation.
    num_folds
        Total number of CV folds.
    edge_weights
        If ``True``, include edge weights from the graph.
    _shared
        Internal use only. Precomputed fold-invariant tensors (from
        :func:`_build_shared_data`), used by :func:`spice.tl.cross_validate`
        to avoid rebuilding features/edges/labels for every fold. Leave as
        ``None`` for normal standalone use — it is computed automatically.
    """

    def __init__(
        self,
        adata: AnnData,
        graph,
        *,
        feature_mode: Literal["celltype", "intrinsic", "combined"] = "celltype",
        celltype_key: str = "celltype_minor",
        label_key: str = "state_label",
        score_key: str = "EMT_hallmarks",
        sample_key: str = "sample",
        pca_key: str = "X_pca",
        n_pcs: int = 7,
        continuous_y: bool = False,
        inductive_split: bool = True,
        fold_idx: int = 1,
        num_folds: int = 5,
        edge_weights: bool = True,
        _shared: dict | None = None,
    ):
        super().__init__(".", None, None, None)
        self.adata = adata
        self.graph = graph
        self.feature_mode = feature_mode
        self.celltype_key = celltype_key
        self.label_key = label_key
        self.score_key = score_key
        self.sample_key = sample_key
        self.pca_key = pca_key
        self.n_pcs = n_pcs
        self.continuous_y = continuous_y
        self.inductive_split = inductive_split
        self.fold_idx = fold_idx
        self.num_folds = num_folds
        self.edge_weights = edge_weights
        self._shared = _shared
        self.batch_index_to_test = None
        self._process_data()

    # ------------------------------------------------------------------
    # Required InMemoryDataset stubs
    # ------------------------------------------------------------------
    def _download(self):
        return

    def _process(self):
        return

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _process_data(self):
        if self._shared is not None:
            # Reuse precomputed fold-invariant tensors (see
            # _build_shared_data): features, labels, edges, cell IDs and
            # sample assignments never depend on the fold split, so
            # spice.tl.cross_validate builds them once and shares them
            # across folds. Clone so each fold owns independent tensors
            # (no aliasing across Data objects), which is cheap relative to
            # rebuilding features/edges from scratch.
            shared = self._shared
            data_x = shared["data_x"].clone()
            data_y = shared["data_y"].clone()
            edge_index = shared["edge_index"].clone()
            edge_attr = shared["edge_attr"].clone() if shared["edge_attr"] is not None else None
            cell_id = shared["cell_id"].copy()
            samples = shared["samples"].copy()
            num_nodes = shared["num_nodes"]
        else:
            # Node features
            x = prepare_features(
                self.adata,
                mode=self.feature_mode,
                celltype_key=self.celltype_key,
                pca_key=self.pca_key,
                n_pcs=self.n_pcs,
            )
            data_x = torch.from_numpy(x)  # x is already float32

            # Extract all node attributes once to avoid repeated dict lookups
            nodes = list(self.graph.nodes())
            node_data = self.graph.nodes

            # Labels
            if self.continuous_y:
                y_raw = np.array(
                    [node_data[n].get(self.score_key, np.nan) for n in nodes],
                    dtype=np.float64,
                )
                np.round(y_raw, 5, out=y_raw)
                y_raw[np.isnan(y_raw)] = -1
                data_y = torch.from_numpy(y_raw).float()
            else:
                y_raw = np.array(
                    [node_data[n].get("labels", -1) for n in nodes],
                    dtype=np.float64,
                )
                y_raw[np.isnan(y_raw)] = -1
                data_y = torch.from_numpy(y_raw.astype(np.int64))

            # Edges
            edge_index, edge_attr = create_edge_index(self.graph, edge_lengths=self.edge_weights)

            # Cell IDs and sample info
            cell_id = np.array([node_data[n].get("cell_id", n) for n in nodes])
            samples = np.array([node_data[n].get("sample", np.nan) for n in nodes])
            num_nodes = self.graph.number_of_nodes()

        # Assemble Data object
        data = Data(x=data_x, y=data_y, edge_index=edge_index)
        if edge_attr is not None:
            data.edge_attr = edge_attr
        data.num_nodes = num_nodes
        data.continuous_score_bool = self.continuous_y

        # Cell IDs and sample info
        data.cell_id = cell_id
        data.samples = samples

        # Cross-validation masks
        self.batch_index_to_test = self._get_test_samples(samples)
        train_mask, test_mask = self._compute_masks(samples)

        # Exclude unlabeled / NaN-feature cells
        if self.continuous_y:
            nan_mask = torch.isnan(data_x).any(dim=1) | (data_y == -1)
        else:
            nan_mask = data_y == -1

        data.train_mask = train_mask & ~nan_mask
        data.test_mask = test_mask & ~nan_mask

        # Number of classes
        if self.continuous_y:
            data.num_classes = 1
        else:
            data.num_classes = int(data_y[data.train_mask].unique().shape[0])

        self.data, self.slices = self.collate([data])

    def _get_test_samples(self, samples: np.ndarray) -> np.ndarray | None:
        """Determine which samples fall into the test fold."""
        if not self.inductive_split:
            return None

        unique_samples = np.unique(samples)
        unique_samples = unique_samples[~np.isnan(unique_samples.astype(float))]
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        folds = list(kf.split(unique_samples))
        _, test_idx = folds[self.fold_idx - 1]
        return unique_samples[test_idx]

    def _compute_masks(
        self, samples: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute train/test boolean masks."""
        n = self.graph.number_of_nodes()

        if self.inductive_split and self.batch_index_to_test is not None:
            test_mask = torch.from_numpy(np.isin(samples, self.batch_index_to_test))
            train_mask = ~test_mask
        else:
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
            folds = list(kf.split(np.arange(n)))
            train_idx, test_idx = folds[self.fold_idx - 1]
            train_mask = torch.zeros(n, dtype=torch.bool)
            test_mask = torch.zeros(n, dtype=torch.bool)
            train_mask[train_idx] = True
            test_mask[test_idx] = True

        return train_mask, test_mask
