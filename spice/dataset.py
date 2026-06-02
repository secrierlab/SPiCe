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

        # Assemble Data object
        data = Data(x=data_x, y=data_y, edge_index=edge_index)
        if edge_attr is not None:
            data.edge_attr = edge_attr
        data.num_nodes = self.graph.number_of_nodes()
        data.continuous_score_bool = self.continuous_y

        # Cell IDs and sample info
        data.cell_id = np.array([node_data[n].get("cell_id", n) for n in nodes])
        samples = np.array([node_data[n].get("sample", np.nan) for n in nodes])
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
