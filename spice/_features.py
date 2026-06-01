"""Internal node feature matrix construction."""

from __future__ import annotations

from typing import Literal

import numpy as np
from anndata import AnnData
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler


def prepare_features(
    adata: AnnData,
    mode: Literal["celltype", "intrinsic", "combined"] = "celltype",
    celltype_key: str = "celltype_minor",
    pca_key: str = "X_pca",
    n_pcs: int = 7,
    scale_pca: bool = True,
) -> np.ndarray:
    """Build the node feature matrix from an AnnData object."""
    features = []

    if mode in ("intrinsic", "combined"):
        if pca_key not in adata.obsm:
            raise KeyError(
                f"PCA key '{pca_key}' not found in adata.obsm. "
                "Run sc.pp.pca(adata) first or pass the correct key."
            )
        pca = adata.obsm[pca_key][:, :n_pcs].copy()
        if scale_pca:
            pca = MinMaxScaler().fit_transform(pca)
        pca = np.nan_to_num(pca, nan=0.0)
        features.append(pca)

    if mode in ("celltype", "combined"):
        lb = LabelBinarizer()
        features.append(lb.fit_transform(adata.obs[celltype_key].values))

    return np.hstack(features).astype(np.float32)
