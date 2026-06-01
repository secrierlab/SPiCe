"""
SPiCe — Spatial Plasticity in Cellular Environments.

A framework for modelling intrinsic and extrinsic factors driving cell
plasticity using spatial transcriptomics data and graph neural networks.

Follows the scanpy-style ``tl`` / ``pl`` convention::

    import spice

    spice.tl.assign_state_labels(adata, score_key="EMT_hallmarks", n_states=4)
    spice.tl.build_graph(adata, celltype_key="celltype_minor")
    spice.tl.cross_validate(adata, num_epochs=500)
    spice.tl.evaluate(adata)
    spice.tl.explain_nodes(adata, n_explanations=100)

    spice.pl.auc_per_class(adata, state_map={0: "EPI", 1: "MES"})
    spice.pl.node_importance(adata)

Reference
---------
Withnell E, Celik C, Secrier M. Integrative Spatial Modelling of Cellular
Plasticity using Graph Neural Networks and Geostatistics. bioRxiv (2025).
https://doi.org/10.1101/2025.09.24.678189
"""

from spice import tl, pl  # noqa: F401
from spice.models import SpatialGCN  # noqa: F401
from spice.dataset import SpatialGNNDataset  # noqa: F401

__version__ = "0.1.0"

__all__ = ["tl", "pl", "SpatialGCN", "SpatialGNNDataset"]
