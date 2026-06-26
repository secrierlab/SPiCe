# SPiCe — Spatial Plasticity in Cellular Environments

A computational framework for modelling intrinsic and extrinsic factors driving cell plasticity using spatial transcriptomics data, graph neural networks (GNNs) and geostatistical regression.

SPiCe formalizes *state predictability* as a quantitative proxy for plasticity: stable cell states are predictable from their spatial neighborhood, while plastic states are not.

**Original implementation:** [Eloise Withnell](https://github.com/elliewith) (scripts).

**Package conversion:** [Cenk Celik](https://github.com/cenk-celik) restructured the codebase into an installable Python package.

## Installation
```bash
git clone https://github.com/secrierlab/SPiCe.git
cd SPiCe && pip install -e .
```

## Quick start

SPiCe follows the [scverse](https://scverse.org) convention: tools live under `spice.tl`, plotting under `spice.pl`. All tool functions modify `adata` in place.

```python
import spice

# 1. Discretise the continuous state score
spice.tl.assign_state_labels(
    adata,
    score_key="EMT_score",
    n_states=4, # or 2 for binary classification
    tumour_mask_key="tumour_cells", # only label tumour cells
)

# 2. Build the spatial k-NN graph (stored in adata.uns['spice'])
spice.tl.build_graph(
    adata,
    spatial_key="spatial",
    n_neighbors=12,
    celltype_key="cell_type",
)

# 3. Cross-validate
spice.tl.cross_validate(
    adata,
    feature_mode="celltype", # "intrinsic" or "combined"
    num_folds=5,
    num_epochs=500,
)

# 4. Evaluate and plot
spice.tl.evaluate(adata)
spice.pl.auc_per_class(adata, state_map={0: "EPI", 1: "H-EPI", 2: "H-MES", 3: "MES"})

# 5. Explain and plot
spice.tl.explain_nodes(adata, n_explanations=50)
spice.pl.node_importance(adata, state_map={0: "EPI", 1: "H-EPI", 2: "H-MES", 3: "MES"})

spice.tl.explain_edges(adata, n_explanations=50)
spice.pl.edge_network(adata, state_map={0: "EPI", 1: "MES"})

# 6. Baseline comparison
spice.tl.run_baseline(adata)
spice.pl.baseline(adata)

# 7. Save adata
spice.tl.sanitize(adata)
adata.write_h5ad(adata, "adata.h5ad")
```

## Storage layout

All results are written to `adata.uns['spice']`:

| Key | Written by | Content |
|---|---|---|
| `graph` | `tl.build_graph` | NetworkX spatial k-NN graph |
| `label_binarizer` | `tl.build_graph` | Fitted LabelBinarizer for cell types |
| `params` | `tl.build_graph` | Dict of run parameters |
| `folds` | `tl.cross_validate` | List of PyG Data objects |
| `cv_results` | `tl.cross_validate` | Predictions, models, fold performances |
| `auc` | `tl.evaluate` | Classification AUC DataFrame |
| `regression` | `tl.evaluate` | Regression R²/correlation DataFrame |
| `node_attributions` | `tl.explain_nodes` | IG attributions per label |
| `node_pvalues` | `tl.explain_nodes` | Mann-Whitney p-values (label × feature) |
| `edge_explanations` | `tl.explain_edges` | Edge importance per label |
| `baseline` | `tl.run_baseline` | Summary DataFrame + per-fold results |

## Feature modes

| Mode | Features | Use case |
|---|---|---|
| `"celltype"` | One-hot cell type | TME influence on state (default) |
| `"intrinsic"` | PCA of gene expression / CNAs | Intrinsic genomic drivers |
| `"combined"` | PCA + cell type | Joint intrinsic + extrinsic modelling |

## Citation

If you use SPiCe in your research, please cite:

> Withnell E, Celik C, Secrier M. Integrative Spatial Modelling of Cellular
> Plasticity using Graph Neural Networks and Geostatistics. *bioRxiv* (2025).
> [https://doi.org/10.1101/2025.09.24.678189](https://doi.org/10.1101/2025.09.24.678189)

## Tutorial

See [tutorial.md](tutorial.md) for a step-by-step guide covering the full pipeline, feature modes, explanations and plotting.
spice_example.ipynb has an example adata file to replicate the analysis. 

## Licence

GPL-3.0. See [LICENSE](LICENSE) for details.
