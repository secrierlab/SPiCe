# SPiCe Tutorial

This tutorial walks through the full SPiCe pipeline — from loading spatial transcriptomics data to interpreting cell-state plasticity with graph neural networks. It assumes you have an `AnnData` object with spatial coordinates and cell type annotations.

## Installation
```bash
git clone https://github.com/secrierlab/SPiCe.git
cd SPiCe && pip install -e .
```

SPiCe requires Python 3.11 or 3.12. Core dependencies include `PyTorch`, `PyTorch Geometric`, `scanpy`, `anndata` and `Captum`.

## API conventions

SPiCe follows the `scverse` convention. Analysis functions live under `spice.tl` and plotting functions under `spice.pl`. All `tl` functions modify the AnnData object in place and return `None` by default. Pass `copy=True` to operate on (and return) a copy instead.

Results are stored in `adata.uns['spice']` — a dictionary that grows as you progress through the pipeline.

## Preparing your data

SPiCe expects an AnnData object with:

- **Spatial coordinates** in `adata.obsm['spatial']` (an n × 2 array), or alternatively as `adata.obs` columns named `array_row`/`array_col` or `x`/`y`.
- **Cell-type annotations** in an `adata.obs` column (default: `"cell_type"`).
- **A continuous state score** in `adata.obs` (e.g. `"EMT_score"`), which SPiCe will discretize into categorical labels.
- Optionally, **PCA coordinates** in `adata.obsm['X_pca']` if you plan to use intrinsic or combined feature modes.

A typical starting point:

```python
import scanpy as sc
import spice

adata = sc.read_h5ad("adata.h5ad")
```

## Step 1: Assign state labels

The first step discretizes a continuous score (e.g. an EMT hallmark score) into categorical labels suitable for classification. You can use spatially aware enrichment tool, EnrichMap[https://enrichmap.readthedocs.io/en/stable/]

```python
spice.tl.assign_state_labels(
    adata,
    score_key="EMT_score", # column in adata.obs with the continuous score
    n_states=4, # 2 (median split) or 4 (quartile split)
    tumour_mask_key="tumour_cells", # only label tumour cells; None labels all cells
    label_key="state_label", # destination column in adata.obs
)
```

Cells where `tumour_mask_key` is NaN receive a label of `-1` and are excluded from training and evaluation. With `n_states=4`, the score is split at the 25th, 50th and 75th percentiles, producing four ordinal classes (e.g. epithelial, hybrid-epithelial, hybrid-mesenchymal, mesenchymal). With `n_states=2`, a simple median split is used.

After this step, `adata.obs["state_label"]` contains integer labels.

## Step 2: Build the spatial graph

SPiCe constructs a k-nearest-neighbor graph from spatial coordinates. Each cell becomes a node; edges connect spatially proximate cells, weighted by inverse Euclidean distance.

```python
spice.tl.build_graph(
    adata,
    spatial_key="spatial", # key in adata.obsm
    n_neighbors=12, # number of nearest neighbors
    celltype_key="celltype_minor", # cell-type column in adata.obs
    label_key="state_label", # labels from step 1
    score_key="EMT_score", # continuous score (stored as node attribute)
    n_blocks=4, # grid divisions for spatial block assignment
)
```

This writes three entries to `adata.uns['spice']`:

- `graph` — a NetworkX graph with node attributes (cell type, label, coordinates, etc.) and weighted edges.
- `label_binarizer` — a fitted `LabelBinarizer` for cell types.
- `params` — a dictionary of key parameters used by downstream functions.

Cells with NaN or Inf spatial coordinates are dropped in place before graph construction. The graph is also partitioned into spatial blocks (a grid of `n_blocks × n_blocks`) used for inductive cross-validation splits.

## Step 3: Cross-validate

Cross-validation trains a GCN (graph convolutional network) model across k folds and stores predictions and model objects.

```python
spice.tl.cross_validate(
    adata,
    feature_mode="celltype", # "celltype", "intrinsic" or "combined"
    num_folds=5,
    num_epochs=500,
    hidden_dim1=16,
    hidden_dim2=32,
    dropout=0.5,
    learning_rate=0.01,
    class_weights=True, # inverse-frequency weighting
    inductive_split=True, # spatial block split (True) or random node split (False)
    return_last=True, # use last epoch's predictions
    verbose=True,
)
```

### Feature modes

The `feature_mode` parameter controls what node features the GNN receives:

| Mode | Features | Use case |
|---|---|---|
| `"celltype"` | One-hot encoded cell type | TME (extrinsic) influence on state |
| `"intrinsic"` | PCA of gene expression or CNAs | Intrinsic genomic drivers |
| `"combined"` | PCA + one-hot cell type | Joint intrinsic and extrinsic modelling |

For `"intrinsic"` or `"combined"`, you need PCA coordinates in `adata.obsm['X_pca']`. Run `sc.pp.pca(adata)` beforehand if needed. You can control the number of components with `n_pcs` (default: 7) and the PCA key with `pca_key`.

### Splitting strategy

When `inductive_split=True` (default), folds are defined by spatial blocks or sample identifiers — the model never sees test-region cells during training. When `False`, a random node-level split is used (transductive setting).

### What gets stored

After cross-validation, `adata.uns['spice']` gains:

- `folds` — a list of PyG `Data` objects, one per fold.
- `cv_results` — a dictionary with keys `all_predicted` (list of prediction arrays), `all_true` (list of ground-truth arrays), `models` (list of trained `SpatialGCN` instances) and `fold_performances` (list of per-fold AUC or MSE values).
- `feature_mode` — the feature mode used.
- `continuous_y` — whether regression was used.

## Step 4: Evaluate

Evaluation computes aggregate metrics from the cross-validation results.

```python
spice.tl.evaluate(adata)
```

For classification, this writes a DataFrame to `adata.uns['spice']['auc']` containing per-fold, per-class AUC values as well as micro- and macro-averaged AUC. For regression (`continuous_y=True` in `cross_validate`), it writes R² and Pearson correlation to `adata.uns['spice']['regression']`.

### Plotting: AUC per class

```python
spice.pl.auc_per_class(
    adata,
    state_map={0: "EPI", 1: "H-EPI", 2: "H-MES", 3: "MES"}
)
```

This produces a strip plot of AUC values across folds for each class, with horizontal lines at the mean. The `state_map` dictionary translates integer labels to readable names.

For regression, use:

```python
spice.pl.regression_r2(adata, save="r2_folds.pdf")
```

## Step 5: Explain — node-level feature importance

SPiCe uses Integrated Gradients (via Captum) to quantify how much each node feature contributes to the model's predictions. A permutation baseline establishes statistical significance.

```python
spice.tl.explain_nodes(
    adata,
    n_explanations=50, # number of test nodes to explain per label
    fold_index=0, # which fold's model and data to use (0-based)
)
```

This writes:

- `node_attributions` — a dictionary mapping each label to a DataFrame of IG attribution scores.
- `node_pvalues` — a DataFrame of Mann-Whitney U p-values (label × feature), testing whether each feature's attribution is significantly greater than the permutation baseline.

### Plotting: node importance

```python
spice.pl.node_importance(
    adata,
    state_map={0: "EPI", 1: "H-EPI", 2: "H-MES", 3: "MES"},
    alpha=0.05
)
```

This produces scatter plots of −log₁₀(p) per feature. Significant features (p < alpha) appear above the dashed threshold line and are highlighted in red.

## Step 6: Explain — edge-level interactions

GNNExplainer identifies which cell-type interactions (edges) are most important for predicting each state.

```python
spice.tl.explain_edges(
    adata,
    n_explanations=50,
    fold_index=0
)
```

This writes `edge_explanations` — a dictionary mapping each label to a DataFrame with columns `cell_type_pair`, `mean_importance_target`, `mean_importance_permutation` and `p_value`.

### Plotting: edge interaction network

```python
spice.pl.edge_network(
    adata,
    state_map={0: "EPI", 1: "H-EPI", 2: "H-MES", 3: "MES"},
    alpha=0.05
)
```

This draws circular network diagrams showing significant cell-type interactions, with edge width proportional to importance.

To plot a single state:

```python
fig, ax = spice.pl.edge_network(adata, state=0, state_map={0: "EPI"})
```

## Step 7: Baseline comparison

SPiCe can compare the GNN against Random Forest and MLP baselines, with and without neighbor-aggregated features.

```python
spice.tl.run_baseline(adata, k=10, verbose=True)
```

This writes a tuple `(summary_df, results_dict)` to `adata.uns['spice']['baseline']`. The summary DataFrame has columns `model`, `mean_AUC` and `std`.

### Plotting: baseline comparison

```python
spice.pl.baseline(adata, save="baseline_comparison.pdf")
```

This produces a horizontal bar chart of mean AUC with error bars for each model.

## Additional features

### Preparing features separately

If you want to inspect or reuse the feature matrix without running cross-validation, you can call `prepare_features` directly:

```python
spice.tl.prepare_features(adata, mode="combined", n_pcs=7)
# features are now in adata.obsm['X_spice']
```

### Comparing multiple runs

To compare feature modes or splitting strategies side by side, collect AUC DataFrames from separate runs and use `compare_runs`:

```python
# Run with different feature modes and store AUC DataFrames
auc_dfs = {
    "TME": adata_tme.uns['spice']['auc'],
    "Intrinsic": adata_intr.uns['spice']['auc'],
    "Combined": adata_comb.uns['spice']['auc'],
}
spice.pl.compare_runs(auc_dfs, state_map={0: "EPI", 1: "MES"})
```

### MERFISH-style per-cell-type analysis

For benchmarks where each cell type is trained independently (as in the MERFISH analysis), use the standalone dot plot:

```python
spice.pl.auc_dotplot(df_auc, celltype_col="Cell_Type", auc_col="AUC")
```

### Regression mode

To predict a continuous score rather than discrete classes, set `continuous_y=True` in `cross_validate`:

```python
spice.tl.cross_validate(adata, continuous_y=True, num_epochs=500)
spice.tl.evaluate(adata)
spice.pl.regression_r2(adata)
```

### Device selection

By default SPiCe uses CUDA if available, falling back to CPU. You can override this:

```python
spice.tl.cross_validate(adata, device="cpu")
```

## Saving results

Several objects SPiCe stores in `adata.uns['spice']` (NetworkX graphs, PyTorch models, PyG Data objects) are not HDF5-serializable and will cause `adata.write_h5ad()` to fail. Call `sanitize` before saving:

```python
spice.tl.sanitize(adata)
adata.write_h5ad("adata.h5ad")
```

This removes the non-serializable objects (graph, models, folds, label binarizer) while keeping all scientific results: AUC tables, p-value DataFrames, attribution scores, baseline summaries and run parameters. Cell-type class names from the binarizer are preserved in `params['celltype_classes']`.

## Storage layout reference

All results live in `adata.uns['spice']`:

| Key | Written by | Content |
|---|---|---|
| `graph` | `tl.build_graph` | NetworkX spatial k-NN graph |
| `label_binarizer` | `tl.build_graph` | Fitted LabelBinarizer for cell types |
| `params` | `tl.build_graph` | Dict of run parameters |
| `folds` | `tl.cross_validate` | List of PyG Data objects |
| `cv_results` | `tl.cross_validate` | Predictions, models, fold performances |
| `feature_mode` | `tl.cross_validate` / `tl.prepare_features` | Feature strategy used |
| `continuous_y` | `tl.cross_validate` | Whether regression was used |
| `auc` | `tl.evaluate` | Classification AUC DataFrame |
| `regression` | `tl.evaluate` | Regression R²/correlation DataFrame |
| `node_attributions` | `tl.explain_nodes` | IG attributions per label |
| `node_pvalues` | `tl.explain_nodes` | Mann-Whitney p-values (label × feature) |
| `edge_explanations` | `tl.explain_edges` | Edge importance per label |
| `baseline` | `tl.run_baseline` | Tuple of (summary DataFrame, per-fold results) |

After `tl.sanitize`, the graph, label binarizer, folds and models are removed; dict-of-DataFrame entries are flattened; and the baseline tuple is reduced to just the summary DataFrame.

## Full example

```python
import scanpy as sc
import spice

# Load data
adata = sc.read_h5ad("xenium_breast_cancer.h5ad")

# 1. Discretise EMT score
spice.tl.assign_state_labels(
    adata, score_key="EMT_hallmarks", n_states=4, tumour_mask_key="tumour_cells"
)

# 2. Build spatial graph
spice.tl.build_graph(
    adata, spatial_key="spatial", n_neighbors=12, celltype_key="cell_type"
)

# 3. Cross-validate
spice.tl.cross_validate(
    adata, feature_mode="celltype", num_folds=5, num_epochs=500
)

# 4. Evaluate
spice.tl.evaluate(adata)

state_map = {0: "EPI", 1: "H-EPI", 2: "H-MES", 3: "MES"}
spice.pl.auc_per_class(adata, state_map=state_map)

# 5. Node explanations
spice.tl.explain_nodes(adata, n_explanations=50)
spice.pl.node_importance(adata, state_map=state_map)

# 6. Edge explanations
spice.tl.explain_edges(adata, n_explanations=50)
spice.pl.edge_network(adata, state_map=state_map)

# 7. Baseline comparison
spice.tl.run_baseline(adata)
spice.pl.baseline(adata)
```

## Citation

If you use SPiCe in your research, please cite:

> Withnell E, Celik C, Secrier M. Integrative Spatial Modelling of Cellular
> Plasticity using Graph Neural Networks and Geostatistics. *bioRxiv* (2025).
> [https://doi.org/10.1101/2025.09.24.678189](https://doi.org/10.1101/2025.09.24.678189)
