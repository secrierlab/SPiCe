# Spatial Plasticity in Cellular Environments (SPiCe)

## A computational framework for modelling intrinsic and extrinsic factors driving cell plasticity using spatial transcriptomics

## Author: Eloise Withnell

We propose an integrative modelling approach using graph neural networks (GNNs) and geostatistical regression models to disentangle the influence of intrinsic (e.g. genomic alterations) and extrinsic (e.g. tumor microenvironment, TME) factors in driving cell plasticity, with a focus on epithelial-mesenchymal plasticity (EMP).

## Project Overview
The pipeline combines:
- `emt_plasticity_analysis/` – GNN-based modelling of EMT states using Xenium data. Use analysis.ipynb to run scripts.  
- `figure_plotting/` – Scripts for generating figures used in the manuscript  
- `merfish_analysis/` – Benchmarking the GNN prediction pipeline on MERFISH mouse brain data. Use analysis.ipynb to run scripts.  
- `source_data/` – Processed Xenium data with cell type and EMT annotations  
- `spatial_regression/` – Code for spatial regression analyses (SEM, GWR, MGWR)  
- `LICENSE`  
- `README.md`  

These methods are applied to:
-Breast cancer Xenium spatial transcriptomics data for EMT modelling
-MERFISH mouse brain data as a benchmark for spatial predictability

## Environment Setup

Two conda environments are provided:

- `base_env_gnn` → for GNN training and evaluation (MERFISH & Xenium)
- `figures_paper_gnn.yml` → for  figures

To create the environments:

```bash
conda env create -f base_env_gnn.yml
conda env create -f figures_paper_gnn.yml


