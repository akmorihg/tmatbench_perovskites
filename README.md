# PerovskiteScout: ML for Perovskite Formation Energy Prediction

This repository contains Team 3 project materials for predicting perovskite formation energy with machine learning as a faster proxy for expensive DFT calculations.

## Repository Contents

- `Team3, Project Proposal-4.pdf`: project proposal (motivation, plan, target metrics).
- `Team3_P1-2.pdf`: presentation slides (business framing, data, feature strategy, intermediate results).
- `ML_baselines_exploration.ipynb`: broad baseline exploration with composition features and multiple regressors.
- `RF_baseline_exploration.ipynb`: Random Forest baseline using composition + structure-derived features with Matbench folds.
- `GNN.ipynb`: custom graph neural network (MPNN-style) pipeline with 5-fold CV.

## Project Goal

Predict perovskite formation energy (`e_form`) to support rapid candidate screening in materials discovery.

From the proposal/slides:

- Problem: combinatorial perovskite search space + high DFT cost.
- Objective: build practical baselines and improve toward stronger non-GNN/GNN performance.
- Evaluation focus: MAE (primary), with RMSE and R2 where applicable.

## Data

Primary benchmark task:

- `matbench_perovskites` (~18,928 samples)
- Inputs: crystal structure (`pymatgen.Structure`), composition-derived features
- Target: `e_form` (formation energy; project files label it as eV/atom in some places and eV/cell in others)

Additional dataset mentioned in the proposal:

- `matbench_mp_e_form` (~132,752 samples) for transfer-learning motivation

## Implemented Modeling Work

### 1) `ML_baselines_exploration.ipynb`

- Loads `matbench_perovskites` via `matbench`.
- Builds composition features with Matminer Magpie descriptors.
- Cleans features (numeric-only selection, NA handling, low-variance and high-correlation pruning).
- Tries multiple models/pipelines:
  - Random Forest
  - Ridge/Lasso/Decision Tree + PCA
  - XGBoost + PCA + Polynomial features
  - SVR + PCA + Polynomial features
  - KNN + PCA + Polynomial features
  - Random kitchen sinks (RBFSampler) + Ridge / ElasticNet

Example notebook outputs:

- Default RF: MAE `0.5908`, RMSE `0.7888`
- SVM pipeline: MAE `0.4328`, R2 `0.3837`
- XGBoost pipeline: MAE `0.5371`

### 2) `RF_baseline_exploration.ipynb`

- Uses official Matbench folds from `matbench_perovskites`.
- Feature pipeline:
  - `StructureComposition(ElementProperty.from_preset("magpie"))`
  - `SineCoulombMatrix(flatten=True)`
- Fold-wise train-only feature filtering:
  - drop all-zero / constant / near-constant features
  - numeric alignment + mean imputation
- Model: `RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)`

Captured notebook fold outputs:

- Fold 0: MAE `0.2358`, RMSE `0.3300`, R2 `0.7990`
- Fold 1: MAE `0.2364`, RMSE `0.3388`, R2 `0.8021`
- Fold 2: MAE `0.2372`, RMSE `0.3381`, R2 `0.7934`

Slides also report:

- RF (structure + composition): about `0.23` MAE
- XGBoost (same features): about `0.22` MAE

### 3) `GNN.ipynb`

- Implements a custom MPNN-like architecture with:
  - atom embeddings
  - edge radial-basis expansion (128 bins)
  - message passing with dropout + batch norm
  - residual updates
  - mean graph pooling
- Training:
  - 5-fold CV (`KFold`)
  - AdamW optimizer
  - LR scheduler (`ReduceLROnPlateau`)
  - early stopping (patience 20)

Final notebook output:

- Mean test MAE across 5 folds: **`0.0541 +/- 0.0030`** (logged as eV/cell)

## Quickstart

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas scikit-learn matplotlib tqdm pymatgen matminer xgboost torch jupyter
pip install matbench --no-deps
```

### 2) Run notebooks

```bash
jupyter lab
```

Recommended order:

1. `ML_baselines_exploration.ipynb`
2. `RF_baseline_exploration.ipynb`
3. `GNN.ipynb`

### 3) GNN data file requirement

`GNN.ipynb` expects a local file named `matbench_perovskites.json.gz` in this project root.

Create it with:

```python
import gzip, json
from matbench.bench import MatbenchBenchmark

mb = MatbenchBenchmark(autoload=False, subset=["matbench_perovskites"])
task = list(mb.tasks)[0]
task.load()

df = task.df.copy()
df["structure"] = df["structure"].apply(lambda s: s.as_dict())
df["e_form"] = df["e_form"].astype(float)

payload = {
    "columns": df.columns.tolist(),
    "data": df.values.tolist(),
}

with gzip.open("matbench_perovskites.json.gz", "wt") as f:
    json.dump(payload, f)
```

## Team

- Batyr Bodaubay
- Aruzhan Suleimenova
- Alisher Kalabayev
- Yernar Sakenov
- Arukhan Kenes
- Nurzhigit Askaraliyev

## References

- Dunn et al., *Benchmarking Materials Property Prediction Methods*, npj Computational Materials (2020)
- Wang et al., *CrabNet: A Transformer Model for Composition-Based Property Prediction*, npj Computational Materials (2021)
- Chen et al., *Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals*, Chemistry of Materials (2019)
