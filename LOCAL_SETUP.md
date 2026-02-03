# Local Setup (UV + Jupyter)

## 1) Install UV (one-time)
PowerShell:
```
irm https://astral.sh/uv/install.ps1 | iex
```

## 2) Install Python 3.11 via UV
```
uv python install 3.11
```

## 3) Create venv + install dependencies
```
uv venv .venv
uv lock
uv sync
```

## 4) Data placement
Place the training data file here:
- `usd_jpy_1min_20231028_20260131_utc.parquet`

If you want to store it elsewhere, edit `DATA_PATH` in:
- `米ドルFxモデル構築_v_5_local.ipynb`

## 5) Run JupyterLab
```
uv run jupyter lab
```
Open `米ドルFxモデル構築_v_5_local.ipynb` and run cells top to bottom.

## GPU note (CatBoost)
- The notebook defaults to `USE_GPU = True`.
- If CUDA is not available, set `USE_GPU = False` at the top of the notebook.
