# TKAN-Time-Series-for-FinalDataNN

A **TKAN (Temporal Kolmogorov-Arnold Network)** implementation in Keras 3 for time-series forecasting on the `FinalDataNN.xlsx` dataset. The project builds a custom RNN layer (TKAN) whose cell replaces the usual linear LSTM/GRU gates with B-spline-based **KAN (Kolmogorov-Arnold Network)** sub-layers for local memory modeling.

## Repository Contents

| File | Description |
|---|---|
| `FinalDataNN.xlsx` | Input dataset (columns `P`, `Y`, `Ptest`, `Ytest`) |
| `TKAN_Keras_TS_for_FinalDataNN.ipynb` | Main notebook (Google Colab output) |
| `tkan_keras_ts_for_finaldatann.py` | Python script version of the same notebook |

## Model Architecture

- **KANLinear**: B-spline-based KAN layer implementation (a Keras 3 port of `keras_efficient_kan`), combining a base linear path (ReLU) with a spline path.
- **TKANCellNoGF**: TKAN cell with local memory in each KAN sub-layer, but with the Global Feedback path removed — the `i, f, c~` gates depend only on the input `x_t`, not on `h_{t-1}`.
- **TKAN**: A `keras.layers.RNN` wrapper around the cell above, producing a full recurrent layer.
- **Final model**: Several stacked TKAN layers followed by a `Dense` output layer that predicts the forecast horizon.

Data flow: the `P` time series is converted into sliding windows of length `Context_Len`, each mapped to a `Pred_Len`-length segment of the `Y` series.

## Requirements

```bash
pip install keras numpy pandas matplotlib scikit-learn openpyxl
```

A Keras 3 backend is also required (the code defaults to `torch`):

```bash
pip install torch      # or: pip install jax jaxlib   /   pip install tensorflow
```

> Note: the backend is set via `os.environ["KERAS_BACKEND"] = "torch"` **before** Keras is imported. Edit this line to switch backends.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/BenyaminTo/TKAN-Time-Series-for-FinalDataNN.git
   cd TKAN-Time-Series-for-FinalDataNN
   ```
2. Place `FinalDataNN.xlsx` where the script expects it (the code defaults to the Colab path `/content/FinalDataNN.xlsx`; for a local run, update the `excel_path` variable to point to the file in this folder).
3. Run the script:
   ```bash
   python tkan_keras_ts_for_finaldatann.py
   ```
   or open `TKAN_Keras_TS_for_FinalDataNN.ipynb` in Jupyter/Colab.

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `Spline_Order` | 3 | B-spline order for every KANLinear |
| `Grid_Size` | 3 | Grid size for every KANLinear |
| `N_Sub_KANs` | 3 | Number of KAN sub-layers per TKAN cell |
| `Context_Len` | 120 | Input window length |
| `Pred_Len` | 20 | Forecast horizon length |
| `H_Dims` | 32 | Hidden dimension |
| `Num_Layers` | 2 | Number of stacked TKAN layers |
| `Batch_Size` | 32 | Batch size |
| `N_Epochs` | 500 | Number of training epochs |
| `Learning_Rate` | 3e-4 | Learning rate (AdamW optimizer) |
| `Seq_Step` | 5 | Step used to split each window into smaller time steps |

## Evaluation Metrics

After training, a table of metrics is computed and printed for the train/test sets, including:

- MAE, MSE, RMSE, R²
- SNR (dB)
- SMAPE (%)
- NRMSE (range-normalized and std-normalized)
- Pointwise relative error (for reference only)
- Total trainable parameters, training/inference time, and model structure

Plots are also generated for per-epoch MAE/MSE and for the model's predictions vs. actual values over the full test set.

## Important Note on "Speedup"

The reported `[PLACEHOLDER, not physical] Speedup` value is computed against a mock simulation function (`physics_simulation`, based on `time.sleep`) and is **only a placeholder**, not a real physical benchmark.

## Source

This file was automatically generated from a Google Colab notebook:
`https://colab.research.google.com/drive/1puZzo6smZwGb7h4Y9HN-OyS5LFxY-N9N`

## License

No license is currently specified in the repository. If you plan to publish this publicly, consider adding an appropriate `LICENSE` file.
