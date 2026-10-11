
***

# Report: Time Series Forecasting of GitHub Stars

## 1. Data Preparation & Transformations
**Dataset:** The analysis used the `stars_data_cleaned.csv` dataset, focusing on two distinct repositories: **facebook/react** (high activity) and **pallets/flask** (moderate activity).

**Cleaning & Alignment:**
* **Timestamps:** Converted to datetime objects; sorted chronologically.
* **Anomalies:** Entries with the capped value of 4000 stars were removed to prevent bias.
* **Missing Data:** Daily frequencies were enforced. Missing dates were forward-filled to maintain the cumulative nature of star counts.

**Feature Engineering:**
* **Domain Transformation:** The cumulative star count $y_t$ was transformed into **daily increments** $\Delta y_t = y_t - y_{t-1}$ to achieve stationarity, which is essential for ARMA and beneficial for deep learning models.
* **Temporal Features:** Extracted year, month, day, day of week, and applied **cyclical encoding** (sine/cosine) for month and day-of-week to capture seasonality.
* **Lag Features:** Generated lags of 1, 7, 14, and 30 days to capture autocorrelation.
* **Rolling Statistics:** Calculated rolling mean, standard deviation, min, and max over windows of 7, 14, and 30 days.
* **Scaling:** Data was scaled using `MinMaxScaler` (for cumulative targets) and `StandardScaler` (for incremental inputs) *after* splitting to prevent data leakage.

**Split Strategy:** A chronological **80-20 train-test split** was used to respect the temporal order of the data.

## 2. Model Architectures
Three distinct modeling approaches were implemented and tuned:

### A. ARMA (Classical Baseline)
* **Method:** AutoRegressive Moving Average on the incremental data series.
* **Tuning:** Grid search was performed for parameters $(p, q)$ over the range $[0, 5]$ using Akaike Information Criterion (AIC) for selection.
    * *React Optimal Order:* $(0, 2)$
    * *Flask Optimal Order:* $(3, 3)$

### B. LSTM (Recurrent Neural Network)
Designed to capture long-term temporal dependencies.
* **Architecture:**
    * Input Layer: Sequence length of 30 days.
    * Hidden Layers: 2 LSTM layers with **64 hidden units** each.
    * Regularization: **Dropout (0.2)** to prevent overfitting.
    * Output: Fully connected layer predicting the next time step.
* **Training:** Trained for 50 epochs using the **Adam optimizer** (lr=0.001) and **MSE loss**.

### C. 1D CNN (Convolutional Neural Network)
Designed to extract local invariant features from short temporal windows.
* **Architecture:**
    * 3 Conv1d blocks with increasing filter sizes (**64 $\to$ 128 $\to$ 256**).
    * **Kernel Size:** 3.
    * **Pooling:** MaxPool1d after convolutions; Adaptive Average Pooling before the final dense layer.
    * Output: Fully connected layer (256 $\to$ 1).
* **Training:** Same hyperparameters as LSTM (50 epochs, Adam, MSE).

## 3. Quantitative Metrics & Generalization

### Single-Step Forecasting Results (Test Set)
Evaluation was performed on the held-out 20% test set.

| Repository | Model | MAE | RMSE | Insight |
| :--- | :--- | :--- | :--- | :--- |
| **facebook/react** | ARMA | **10.87** | 16.66 | Best interpretation of linear trend. |
| | LSTM | 10.90 | **15.17** | Best overall error minimization (RMSE). |
| | CNN | 30.74 | 32.87 | Struggled with high volatility in React data. |
| **pallets/flask** | ARMA | 1.58 | 1.76 | Solid baseline performance. |
| | LSTM | 1.37 | 1.52 | Good generalization. |
| | CNN | **0.83** | **1.00** | **Best Performance.** Captured local patterns effectively. |

### Multi-Step & Generalization
Forecasting horizons of 1, 3, 7, 14, and 30 days were evaluated recursively.
* **Generalization:** The CNN model generalized best on the stable `pallets/flask` dataset, achieving the lowest error. However, on the more volatile `facebook/react` dataset, the classical ARMA and LSTM models outperformed the CNN, suggesting CNNs may be sensitive to high variance without extensive tuning.
* **Error Propagation:** As expected, errors compounded over longer horizons. For `facebook/react`, LSTM error exploded at 30 days (MAE ~372), while ARMA remained stable (MAE ~20), highlighting the robustness of classical statistical methods for longer-term trend extrapolation in this specific domain.

### Visuals Summary
* **Forecast Plots:** Generated plots overlaying Predicted vs. Actual stars for both cumulative and incremental domains.
* **Residual Analysis:** Scatter plots of residuals confirmed that the best-performing models had errors centered around zero with constant variance (homoscedasticity).

## 4. Reproducibility
All results are fully reproducible using the provided `forecasting.ipynb` notebook.
* **Seeds:** Random seeds for `numpy` and `torch` were fixed to `42`.
* **Self-Contained:** The notebook includes all data loading, cleaning, feature engineering, and model training classes (`ARMAModel`, `LSTMModel`, `CNNModel`) within the file.
* **Execution:** Running the cells sequentially will regenerate the preprocessed datasets (saved to `obs/Q3_obs/`) and the metrics CSV.