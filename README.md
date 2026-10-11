# Forecasting GitHub Star Growth: ARMA vs LSTM vs 1D CNN

How many stars will a repository get over the next few days? This project forecasts star growth for two very
different repositories, **facebook/react** (large and bursty) and **pallets/flask** (smaller and steady), and compares
a classical time-series model against two small neural networks on the same data.

The interesting part is that no model wins everywhere, and the reason why says something about the data.

## Approach

**Preparing the data**
- Removed entries stuck at a 4,000-star cap in the raw data, so they wouldn't bias the series.
- Enforced a daily frequency and forward-filled gaps, since cumulative counts can't go down.
- Modelled **daily new stars** (Δyₜ = yₜ − yₜ₋₁) instead of the cumulative total. The cumulative curve is strongly
  trending, while the increments are much closer to stationary, which ARMA needs and the neural nets also benefit from.
- Added lag features (1, 7, 14, 30 days), rolling mean/std/min/max over 7/14/30 days, and sine/cosine encodings of
  month and day of week.
- Split chronologically (first 80% train, last 20% test) and fitted every scaler on the training part only, so no
  information from the future leaks into training.

**Models**

| Model | Setup |
|---|---|
| ARMA | (p, q) chosen by AIC over a 0–5 grid: (0, 2) for react, (3, 3) for flask |
| LSTM | 30-day input window, 2 layers × 64 units, dropout 0.2 |
| 1D CNN | 3 conv blocks (64 → 128 → 256 filters, kernel 3), adaptive pooling, linear output |

Both neural models were trained with Adam (lr 1e-3) and MSE loss for 50 epochs, with seeds fixed to 42.

## Results

Next-day forecasts on the held-out test period (daily new stars):

| Repository | Model | MAE | RMSE |
|---|---|---|---|
| facebook/react | ARMA | **10.87** | 16.66 |
| | LSTM | 10.90 | **15.17** |
| | 1D CNN | 30.74 | 32.87 |
| pallets/flask | ARMA | 1.58 | 1.76 |
| | LSTM | 1.37 | 1.52 |
| | 1D CNN | **0.83** | **1.00** |

I also forecast 1, 3, 7, 14 and 30 days ahead by feeding each prediction back in as input.

**What I took away from this**

- On the steady flask series, the CNN's local filters pick up the short-term pattern well and roughly halve ARMA's error.
- On the volatile react series, the CNN does worst, and ARMA and the LSTM are tied on MAE.
- Over longer horizons the picture flips. Feeding predictions back in compounds errors, and on react the LSTM's 30-day MAE
  blew up to around 372, while ARMA stayed near 20. For trend extrapolation on noisy data, the simple model was the
  safest choice.
- The test periods are short (52 days for react, 197 for flask), so these numbers point in a direction rather than
  settle the question.

## Files

- `forecasting.ipynb`: cleaning, feature engineering, all three models, single- and multi-step evaluation, plots
- `REPORT.md`: a written summary of the method and results

## Running it

```bash
pip install -r requirements.txt
mkdir -p data
# put stars_data_cleaned.csv and repo_metadata.json in data/
jupyter notebook forecasting.ipynb
```

Preprocessed splits and the metrics table are written to `obs/Q3_obs/`.

*Built as part of the Statistical Methods in AI course at IIIT Hyderabad (Monsoon 2025).*
