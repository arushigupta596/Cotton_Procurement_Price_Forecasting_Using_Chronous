# Vardhaman Cotton Procurement Price Forecasting

A 3-month cotton price forecasting system for Vardhaman Group's raw cotton procurement, combining zero-shot time series prediction with weather-based correction.

---

## Overview

This project predicts weekly cotton candy rates (Rs/candy) for the next 12 weeks using a two-stage hybrid approach:

1. **Amazon Chronos T5-Small** — a pretrained transformer model that generates a zero-shot baseline forecast purely from price history (no training required)
2. **Weather Regression (GradientBoosting)** — a correction model that learns how weather conditions across India's cotton belt cause prices to deviate from the Chronos baseline

```
Final Predicted Price = Chronos Zero-Shot Forecast + Weather-Based Adjustment
```

---

## Data Sources

### Purchase Data (`Purchase Data 7 year - Final.xls`)
- 20,519 cotton purchase transactions from Sep 2019 to Jan 2026
- Fields: Purchase Date, Item Description, Bales, Candy Rate (Rs/candy), State
- 74 cotton varieties across 12 Indian states
- Top procurement states: Maharashtra (36%), Gujarat (28%), Haryana (10%), Rajasthan (9%), Madhya Pradesh (8%)

### Weekly Market Report (`15-WBS-24.01.26.xlsx`)
- Vardhaman Group's Corporate Raw Materials weekly report (dated 24 Jan 2026)
- 8 sheets covering: global supply-demand, Indian cotton balance sheets, state-wise crop data, arrivals, exports, and weekly price indices (NY Futures, Cotlook A', Shankar-6, Yarn Index)

### Weather Data (Open-Meteo API)
- Daily weather fetched for 5 major cotton-growing regions, weighted by Vardhaman's procurement share
- Variables: temperature (max/min/mean), rainfall, humidity, evapotranspiration, solar radiation
- Historical coverage: Sep 2019 – Jan 2026 (2,341 days)
- Forecast coverage: 16-day ahead from Open-Meteo forecast API

---

## Pipeline Architecture

### Stage 1: Data Preparation
Raw transactions are aggregated into a weekly bale-weighted average candy rate, producing a 277-point time series (`weekly_candy_rate.csv`).

### Stage 2: Chronos Baseline
Amazon Chronos-T5-Small tokenizes the numeric price series and generates 100 probabilistic future paths. Percentiles (p10, p25, median, p75, p90) are extracted for uncertainty quantification.

### Stage 3: Weather Feature Engineering
Daily weather from 5 regions is combined into a procurement-weighted average and transformed into 19 weekly features:

| Category | Features | Count |
|---|---|---|
| Direct aggregates | Max/min/mean temp, total rain, avg humidity, total ET0, total radiation | 7 |
| Derived signals | Temp range, heat stress days (>38C), heavy rain days (>20mm), dry days (<1mm) | 4 |
| Anomalies | Temperature and rainfall deviation from same-week historical mean | 2 |
| Lagged features | 1-week and 4-week lags of rain and temp, 4-week rolling averages | 6 |

### Stage 4: Walk-Forward Residual Training
A GradientBoostingRegressor is trained on Chronos prediction errors (residuals) using weather features:
- 14 walk-forward folds with 12-week horizons
- 168 total training samples
- The model learns when and how weather causes Chronos to over/under-predict

### Stage 5: Hybrid Forecast
The weather correction is added to each Chronos percentile to produce the final forecast.

---

## Weather Stations

| Region | Coordinates | Procurement Weight |
|---|---|---|
| Vidarbha, Maharashtra | 21.15N, 79.09E | 39.6% |
| Saurashtra, Gujarat | 22.30N, 70.78E | 30.8% |
| Hisar, Haryana | 29.15N, 75.72E | 11.0% |
| Sri Ganganagar, Rajasthan | 29.91N, 73.88E | 9.9% |
| Malwa, Madhya Pradesh | 22.72N, 75.86E | 8.8% |

---

## Installation

```bash
pip install pandas numpy torch chronos-forecasting matplotlib scikit-learn requests openpyxl xlrd
```

Python 3.10+ required. No GPU needed — runs on CPU.

---

## Usage

### Chronos-Only Forecast (Baseline)
```bash
python chronos_forecast.py
```
Outputs:
- `forecast_3months.csv` — 12-week predictions with percentiles
- `cotton_price_forecast.png` — historical + forecast chart

### Hybrid Weather-Augmented Forecast
```bash
python weather_hybrid_forecast.py
```
Outputs:
- `forecast_3months_hybrid.csv` — 12-week predictions with Chronos baseline, weather adjustment, and final percentiles
- `cotton_price_forecast_hybrid.png` — chart with Chronos baseline (dashed) and hybrid forecast (solid)
- `weather_impact_analysis.png` — feature importance ranking and per-week adjustment bars
- `weather_cache_historical.csv` — cached weather data (reused on subsequent runs)

Estimated runtime: 2-4 minutes (dominated by Chronos inference across walk-forward folds).

---

## Output Files

| File | Description |
|---|---|
| `weekly_candy_rate.csv` | 277-week bale-weighted average price series |
| `forecast_3months.csv` | Chronos-only 12-week forecast |
| `cotton_price_forecast.png` | Chronos-only forecast chart |
| `forecast_3months_hybrid.csv` | Hybrid forecast with weather adjustments |
| `cotton_price_forecast_hybrid.png` | Hybrid forecast chart |
| `weather_impact_analysis.png` | Feature importance + weekly adjustment chart |
| `weather_cache_historical.csv` | Cached daily weather data (avoids re-fetching) |

---

## Key Findings

### Top Weather Drivers
| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | temp_lag4 | 14.8% | Temperature from 4 weeks prior |
| 2 | precip_roll4 | 10.3% | 4-week cumulative rainfall pattern |
| 3 | temp_lag1 | 8.8% | Last week's temperature |
| 4 | humidity_avg | 8.3% | Weekly average humidity (pest/disease pressure) |
| 5 | radiation_sum | 8.1% | Solar radiation (crop drying conditions) |

Lagged features dominate, confirming that cotton prices respond to weather with a 1-4 week delay: weather affects crop condition, which affects market arrivals, which then moves prices.

### Sample Forecast (Jan 2026 Run)

| Period | Chronos-Only | Hybrid (Weather-Adjusted) |
|---|---|---|
| Late Jan 2026 | ~57,000 | ~56,100 |
| Feb 2026 avg | ~55,800 | ~54,700 |
| Mar 2026 avg | ~54,900 | ~54,000 |
| Apr 2026 avg | ~54,200 | ~53,000 |

Weather conditions at forecast time pushed all weeks downward by Rs 375-1,443 per candy.

---

## Technical Notes

- **Candy**: Traditional Indian cotton trading unit. 1 candy = 355.62 kg = 20 maunds.
- **Rs/candy**: Indian Rupees per candy — the standard unit for raw cotton pricing in India.
- **Chronos zero-shot**: No model training on this specific data. The pretrained model generalizes from millions of other time series.
- **Walk-forward validation**: Prevents data leakage by simulating real-time forecasting — the model only sees past data at each fold.
- **Weather caching**: Historical weather is saved locally after the first API call. Delete `weather_cache_historical.csv` to force a refresh.
- **Open-Meteo free tier**: No API key required. Rate-limited with 0.5s delays between calls.

---

## Limitations

- The Chronos zero-shot forecast is smooth and does not capture sudden price spikes driven by policy changes, trade disruptions, or speculative trading
- Weather forecast data is limited to 16 days from Open-Meteo's free API; beyond that, the model falls back on historical climatological patterns
- The residual model has a relatively small training set (168 samples from walk-forward folds)
- The model does not account for government MSP changes, import/export policy shifts, or global commodity market shocks

---

## Dependencies

| Package | Purpose |
|---|---|
| pandas | Data manipulation |
| numpy | Numerical operations |
| torch | PyTorch backend for Chronos |
| chronos-forecasting | Amazon Chronos T5-Small model |
| scikit-learn | GradientBoostingRegressor for weather residuals |
| matplotlib | Visualization |
| requests | Open-Meteo API calls |
| openpyxl / xlrd | Excel file reading |
