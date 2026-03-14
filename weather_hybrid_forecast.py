import pandas as pd
import numpy as np
import torch
import requests
import time
import os
import re
from datetime import datetime, timedelta
from chronos import Chronos2Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
REGIONS = {
    'Maharashtra_Vidarbha': {'lat': 21.15, 'lon': 79.09, 'weight': 0.396},
    'Gujarat_Saurashtra':   {'lat': 22.30, 'lon': 70.78, 'weight': 0.308},
    'Haryana_Hisar':        {'lat': 29.15, 'lon': 75.72, 'weight': 0.110},
    'Rajasthan_Ganganagar': {'lat': 29.91, 'lon': 73.88, 'weight': 0.099},
    'MP_Malwa':             {'lat': 22.72, 'lon': 75.86, 'weight': 0.088},
}

WEATHER_VARS = [
    'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
    'precipitation_sum', 'relative_humidity_2m_mean',
    'et0_fao_evapotranspiration', 'shortwave_radiation_sum'
]

PREDICTION_LENGTH = 12  # weeks (~3 months)
CACHE_FILE = 'weather_cache_historical.csv'

COMMODITY_COLS = ['ny_futures', 'cotlook_a', 'china_b', 'yarn_index', 'shankar6', 'forex']
# Covariates we can forecast into the future (via yfinance)
FUTURE_COVARIATE_NAMES = ['ny_futures', 'forex']

# ============================================================
# STEP 1: COMMODITY DATA EXTRACTION
# ============================================================
def parse_commodity_week(val):
    """Parse diverse week date formats from the spreadsheet."""
    if isinstance(val, pd.Timestamp):
        return val
    try:
        return pd.to_datetime(val)
    except Exception:
        pass
    s = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', str(val))
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT


def extract_commodity_data(spreadsheet_path='Data/15-WBS-24.01.26.xlsx'):
    """Extract 6 weekly commodity series from Vardhaman market report."""
    print("\nExtracting commodity data from market report...")
    df = pd.read_excel(spreadsheet_path, sheet_name='spreadsheets-8', header=None)

    data = df.iloc[4:].copy()
    col_names = ['week_str', 'ny_futures', 'cotlook_a', 'china_b', 'yarn_index',
                 'shankar6', 'forex'] + [f'gap_{i}' for i in range(9)]
    data.columns = col_names[:len(data.columns)]

    # Remove summary/empty rows
    data = data.dropna(subset=['week_str'])
    data = data[~data['week_str'].astype(str).str.contains('Avg|Max|Min', na=False)]

    # Parse dates
    data['week'] = data['week_str'].apply(parse_commodity_week)
    data = data.dropna(subset=['week'])

    # Keep only the 6 commodity columns + week
    commodity = data[['week'] + COMMODITY_COLS].copy()
    for col in COMMODITY_COLS:
        commodity[col] = pd.to_numeric(commodity[col], errors='coerce')

    commodity = commodity.sort_values('week').reset_index(drop=True)

    # Create a week key (Monday-start) for alignment with price data
    commodity['week_key'] = commodity['week'].dt.to_period('W-SUN').apply(lambda r: r.start_time)

    print(f"  Commodity data: {len(commodity)} weeks ({commodity['week'].min().date()} to {commodity['week'].max().date()})")
    print(f"  Columns: {COMMODITY_COLS}")

    return commodity


def align_commodity_with_prices(commodity_df, weekly_df):
    """Align commodity series to the same weeks as the price series."""
    weekly_df = weekly_df.copy()
    weekly_df['week_key'] = weekly_df['week'].dt.to_period('W-SUN').apply(lambda r: r.start_time)

    # Deduplicate commodity by week_key (keep last occurrence)
    commodity_dedup = commodity_df.drop_duplicates(subset='week_key', keep='last')

    merged = weekly_df.merge(
        commodity_dedup[['week_key'] + COMMODITY_COLS],
        on='week_key',
        how='left'
    )

    # Forward-fill any NaN (for weeks where commodity data is missing)
    for col in COMMODITY_COLS:
        merged[col] = merged[col].ffill().bfill()

    print(f"  Aligned {len(merged)} weeks of commodity data with price series")
    print(f"  Latest values: NY={merged['ny_futures'].iloc[-1]:.2f}, "
          f"Cotlook={merged['cotlook_a'].iloc[-1]:.2f}, "
          f"Forex={merged['forex'].iloc[-1]:.2f}")

    return merged


def fetch_live_commodity_forecasts(commodity_df, prediction_length=12):
    """Fetch live NY Futures and Forex via yfinance for future covariates."""
    print("\nFetching live commodity data via yfinance...")
    future_covariates = {}

    try:
        import yfinance as yf

        # NY Cotton Futures (CT=F)
        try:
            ct = yf.Ticker("CT=F")
            hist = ct.history(period="5d")
            if len(hist) > 0:
                latest_ny = hist['Close'].iloc[-1]
                future_covariates['ny_futures'] = np.full(prediction_length, latest_ny)
                print(f"  NY Futures (CT=F): {latest_ny:.2f} usc/lb (flat-forward {prediction_length} weeks)")
            else:
                raise ValueError("No data returned")
        except Exception as e:
            print(f"  NY Futures fetch failed ({e}), using last known value")
            latest_ny = commodity_df['ny_futures'].iloc[-1]
            future_covariates['ny_futures'] = np.full(prediction_length, latest_ny)

        # USD/INR Forex
        try:
            inr = yf.Ticker("INR=X")
            hist = inr.history(period="5d")
            if len(hist) > 0:
                latest_fx = hist['Close'].iloc[-1]
                future_covariates['forex'] = np.full(prediction_length, latest_fx)
                print(f"  USD/INR Forex (INR=X): {latest_fx:.2f} (flat-forward {prediction_length} weeks)")
            else:
                raise ValueError("No data returned")
        except Exception as e:
            print(f"  Forex fetch failed ({e}), using last known value")
            latest_fx = commodity_df['forex'].iloc[-1]
            future_covariates['forex'] = np.full(prediction_length, latest_fx)

    except ImportError:
        print("  yfinance not installed, using last known values for future covariates")
        future_covariates['ny_futures'] = np.full(prediction_length, commodity_df['ny_futures'].iloc[-1])
        future_covariates['forex'] = np.full(prediction_length, commodity_df['forex'].iloc[-1])

    return future_covariates


# ============================================================
# STEP 2: WEATHER DATA FETCHING
# ============================================================
def fetch_historical_weather(lat, lon, start_date, end_date, retries=3):
    url = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'daily': ','.join(WEATHER_VARS),
        'timezone': 'Asia/Kolkata',
    }
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            daily = data['daily']
            df = pd.DataFrame(daily)
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            return df
        except Exception as e:
            print(f"  Retry {attempt+1}/{retries}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch weather for ({lat}, {lon})")


def fetch_forecast_weather(lat, lon, retries=3):
    url = 'https://api.open-meteo.com/v1/forecast'
    forecast_vars = [
        'temperature_2m_max', 'temperature_2m_min',
        'precipitation_sum', 'et0_fao_evapotranspiration',
        'shortwave_radiation_sum'
    ]
    params = {
        'latitude': lat,
        'longitude': lon,
        'daily': ','.join(forecast_vars),
        'timezone': 'Asia/Kolkata',
        'forecast_days': 16,
    }
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            daily = data['daily']
            df = pd.DataFrame(daily)
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            df['temperature_2m_mean'] = (df['temperature_2m_max'] + df['temperature_2m_min']) / 2
            if 'relative_humidity_2m_mean' not in df.columns:
                df['relative_humidity_2m_mean'] = np.nan
            return df[WEATHER_VARS]
        except Exception as e:
            print(f"  Retry {attempt+1}/{retries}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch forecast for ({lat}, {lon})")


def fetch_all_regions_weather(regions, start_date, end_date, mode='historical'):
    all_dfs = []
    weights = []
    for name, cfg in regions.items():
        print(f"  Fetching {mode} weather for {name} ({cfg['lat']}, {cfg['lon']})...")
        if mode == 'historical':
            df = fetch_historical_weather(cfg['lat'], cfg['lon'], start_date, end_date)
        else:
            df = fetch_forecast_weather(cfg['lat'], cfg['lon'])
        all_dfs.append(df)
        weights.append(cfg['weight'])
        time.sleep(0.5)

    common_idx = all_dfs[0].index
    for df in all_dfs[1:]:
        common_idx = common_idx.intersection(df.index)

    weighted = pd.DataFrame(index=common_idx, columns=WEATHER_VARS, dtype=float)
    weighted[:] = 0.0
    for df, w in zip(all_dfs, weights):
        for var in WEATHER_VARS:
            if var in df.columns:
                weighted[var] += df.loc[common_idx, var].fillna(0).values * w

    return weighted


def load_or_fetch_historical_weather(regions, start_date, end_date):
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached weather data from {CACHE_FILE}...")
        df = pd.read_csv(CACHE_FILE, parse_dates=['time'], index_col='time')
        if df.index.max() >= pd.Timestamp(end_date) - pd.Timedelta(days=7):
            return df
        print("Cache is stale, re-fetching...")

    print("Fetching historical weather data from Open-Meteo...")
    df = fetch_all_regions_weather(regions, start_date, end_date, mode='historical')
    df.to_csv(CACHE_FILE)
    print(f"Cached weather data to {CACHE_FILE}")
    return df


# ============================================================
# STEP 3: WEEKLY FEATURE ENGINEERING
# ============================================================
def engineer_weekly_features(daily_weather, weekly_dates):
    records = []
    for week_start in weekly_dates:
        week_end = week_start + pd.Timedelta(days=6)
        mask = (daily_weather.index >= week_start) & (daily_weather.index <= week_end)
        week_data = daily_weather.loc[mask]

        if len(week_data) < 3:
            records.append({col: np.nan for col in [
                'temp_max', 'temp_min', 'temp_mean', 'precip_sum', 'humidity_avg',
                'et0_sum', 'radiation_sum', 'temp_range', 'heat_stress_days',
                'heavy_rain_days', 'dry_days'
            ]})
            continue

        row = {
            'temp_max': week_data['temperature_2m_max'].max(),
            'temp_min': week_data['temperature_2m_min'].min(),
            'temp_mean': week_data['temperature_2m_mean'].mean(),
            'precip_sum': week_data['precipitation_sum'].sum(),
            'humidity_avg': week_data['relative_humidity_2m_mean'].mean(),
            'et0_sum': week_data['et0_fao_evapotranspiration'].sum(),
            'radiation_sum': week_data['shortwave_radiation_sum'].sum(),
            'temp_range': week_data['temperature_2m_max'].max() - week_data['temperature_2m_min'].min(),
            'heat_stress_days': (week_data['temperature_2m_max'] > 38).sum(),
            'heavy_rain_days': (week_data['precipitation_sum'] > 20).sum(),
            'dry_days': (week_data['precipitation_sum'] < 1).sum(),
        }
        records.append(row)

    features = pd.DataFrame(records, index=weekly_dates)
    features.index.name = 'week'

    iso_week = features.index.isocalendar().week.values
    clim_temp = features.groupby(iso_week)['temp_mean'].transform('mean')
    clim_precip = features.groupby(iso_week)['precip_sum'].transform('mean')
    features['temp_anomaly'] = features['temp_mean'] - clim_temp
    features['precip_anomaly'] = features['precip_sum'] - clim_precip

    features['precip_lag1'] = features['precip_sum'].shift(1)
    features['precip_lag4'] = features['precip_sum'].shift(4)
    features['temp_lag1'] = features['temp_mean'].shift(1)
    features['temp_lag4'] = features['temp_mean'].shift(4)
    features['precip_roll4'] = features['precip_sum'].rolling(4).mean()
    features['temp_roll4'] = features['temp_mean'].rolling(4).mean()

    return features


# ============================================================
# STEP 4: CHRONOS-2 MULTIVARIATE BASELINE
# ============================================================
def load_chronos2_pipeline():
    print("Loading Chronos-2 model (amazon/chronos-2, multivariate)...")
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        dtype=torch.float32,
    )
    print(f"  Model loaded — {len(pipeline.quantiles)} quantiles: "
          f"[{pipeline.quantiles[0]:.2f}, ..., {pipeline.quantiles[-1]:.2f}]")
    return pipeline


def run_chronos2_multivariate(pipeline, price_series, commodity_data,
                               future_covariates=None, prediction_length=12):
    """Run Chronos-2 with commodity covariates for multivariate zero-shot forecast."""
    target = torch.tensor(price_series.values, dtype=torch.float32)

    # Build past covariates from commodity data (aligned to price series)
    past_covariates = {}
    for col in COMMODITY_COLS:
        if col in commodity_data.columns:
            past_covariates[col] = torch.tensor(
                commodity_data[col].values, dtype=torch.float32
            )

    # Build input dict
    input_dict = {
        "target": target,
        "past_covariates": past_covariates,
    }

    # Add future covariates if available
    if future_covariates:
        fc_dict = {}
        for name, values in future_covariates.items():
            fc_dict[name] = torch.tensor(values[:prediction_length], dtype=torch.float32)
        input_dict["future_covariates"] = fc_dict

    forecast = pipeline.predict(
        inputs=[input_dict],
        prediction_length=prediction_length,
    )

    # Output shape: (n_variates=1, n_quantiles=21, prediction_length)
    forecast_tensor = forecast[0]  # first (and only) input
    forecast_np = forecast_tensor.squeeze(0).numpy()  # (n_quantiles, prediction_length)

    # Extract specific quantiles from the 21 available
    # quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, ..., 0.9, 0.95, 0.99]
    quantile_list = pipeline.quantiles
    q_indices = {
        'p10': quantile_list.index(0.1),
        'p25': quantile_list.index(0.25),
        'median': quantile_list.index(0.5),
        'p75': quantile_list.index(0.75),
        'p90': quantile_list.index(0.9),
    }

    return {
        'median': forecast_np[q_indices['median']],
        'p10': forecast_np[q_indices['p10']],
        'p25': forecast_np[q_indices['p25']],
        'p75': forecast_np[q_indices['p75']],
        'p90': forecast_np[q_indices['p90']],
    }


def run_chronos2_univariate(pipeline, price_series, prediction_length=12):
    """Run Chronos-2 univariate (no covariates) for walk-forward folds."""
    target = torch.tensor(price_series.values, dtype=torch.float32)

    forecast = pipeline.predict(
        inputs=[{"target": target}],
        prediction_length=prediction_length,
    )

    forecast_tensor = forecast[0]
    forecast_np = forecast_tensor.squeeze(0).numpy()

    quantile_list = pipeline.quantiles
    median_idx = quantile_list.index(0.5)

    return {
        'median': forecast_np[median_idx],
    }


# ============================================================
# STEP 5: WALK-FORWARD RESIDUAL TRAINING
# ============================================================
def walk_forward_train(weekly_df, weather_features, pipeline):
    """Train weather residual model using walk-forward validation.
    Uses Chronos-2 univariate for efficiency during walk-forward folds."""
    merged = weekly_df.merge(weather_features, left_on='week', right_index=True, how='inner')
    merged = merged.dropna().reset_index(drop=True)

    feature_cols = [c for c in weather_features.columns if c in merged.columns]
    print(f"\nWeather feature columns ({len(feature_cols)}): {feature_cols}")

    all_X, all_y = [], []
    all_chronos_mae = []

    min_train = 100
    step = 12
    cutoffs = list(range(min_train, len(merged) - step, step))

    print(f"\nWalk-forward validation: {len(cutoffs)} folds")
    print("-" * 60)

    for i, cutoff in enumerate(cutoffs):
        test_end = min(cutoff + step, len(merged))
        if test_end - cutoff < 4:
            continue

        train_prices = merged['avg_candy_rate'].iloc[:cutoff]
        actual_prices = merged['avg_candy_rate'].iloc[cutoff:test_end].values
        test_weather = merged[feature_cols].iloc[cutoff:test_end].values

        pred_len = test_end - cutoff
        chronos_forecast = run_chronos2_univariate(pipeline, train_prices, prediction_length=pred_len)
        chronos_median = chronos_forecast['median']

        residuals = actual_prices - chronos_median

        all_X.append(test_weather)
        all_y.append(residuals)

        chronos_mae = np.mean(np.abs(actual_prices - chronos_median))
        all_chronos_mae.append(chronos_mae)

        print(f"  Fold {i+1}: cutoff={cutoff}, test_size={pred_len}, "
              f"Chronos-2 MAE=₹{chronos_mae:,.0f}")

    X_train = np.vstack(all_X[:-1]) if len(all_X) > 1 else np.vstack(all_X)
    y_train = np.concatenate(all_y[:-1]) if len(all_y) > 1 else np.concatenate(all_y)

    print(f"\nTraining residual model on {len(y_train)} samples...")
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        loss='huber',
        random_state=42,
    )
    model.fit(X_train, y_train)

    if len(all_X) > 1:
        X_test = all_X[-1]
        y_test = all_y[-1]
        residual_pred = model.predict(X_test)
        last_chronos_mae = all_chronos_mae[-1]
        last_actual = merged['avg_candy_rate'].iloc[cutoffs[-1]:cutoffs[-1]+len(y_test)].values
        last_chronos_median = last_actual - y_test
        hybrid_pred = last_chronos_median + residual_pred
        hybrid_mae = np.mean(np.abs(last_actual - hybrid_pred))

        print(f"\nHold-out fold results:")
        print(f"  Chronos-2 only MAE:  ₹{last_chronos_mae:,.0f}")
        print(f"  Hybrid MAE:          ₹{hybrid_mae:,.0f}")
        improvement = (last_chronos_mae - hybrid_mae) / last_chronos_mae * 100
        print(f"  Improvement:         {improvement:+.1f}%")

    final_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        loss='huber',
        random_state=42,
    )
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    final_model.fit(X_all, y_all)
    print(f"Final model trained on {len(y_all)} total residual samples.")

    importances = pd.Series(final_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print(f"\nTop 10 weather feature importances:")
    for feat, imp in importances.head(10).items():
        print(f"  {feat:25s} {imp:.4f}")

    return final_model, feature_cols, importances


# ============================================================
# STEP 6: GENERATE HYBRID FORECAST
# ============================================================
def generate_hybrid_forecast(pipeline, weekly_df, commodity_aligned, weather_features,
                              forecast_weather_features, model, feature_cols,
                              future_covariates):
    print("\n" + "=" * 60)
    print("GENERATING FINAL HYBRID FORECAST")
    print("Chronos-2 Multivariate (Commodity Covariates) + Weather Residual")
    print("=" * 60)

    # Run Chronos-2 with commodity covariates
    chronos_result = run_chronos2_multivariate(
        pipeline,
        weekly_df['avg_candy_rate'],
        commodity_aligned,
        future_covariates=future_covariates,
        prediction_length=PREDICTION_LENGTH,
    )
    print(f"Chronos-2 multivariate baseline median range: "
          f"₹{chronos_result['median'].min():,.0f} – ₹{chronos_result['median'].max():,.0f}")

    # Weather residual correction
    X_forecast = forecast_weather_features[feature_cols].values
    residual_pred = model.predict(X_forecast)
    print(f"Weather adjustments: {[f'₹{r:+,.0f}' for r in residual_pred]}")

    last_date = weekly_df['week'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                                  periods=PREDICTION_LENGTH, freq='W-MON')

    forecast_df = pd.DataFrame({
        'week': future_dates,
        'chronos_median': chronos_result['median'],
        'weather_adjustment': residual_pred,
        'predicted_candy_rate_p10': chronos_result['p10'] + residual_pred,
        'predicted_candy_rate_p25': chronos_result['p25'] + residual_pred,
        'predicted_candy_rate_median': chronos_result['median'] + residual_pred,
        'predicted_candy_rate_p75': chronos_result['p75'] + residual_pred,
        'predicted_candy_rate_p90': chronos_result['p90'] + residual_pred,
    })

    forecast_df.to_csv('forecast_3months_hybrid.csv', index=False)
    print(f"\nSaved forecast_3months_hybrid.csv")
    print(forecast_df.to_string(index=False))

    return forecast_df, chronos_result


# ============================================================
# STEP 7: VISUALIZATION
# ============================================================
def plot_hybrid_forecast(weekly_df, forecast_df, chronos_result):
    future_dates = forecast_df['week']

    fig, ax = plt.subplots(figsize=(16, 7))
    hist_window = weekly_df.tail(52)
    ax.plot(hist_window['week'], hist_window['avg_candy_rate'],
            color='#2c3e50', linewidth=2, label='Historical (Weekly Avg)')

    ax.plot(future_dates, chronos_result['median'],
            color='#95a5a6', linewidth=1.5, linestyle='--', label='Chronos-2 Multivariate Baseline')

    ax.plot(future_dates, forecast_df['predicted_candy_rate_median'],
            color='#e74c3c', linewidth=2.5, label='Hybrid Forecast (Median)', marker='o', markersize=5)
    ax.fill_between(future_dates,
                     forecast_df['predicted_candy_rate_p25'],
                     forecast_df['predicted_candy_rate_p75'],
                     alpha=0.3, color='#e74c3c', label='50% Confidence Interval')
    ax.fill_between(future_dates,
                     forecast_df['predicted_candy_rate_p10'],
                     forecast_df['predicted_candy_rate_p90'],
                     alpha=0.15, color='#e74c3c', label='80% Confidence Interval')

    ax.set_title('Vardhaman Cotton Procurement Price — 3-Month Forecast\n'
                 '(Hybrid: Chronos-2 Multivariate Zero-Shot + Weather Regression)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Candy Rate (Rs/candy)', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    plt.tight_layout()
    plt.savefig('cotton_price_forecast_hybrid.png', dpi=150, bbox_inches='tight')
    print("\nSaved cotton_price_forecast_hybrid.png")
    plt.close()


def plot_weather_impact(importances, forecast_df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1.2, 1]})

    top_n = min(15, len(importances))
    top_imp = importances.head(top_n)
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, top_n))
    ax1.barh(range(top_n), top_imp.values, color=colors)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(top_imp.index, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel('Feature Importance', fontsize=11)
    ax1.set_title('Weather Feature Importance in Residual Model', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    weeks = forecast_df['week'].dt.strftime('%b %d')
    adjustments = forecast_df['weather_adjustment'].values
    bar_colors = ['#27ae60' if a > 0 else '#e74c3c' for a in adjustments]
    ax2.bar(weeks, adjustments, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xlabel('Forecast Week', fontsize=11)
    ax2.set_ylabel('Price Adjustment (Rs/candy)', fontsize=11)
    ax2.set_title('Weather-Based Price Adjustments for Next 12 Weeks', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:+,.0f}'))

    plt.tight_layout()
    plt.savefig('weather_impact_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved weather_impact_analysis.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("COMMODITY-AWARE COTTON PRICE FORECAST")
    print("Hybrid: Chronos-2 Multivariate Zero-Shot + Weather Regression")
    print("Covariates: NY Futures, Cotlook A', China B, Yarn Index, Forex")
    print("=" * 60)

    # Load price data
    weekly = pd.read_csv('weekly_candy_rate.csv', parse_dates=['week'])
    print(f"\nPrice data: {len(weekly)} weeks ({weekly['week'].min().date()} to {weekly['week'].max().date()})")

    # Step 1: Extract commodity data
    commodity = extract_commodity_data()
    commodity_aligned = align_commodity_with_prices(commodity, weekly)

    # Save commodity data for dashboard use
    commodity.to_csv('commodity_weekly.csv', index=False)
    print(f"  Saved commodity_weekly.csv ({len(commodity)} rows)")

    # Step 2: Fetch live commodity forecasts for future covariates
    future_covariates = fetch_live_commodity_forecasts(commodity_aligned, PREDICTION_LENGTH)

    # Step 3: Historical weather
    start_date = (weekly['week'].min() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = weekly['week'].max().strftime('%Y-%m-%d')
    daily_weather = load_or_fetch_historical_weather(REGIONS, start_date, end_date)
    print(f"Historical weather: {len(daily_weather)} days")

    # Step 4: Weekly weather features
    print("\nEngineering weekly weather features...")
    weather_features = engineer_weekly_features(daily_weather, weekly['week'])
    weather_features = weather_features.dropna()
    print(f"Weather features: {weather_features.shape[0]} weeks x {weather_features.shape[1]} features")

    # Step 5: Load Chronos-2 & walk-forward train residual model
    pipeline = load_chronos2_pipeline()
    model, feature_cols, importances = walk_forward_train(weekly, weather_features, pipeline)

    # Step 6: Forecast weather
    print("\nFetching forecast weather data...")
    forecast_daily = fetch_all_regions_weather(REGIONS, None, None, mode='forecast')
    print(f"Forecast weather: {len(forecast_daily)} days")

    last_date = weekly['week'].max()
    forecast_weeks = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                                    periods=PREDICTION_LENGTH, freq='W-MON')

    # For forecast features, we also need historical context for lags
    full_daily = pd.concat([daily_weather, forecast_daily])
    full_daily = full_daily[~full_daily.index.duplicated(keep='last')]
    full_daily = full_daily.sort_index()

    all_weeks = list(weekly['week']) + list(forecast_weeks)
    all_features = engineer_weekly_features(full_daily, pd.DatetimeIndex(all_weeks))

    forecast_features = all_features.loc[forecast_weeks]

    # Fill any NaN in forecast features with historical medians
    for col in feature_cols:
        if col in forecast_features.columns and forecast_features[col].isna().any():
            hist_median = weather_features[col].median() if col in weather_features.columns else 0
            forecast_features[col] = forecast_features[col].fillna(hist_median)

    # Step 7: Generate hybrid forecast (Chronos-2 multivariate + weather residual)
    forecast_df, chronos_result = generate_hybrid_forecast(
        pipeline, weekly, commodity_aligned, weather_features,
        forecast_features, model, feature_cols, future_covariates)

    # Step 8: Plots
    print("\nGenerating visualizations...")
    plot_hybrid_forecast(weekly, forecast_df, chronos_result)
    plot_weather_impact(importances, forecast_df)

    print("\n" + "=" * 60)
    print("DONE — All outputs saved:")
    print("  forecast_3months_hybrid.csv   (multivariate commodity-aware)")
    print("  commodity_weekly.csv          (extracted commodity series)")
    print("  cotton_price_forecast_hybrid.png")
    print("  weather_impact_analysis.png")
    print("=" * 60)


if __name__ == '__main__':
    main()
