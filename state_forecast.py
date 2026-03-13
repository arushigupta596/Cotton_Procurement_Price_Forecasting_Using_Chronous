import pandas as pd
import numpy as np
import torch
import requests
import time
import os
from chronos import ChronosPipeline
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STATE CONFIGURATION
# ============================================================
STATES = {
    'Maharashtra': {
        'data_name': 'Maharashtra',
        'lat': 21.15, 'lon': 79.09,
        'color': '#e74c3c',
    },
    'Gujarat': {
        'data_name': 'Gujarat',
        'lat': 22.30, 'lon': 70.78,
        'color': '#3498db',
    },
    'Haryana': {
        'data_name': 'Haryana',
        'lat': 29.15, 'lon': 75.72,
        'color': '#27ae60',
    },
    'Rajasthan': {
        'data_name': 'Rajasthan',
        'lat': 29.91, 'lon': 73.88,
        'color': '#f39c12',
    },
    'Madhya Pradesh': {
        'data_name': 'Madhya Prades',
        'lat': 22.72, 'lon': 75.86,
        'color': '#9b59b6',
    },
}

WEATHER_VARS = [
    'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
    'precipitation_sum', 'relative_humidity_2m_mean',
    'et0_fao_evapotranspiration', 'shortwave_radiation_sum'
]

PREDICTION_LENGTH = 12

# ============================================================
# WEATHER FETCHING
# ============================================================
def fetch_historical_weather(lat, lon, start_date, end_date):
    url = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude': lat, 'longitude': lon,
        'start_date': start_date, 'end_date': end_date,
        'daily': ','.join(WEATHER_VARS),
        'timezone': 'Asia/Kolkata',
    }
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            daily = resp.json()['daily']
            df = pd.DataFrame(daily)
            df['time'] = pd.to_datetime(df['time'])
            return df.set_index('time')
        except Exception as e:
            print(f"    Retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return None


def fetch_forecast_weather(lat, lon):
    url = 'https://api.open-meteo.com/v1/forecast'
    forecast_vars = [
        'temperature_2m_max', 'temperature_2m_min',
        'precipitation_sum', 'et0_fao_evapotranspiration', 'shortwave_radiation_sum'
    ]
    params = {
        'latitude': lat, 'longitude': lon,
        'daily': ','.join(forecast_vars),
        'timezone': 'Asia/Kolkata', 'forecast_days': 16,
    }
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            daily = resp.json()['daily']
            df = pd.DataFrame(daily)
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            df['temperature_2m_mean'] = (df['temperature_2m_max'] + df['temperature_2m_min']) / 2
            if 'relative_humidity_2m_mean' not in df.columns:
                df['relative_humidity_2m_mean'] = np.nan
            return df[WEATHER_VARS]
        except Exception as e:
            print(f"    Retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return None


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def engineer_features(daily_weather, weekly_dates):
    records = []
    for week_start in weekly_dates:
        week_end = week_start + pd.Timedelta(days=6)
        mask = (daily_weather.index >= week_start) & (daily_weather.index <= week_end)
        wd = daily_weather.loc[mask]
        if len(wd) < 3:
            records.append({})
            continue
        records.append({
            'temp_max': wd['temperature_2m_max'].max(),
            'temp_min': wd['temperature_2m_min'].min(),
            'temp_mean': wd['temperature_2m_mean'].mean(),
            'precip_sum': wd['precipitation_sum'].sum(),
            'humidity_avg': wd['relative_humidity_2m_mean'].mean(),
            'et0_sum': wd['et0_fao_evapotranspiration'].sum(),
            'radiation_sum': wd['shortwave_radiation_sum'].sum(),
            'temp_range': wd['temperature_2m_max'].max() - wd['temperature_2m_min'].min(),
            'heat_stress_days': int((wd['temperature_2m_max'] > 38).sum()),
            'heavy_rain_days': int((wd['precipitation_sum'] > 20).sum()),
            'dry_days': int((wd['precipitation_sum'] < 1).sum()),
        })

    features = pd.DataFrame(records, index=weekly_dates)
    features.index.name = 'week'

    iso_week = features.index.isocalendar().week.values
    for col, base in [('temp_anomaly', 'temp_mean'), ('precip_anomaly', 'precip_sum')]:
        clim = features.groupby(iso_week)[base].transform('mean')
        features[col] = features[base] - clim

    features['precip_lag1'] = features['precip_sum'].shift(1)
    features['precip_lag4'] = features['precip_sum'].shift(4)
    features['temp_lag1'] = features['temp_mean'].shift(1)
    features['temp_lag4'] = features['temp_mean'].shift(4)
    features['precip_roll4'] = features['precip_sum'].rolling(4).mean()
    features['temp_roll4'] = features['temp_mean'].rolling(4).mean()
    return features


# ============================================================
# CHRONOS
# ============================================================
def run_chronos(pipeline, series, pred_len=12):
    ctx = torch.tensor(series.values, dtype=torch.float32)
    forecast = pipeline.predict(inputs=ctx.unsqueeze(0), prediction_length=pred_len, num_samples=100, temperature=1.0)
    f = forecast.squeeze(0).numpy()
    return {
        'median': np.median(f, axis=0),
        'p10': np.percentile(f, 10, axis=0),
        'p25': np.percentile(f, 25, axis=0),
        'p75': np.percentile(f, 75, axis=0),
        'p90': np.percentile(f, 90, axis=0),
    }


# ============================================================
# WALK-FORWARD RESIDUAL MODEL
# ============================================================
def train_residual_model(weekly_df, weather_features, pipeline):
    merged = weekly_df.merge(weather_features, left_on='week', right_index=True, how='inner').dropna().reset_index(drop=True)
    feat_cols = [c for c in weather_features.columns if c in merged.columns]

    if len(merged) < 80:
        print(f"    Insufficient data ({len(merged)} rows), skipping weather correction")
        return None, feat_cols

    all_X, all_y = [], []
    min_train = max(50, len(merged) // 3)
    step = 12

    for cutoff in range(min_train, len(merged) - 6, step):
        test_end = min(cutoff + step, len(merged))
        pred_len = test_end - cutoff
        if pred_len < 4:
            continue
        chrono_f = run_chronos(pipeline, merged['avg_candy_rate'].iloc[:cutoff], pred_len)
        residuals = merged['avg_candy_rate'].iloc[cutoff:test_end].values - chrono_f['median']
        all_X.append(merged[feat_cols].iloc[cutoff:test_end].values)
        all_y.append(residuals)

    if not all_X:
        return None, feat_cols

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, loss='huber', random_state=42,
    )
    model.fit(X_all, y_all)
    print(f"    Residual model trained on {len(y_all)} samples")
    return model, feat_cols


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("STATE-WISE COTTON PRICE FORECAST (3 MONTHS)")
    print("Hybrid: Amazon Chronos + State-Specific Weather Regression")
    print("=" * 70)

    # Load purchase data
    df = pd.read_excel("Data/Purchase Data 7 year - Final.xls", sheet_name="Purchase Data", header=1)
    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
    df = df.sort_values('Purchase Date')
    df['week'] = df['Purchase Date'].dt.to_period('W').apply(lambda r: r.start_time)

    # Load Chronos once
    print("\nLoading Chronos model...")
    pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="cpu", dtype=torch.float32)

    all_forecasts = {}

    for state_name, cfg in STATES.items():
        print(f"\n{'='*60}")
        print(f"  {state_name.upper()}")
        print(f"{'='*60}")

        # Build weekly series
        sdf = df[df['State'] == cfg['data_name']]
        weekly = sdf.groupby('week').apply(
            lambda g: (g['Candy Rate'] * g['Bales']).sum() / g['Bales'].sum()
        ).reset_index()
        weekly.columns = ['week', 'avg_candy_rate']
        weekly = weekly.sort_values('week').reset_index(drop=True)

        # Fill gaps with interpolation for continuous series
        full_idx = pd.date_range(weekly['week'].min(), weekly['week'].max(), freq='W-MON')
        weekly = weekly.set_index('week').reindex(full_idx).interpolate(method='linear').reset_index()
        weekly.columns = ['week', 'avg_candy_rate']
        weekly = weekly.dropna()

        print(f"  Price series: {len(weekly)} weeks")
        print(f"  Last price: ₹{weekly['avg_candy_rate'].iloc[-1]:,.0f}")

        # Fetch state-specific weather
        cache_file = f"weather_cache_{state_name.lower().replace(' ', '_')}.csv"
        if os.path.exists(cache_file):
            print(f"  Loading cached weather from {cache_file}")
            daily_w = pd.read_csv(cache_file, parse_dates=['time'], index_col='time')
        else:
            start = (weekly['week'].min() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
            end = weekly['week'].max().strftime('%Y-%m-%d')
            print(f"  Fetching historical weather ({cfg['lat']}, {cfg['lon']})...")
            daily_w = fetch_historical_weather(cfg['lat'], cfg['lon'], start, end)
            if daily_w is not None:
                daily_w.to_csv(cache_file)
            else:
                print(f"  FAILED to fetch weather, using Chronos-only")
                daily_w = None

        # Chronos baseline
        print(f"  Running Chronos forecast...")
        chronos_result = run_chronos(pipeline, weekly['avg_candy_rate'], PREDICTION_LENGTH)

        # Weather correction
        residual_adj = np.zeros(PREDICTION_LENGTH)
        if daily_w is not None:
            weather_feats = engineer_features(daily_w, weekly['week'])
            weather_feats = weather_feats.dropna()
            print(f"  Weather features: {len(weather_feats)} weeks")

            print(f"  Training residual model...")
            model, feat_cols = train_residual_model(weekly, weather_feats, pipeline)

            if model is not None:
                # Forecast weather
                print(f"  Fetching forecast weather...")
                forecast_daily = fetch_forecast_weather(cfg['lat'], cfg['lon'])
                time.sleep(0.3)

                if forecast_daily is not None:
                    full_daily = pd.concat([daily_w, forecast_daily])
                    full_daily = full_daily[~full_daily.index.duplicated(keep='last')].sort_index()

                    last_date = weekly['week'].max()
                    future_weeks = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=PREDICTION_LENGTH, freq='W-MON')
                    all_weeks = list(weekly['week']) + list(future_weeks)
                    all_feats = engineer_features(full_daily, pd.DatetimeIndex(all_weeks))
                    forecast_feats = all_feats.loc[future_weeks]

                    for col in feat_cols:
                        if col in forecast_feats.columns:
                            forecast_feats[col] = forecast_feats[col].fillna(weather_feats[col].median() if col in weather_feats.columns else 0)

                    residual_adj = model.predict(forecast_feats[feat_cols].values)

        # Build forecast
        last_date = weekly['week'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=PREDICTION_LENGTH, freq='W-MON')

        forecast_df = pd.DataFrame({
            'week': future_dates,
            'state': state_name,
            'chronos_median': chronos_result['median'],
            'weather_adjustment': residual_adj,
            'predicted_p10': chronos_result['p10'] + residual_adj,
            'predicted_p25': chronos_result['p25'] + residual_adj,
            'predicted_median': chronos_result['median'] + residual_adj,
            'predicted_p75': chronos_result['p75'] + residual_adj,
            'predicted_p90': chronos_result['p90'] + residual_adj,
        })

        all_forecasts[state_name] = {
            'historical': weekly,
            'forecast': forecast_df,
            'chronos': chronos_result,
        }

        print(f"\n  Forecast (median): ₹{forecast_df['predicted_median'].iloc[0]:,.0f} → ₹{forecast_df['predicted_median'].iloc[-1]:,.0f}")
        print(f"  Weather adjustment avg: ₹{residual_adj.mean():+,.0f}")

    # ============================================================
    # SAVE COMBINED FORECAST CSV
    # ============================================================
    combined = pd.concat([v['forecast'] for v in all_forecasts.values()], ignore_index=True)
    combined.to_csv('state_forecast_3months.csv', index=False)
    print(f"\n\nSaved state_forecast_3months.csv ({len(combined)} rows)")

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("\n" + "=" * 90)
    print(f"{'STATE':<20} {'CURRENT':>12} {'WEEK 1':>12} {'WEEK 6':>12} {'WEEK 12':>12} {'TREND':>12}")
    print("-" * 90)
    for state, data in all_forecasts.items():
        current = data['historical']['avg_candy_rate'].iloc[-1]
        w1 = data['forecast']['predicted_median'].iloc[0]
        w6 = data['forecast']['predicted_median'].iloc[5]
        w12 = data['forecast']['predicted_median'].iloc[-1]
        chg = ((w12 - current) / current) * 100
        trend = "▼" if chg < -1 else ("▲" if chg > 1 else "→")
        print(f"{state:<20} ₹{current:>10,.0f} ₹{w1:>10,.0f} ₹{w6:>10,.0f} ₹{w12:>10,.0f} {trend} {chg:>+.1f}%")
    print("=" * 90)

    # ============================================================
    # VISUALIZATION: ALL STATES ON ONE CHART
    # ============================================================
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    axes = axes.flatten()

    for i, (state, data) in enumerate(all_forecasts.items()):
        ax = axes[i]
        hist = data['historical'].tail(52)
        fc = data['forecast']
        color = STATES[state]['color']

        ax.plot(hist['week'], hist['avg_candy_rate'], color='#2c3e50', linewidth=1.8, label='Historical')
        ax.plot(fc['week'], fc['predicted_median'], color=color, linewidth=2.5, marker='o', markersize=4, label='Hybrid Forecast')
        ax.fill_between(fc['week'], fc['predicted_p25'], fc['predicted_p75'], alpha=0.25, color=color)
        ax.fill_between(fc['week'], fc['predicted_p10'], fc['predicted_p90'], alpha=0.1, color=color)
        ax.plot(fc['week'], fc['chronos_median'], color='#bdc3c7', linewidth=1.2, linestyle='--', label='Chronos Baseline')

        current = hist['avg_candy_rate'].iloc[-1]
        end = fc['predicted_median'].iloc[-1]
        chg = ((end - current) / current) * 100

        ax.set_title(f"{state}  (Current: ₹{current:,.0f} → ₹{end:,.0f}, {chg:+.1f}%)",
                     fontsize=12, fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    # Summary panel
    ax_summary = axes[5]
    ax_summary.axis('off')
    summary_text = "STATE-WISE 12-WEEK FORECAST SUMMARY\n" + "=" * 40 + "\n\n"
    for state, data in all_forecasts.items():
        current = data['historical']['avg_candy_rate'].iloc[-1]
        end = data['forecast']['predicted_median'].iloc[-1]
        chg = ((end - current) / current) * 100
        adj = data['forecast']['weather_adjustment'].mean()
        arrow = "↓" if chg < -1 else ("↑" if chg > 1 else "→")
        summary_text += f"{state:<18} {arrow} {chg:+.1f}%\n"
        summary_text += f"  Current: ₹{current:,.0f}  →  ₹{end:,.0f}\n"
        summary_text += f"  Weather adj: ₹{adj:+,.0f}/candy\n\n"

    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    fig.suptitle('Vardhaman Cotton Procurement — State-Wise 3-Month Price Forecast\n'
                 '(Hybrid: Chronos Zero-Shot + State-Specific Weather Regression)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('state_price_forecast.png', dpi=150, bbox_inches='tight')
    print("Saved state_price_forecast.png")

    # ============================================================
    # COMPARISON CHART: ALL STATES OVERLAID
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(16, 7))
    for state, data in all_forecasts.items():
        fc = data['forecast']
        color = STATES[state]['color']
        ax2.plot(fc['week'], fc['predicted_median'], linewidth=2.5, marker='o',
                 markersize=5, label=state, color=color)
        ax2.fill_between(fc['week'], fc['predicted_p25'], fc['predicted_p75'], alpha=0.15, color=color)

    ax2.set_title('State-Wise Cotton Price Forecast Comparison (Next 12 Weeks)',
                  fontsize=15, fontweight='bold')
    ax2.set_xlabel('Week', fontsize=12)
    ax2.set_ylabel('Candy Rate (Rs/candy)', fontsize=12)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('state_comparison_forecast.png', dpi=150, bbox_inches='tight')
    print("Saved state_comparison_forecast.png")

    print("\n" + "=" * 70)
    print("DONE — State-wise forecast complete!")
    print("  state_forecast_3months.csv")
    print("  state_price_forecast.png")
    print("  state_comparison_forecast.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
