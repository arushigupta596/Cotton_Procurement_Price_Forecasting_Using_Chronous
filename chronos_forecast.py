import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

weekly = pd.read_csv("weekly_candy_rate.csv", parse_dates=['week'])
context = torch.tensor(weekly['avg_candy_rate'].values, dtype=torch.float32)

print("Loading Chronos model (chronos-t5-small)...")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    dtype=torch.float32,
)

prediction_length = 12
print(f"Forecasting {prediction_length} weeks ahead...")

forecast = pipeline.predict(
    inputs=context.unsqueeze(0),
    prediction_length=prediction_length,
    num_samples=100,
    temperature=1.0,
)

forecast_np = forecast.squeeze(0).numpy()
median = np.median(forecast_np, axis=0)
p10 = np.percentile(forecast_np, 10, axis=0)
p25 = np.percentile(forecast_np, 25, axis=0)
p75 = np.percentile(forecast_np, 75, axis=0)
p90 = np.percentile(forecast_np, 90, axis=0)

last_date = weekly['week'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=prediction_length, freq='W-MON')

forecast_df = pd.DataFrame({
    'week': future_dates,
    'predicted_candy_rate_p10': p10,
    'predicted_candy_rate_p25': p25,
    'predicted_candy_rate_median': median,
    'predicted_candy_rate_p75': p75,
    'predicted_candy_rate_p90': p90,
})
forecast_df.to_csv("forecast_3months.csv", index=False)
print("\nForecast saved to forecast_3months.csv")
print(forecast_df.to_string(index=False))

# ---- PLOT ----
fig, ax = plt.subplots(figsize=(16, 7))
hist_window = weekly.tail(52)
ax.plot(hist_window['week'], hist_window['avg_candy_rate'], color='#2c3e50', linewidth=2, label='Historical (Weekly Avg)')
ax.plot(future_dates, median, color='#e74c3c', linewidth=2.5, label='Forecast (Median)', marker='o', markersize=5)
ax.fill_between(future_dates, p25, p75, alpha=0.3, color='#e74c3c', label='50% Confidence Interval')
ax.fill_between(future_dates, p10, p90, alpha=0.15, color='#e74c3c', label='80% Confidence Interval')

ax.set_title('Vardhaman Cotton Procurement Price — 3-Month Forecast\n(Zero-Shot: Amazon Chronos-T5-Small)', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Week', fontsize=12)
ax.set_ylabel('Candy Rate (Rs/candy)', fontsize=12)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
plt.tight_layout()
plt.savefig('cotton_price_forecast.png', dpi=150, bbox_inches='tight')
print("\nChart saved to cotton_price_forecast.png")
