import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="Vardhaman Cotton Price Forecast",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 0.5rem 0 0.2rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        text-align: center;
        padding-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.4rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    weekly = pd.read_csv('weekly_candy_rate.csv', parse_dates=['week'])

    hybrid = None
    if os.path.exists('forecast_3months_hybrid.csv'):
        hybrid = pd.read_csv('forecast_3months_hybrid.csv', parse_dates=['week'])

    chronos = None
    if os.path.exists('forecast_3months.csv'):
        chronos = pd.read_csv('forecast_3months.csv', parse_dates=['week'])

    purchase = None
    purchase_path = 'Data/Purchase Data 7 year - Final.xls'
    if os.path.exists(purchase_path):
        purchase = pd.read_excel(purchase_path, sheet_name='Purchase Data', header=1)
        purchase['Purchase Date'] = pd.to_datetime(purchase['Purchase Date'])

    state_forecast = None
    if os.path.exists('state_forecast_3months.csv'):
        state_forecast = pd.read_csv('state_forecast_3months.csv', parse_dates=['week'])

    return weekly, hybrid, chronos, purchase, state_forecast


weekly, hybrid, chronos, purchase, state_forecast = load_data()

STATE_COLORS = {
    'Maharashtra': '#e74c3c',
    'Gujarat': '#3498db',
    'Haryana': '#27ae60',
    'Rajasthan': '#f39c12',
    'Madhya Pradesh': '#9b59b6',
}

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header">Vardhaman Cotton Procurement Price Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hybrid Model: Amazon Chronos Zero-Shot + Weather Regression | 12-Week Forecast</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cotton.png", width=60)
    st.markdown("### Controls")

    history_weeks = st.slider("Historical weeks to display", 12, len(weekly), 52, step=4)

    show_chronos_baseline = st.checkbox("Show Chronos-only baseline", value=True)
    show_confidence = st.checkbox("Show confidence intervals", value=True)
    confidence_level = st.radio("Confidence band", ["50% (p25-p75)", "80% (p10-p90)"], index=1)

    st.markdown("---")
    st.markdown("### Data Summary")
    st.markdown(f"**Price data:** {len(weekly)} weeks")
    st.markdown(f"**Range:** {weekly['week'].min().strftime('%b %Y')} – {weekly['week'].max().strftime('%b %Y')}")
    if purchase is not None:
        st.markdown(f"**Transactions:** {len(purchase):,}")
        st.markdown(f"**Cotton varieties:** {purchase['Item Description'].nunique()}")
        st.markdown(f"**States:** {purchase['State'].nunique()}")

    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown("**Base model:** Chronos-T5-Small")
    st.markdown("**Correction:** GradientBoosting on weather residuals")
    st.markdown("**Weather regions:** 5 cotton belt stations")
    st.markdown("**Features:** 19 weekly weather features")

# ============================================================
# TOP METRICS
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)

current_price = weekly['avg_candy_rate'].iloc[-1]
prev_price = weekly['avg_candy_rate'].iloc[-2]
price_change = current_price - prev_price
pct_change = (price_change / prev_price) * 100

with col1:
    st.metric("Current Rate (Rs/candy)", f"₹{current_price:,.0f}", f"{pct_change:+.1f}%")

if hybrid is not None:
    forecast_median_next = hybrid['predicted_candy_rate_median'].iloc[0]
    forecast_median_end = hybrid['predicted_candy_rate_median'].iloc[-1]
    total_adj = hybrid['weather_adjustment'].mean()

    with col2:
        delta_next = forecast_median_next - current_price
        st.metric("Next Week Forecast", f"₹{forecast_median_next:,.0f}", f"{delta_next:+,.0f}")
    with col3:
        delta_end = forecast_median_end - current_price
        st.metric("12-Week Forecast", f"₹{forecast_median_end:,.0f}", f"{delta_end:+,.0f}")
    with col4:
        st.metric("Avg Weather Adjustment", f"₹{total_adj:+,.0f}", "downward" if total_adj < 0 else "upward")
    with col5:
        forecast_range = hybrid['predicted_candy_rate_p90'].max() - hybrid['predicted_candy_rate_p10'].min()
        st.metric("Forecast Range (80%)", f"₹{forecast_range:,.0f}")

# ============================================================
# MAIN FORECAST CHART
# ============================================================
st.markdown('<div class="section-title">Price Forecast</div>', unsafe_allow_html=True)

fig = go.Figure()

hist_window = weekly.tail(history_weeks)
fig.add_trace(go.Scatter(
    x=hist_window['week'], y=hist_window['avg_candy_rate'],
    mode='lines', name='Historical (Weekly Avg)',
    line=dict(color='#2c3e50', width=2.5),
))

if hybrid is not None:
    if show_confidence:
        if "80%" in confidence_level:
            fig.add_trace(go.Scatter(
                x=pd.concat([hybrid['week'], hybrid['week'][::-1]]),
                y=pd.concat([hybrid['predicted_candy_rate_p90'], hybrid['predicted_candy_rate_p10'][::-1]]),
                fill='toself', fillcolor='rgba(231, 76, 60, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='80% Confidence', showlegend=True,
            ))
        fig.add_trace(go.Scatter(
            x=pd.concat([hybrid['week'], hybrid['week'][::-1]]),
            y=pd.concat([hybrid['predicted_candy_rate_p75'], hybrid['predicted_candy_rate_p25'][::-1]]),
            fill='toself', fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='50% Confidence', showlegend=True,
        ))

    fig.add_trace(go.Scatter(
        x=hybrid['week'], y=hybrid['predicted_candy_rate_median'],
        mode='lines+markers', name='Hybrid Forecast (Median)',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=7, symbol='circle'),
    ))

if show_chronos_baseline and chronos is not None:
    fig.add_trace(go.Scatter(
        x=chronos['week'] if 'week' in chronos.columns else hybrid['week'],
        y=chronos['predicted_candy_rate_median'],
        mode='lines', name='Chronos-Only Baseline',
        line=dict(color='#95a5a6', width=2, dash='dash'),
    ))

fig.update_layout(
    height=500,
    xaxis_title='Week',
    yaxis_title='Candy Rate (Rs/candy)',
    yaxis_tickformat=',',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    hovermode='x unified',
    template='plotly_white',
    margin=dict(l=60, r=20, t=30, b=60),
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# FORECAST TABLE
# ============================================================
st.markdown('<div class="section-title">Forecast Table</div>', unsafe_allow_html=True)

if hybrid is not None:
    display_df = hybrid.copy()
    display_df['week'] = display_df['week'].dt.strftime('%d %b %Y')
    display_df.columns = [
        'Week', 'Chronos Median', 'Weather Adjustment',
        'Hybrid P10', 'Hybrid P25', 'Hybrid Median', 'Hybrid P75', 'Hybrid P90'
    ]

    for col in display_df.columns[1:]:
        display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.0f}")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=460,
    )

    csv = hybrid.to_csv(index=False)
    st.download_button("Download Forecast CSV", csv, "forecast_3months_hybrid.csv", "text/csv")

elif chronos is not None:
    display_df = chronos.copy()
    display_df['week'] = display_df['week'].dt.strftime('%d %b %Y')
    for col in display_df.columns[1:]:
        display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ============================================================
# STATE-WISE FORECAST
# ============================================================
if state_forecast is not None:
    st.markdown('<div class="section-title">State-Wise 3-Month Price Forecast</div>', unsafe_allow_html=True)

    states_list = state_forecast['state'].unique().tolist()

    # Summary metrics row
    st_cols = st.columns(len(states_list))
    for i, state in enumerate(states_list):
        sdf = state_forecast[state_forecast['state'] == state]
        first_price = sdf['predicted_median'].iloc[0]
        last_price = sdf['predicted_median'].iloc[-1]
        chg = ((last_price - first_price) / first_price) * 100
        color = STATE_COLORS.get(state, '#666')
        with st_cols[i]:
            st.markdown(
                f'<div style="background:{color}; padding:12px; border-radius:10px; color:white; text-align:center;">'
                f'<div style="font-size:0.85rem; opacity:0.9;">{state}</div>'
                f'<div style="font-size:1.4rem; font-weight:700;">₹{last_price:,.0f}</div>'
                f'<div style="font-size:0.8rem;">{chg:+.1f}% over 12 wks</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    tab_chart, tab_table = st.tabs(["Comparison Chart", "Forecast Table"])

    with tab_chart:
        # Overlay all states
        fig_states = go.Figure()
        for state in states_list:
            sdf = state_forecast[state_forecast['state'] == state]
            color = STATE_COLORS.get(state, '#666')

            fig_states.add_trace(go.Scatter(
                x=pd.concat([sdf['week'], sdf['week'][::-1]]),
                y=pd.concat([sdf['predicted_p75'], sdf['predicted_p25'][::-1]]),
                fill='toself', fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{state} (50% CI)', showlegend=False,
            ))
            fig_states.add_trace(go.Scatter(
                x=sdf['week'], y=sdf['predicted_median'],
                mode='lines+markers', name=state,
                line=dict(color=color, width=2.5),
                marker=dict(size=6),
            ))

        fig_states.update_layout(
            title='State-Wise Cotton Price Forecast — Next 12 Weeks',
            xaxis_title='Week', yaxis_title='Candy Rate (Rs/candy)',
            yaxis_tickformat=',',
            height=500, template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified',
        )
        st.plotly_chart(fig_states, use_container_width=True)

    with tab_table:
        # Pivot table: weeks as rows, states as columns
        pivot = state_forecast.pivot(index='week', columns='state', values='predicted_median')
        pivot.index = pivot.index.strftime('%d %b %Y')
        pivot.index.name = 'Week'

        # Add weather adjustment row
        adj_pivot = state_forecast.pivot(index='week', columns='state', values='weather_adjustment')
        adj_pivot.index = adj_pivot.index.strftime('%d %b %Y')

        st.markdown("**Predicted Median Candy Rate (Rs/candy)**")
        display_pivot = pivot.copy()
        for col in display_pivot.columns:
            display_pivot[col] = display_pivot[col].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(display_pivot, use_container_width=True, height=460)

        st.markdown("**Weather Adjustments (Rs/candy)**")
        display_adj = adj_pivot.copy()
        for col in display_adj.columns:
            display_adj[col] = display_adj[col].apply(lambda x: f"₹{x:+,.0f}")
        st.dataframe(display_adj, use_container_width=True, height=460)

        csv_state = state_forecast.to_csv(index=False)
        st.download_button("Download State Forecast CSV", csv_state, "state_forecast_3months.csv", "text/csv")

# ============================================================
# WEATHER IMPACT ANALYSIS
# ============================================================
if hybrid is not None:
    st.markdown('<div class="section-title">Weather Impact Analysis</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        adj = hybrid[['week', 'weather_adjustment']].copy()
        colors = ['#27ae60' if v > 0 else '#e74c3c' for v in adj['weather_adjustment']]

        fig_adj = go.Figure(go.Bar(
            x=adj['week'].dt.strftime('%b %d'),
            y=adj['weather_adjustment'],
            marker_color=colors,
            text=[f"₹{v:+,.0f}" for v in adj['weather_adjustment']],
            textposition='outside',
        ))
        fig_adj.update_layout(
            title='Weekly Weather Price Adjustments',
            xaxis_title='Forecast Week',
            yaxis_title='Adjustment (Rs/candy)',
            yaxis_tickformat=',',
            height=400,
            template='plotly_white',
            margin=dict(t=40),
        )
        st.plotly_chart(fig_adj, use_container_width=True)

    with col_right:
        feature_importance = {
            'temp_lag4': 0.1476, 'precip_roll4': 0.1025, 'temp_lag1': 0.0884,
            'humidity_avg': 0.0828, 'radiation_sum': 0.0807, 'temp_roll4': 0.0800,
            'temp_anomaly': 0.0771, 'precip_lag4': 0.0722, 'temp_range': 0.0630,
            'temp_min': 0.0394, 'precip_anomaly': 0.0310, 'precip_sum': 0.0306,
            'precip_lag1': 0.0288, 'et0_sum': 0.0270, 'temp_max': 0.0160,
        }
        fi_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
        fi_df = fi_df.sort_values('Importance', ascending=True)

        fig_fi = go.Figure(go.Bar(
            x=fi_df['Importance'], y=fi_df['Feature'],
            orientation='h',
            marker_color=fi_df['Importance'].apply(
                lambda x: f'rgba(102, 126, 234, {0.4 + 0.6 * x / fi_df["Importance"].max()})'
            ),
            text=[f"{v:.1%}" for v in fi_df['Importance']],
            textposition='outside',
        ))
        fig_fi.update_layout(
            title='Weather Feature Importance',
            xaxis_title='Importance Score',
            height=400,
            template='plotly_white',
            margin=dict(t=40, l=120),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

# ============================================================
# HISTORICAL ANALYSIS
# ============================================================
if purchase is not None:
    st.markdown('<div class="section-title">Historical Purchase Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Price Trend", "State Breakdown", "Cotton Varieties"])

    with tab1:
        monthly = purchase.set_index('Purchase Date').resample('M').apply(
            lambda g: np.average(g['Candy Rate'], weights=g['Bales']) if len(g) > 0 else np.nan
        ).dropna()

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=monthly.index, y=monthly.values,
            mode='lines', name='Monthly Avg Candy Rate',
            line=dict(color='#3498db', width=2),
            fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.1)',
        ))
        fig_hist.update_layout(
            title='Monthly Weighted-Average Candy Rate (Full History)',
            xaxis_title='Month', yaxis_title='Rs/candy', yaxis_tickformat=',',
            height=400, template='plotly_white',
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        state_data = purchase.groupby('State').agg(
            total_bales=('Bales', 'sum'),
            avg_rate=('Candy Rate', 'mean'),
            transactions=('Bales', 'count'),
        ).sort_values('total_bales', ascending=False)

        col_a, col_b = st.columns(2)

        with col_a:
            fig_state = go.Figure(go.Pie(
                labels=state_data.index,
                values=state_data['total_bales'],
                hole=0.45,
                textinfo='label+percent',
                marker_colors=['#667eea', '#764ba2', '#f093fb', '#4facfe',
                               '#00f2fe', '#43e97b', '#fa709a', '#fee140',
                               '#a18cd1', '#fbc2eb', '#8fd3f4', '#d4fc79'],
            ))
            fig_state.update_layout(title='Procurement by State (Bales)', height=400)
            st.plotly_chart(fig_state, use_container_width=True)

        with col_b:
            state_display = state_data.copy()
            state_display['total_bales'] = state_display['total_bales'].apply(lambda x: f"{x:,.0f}")
            state_display['avg_rate'] = state_display['avg_rate'].apply(lambda x: f"₹{x:,.0f}")
            state_display['transactions'] = state_display['transactions'].apply(lambda x: f"{x:,}")
            state_display.columns = ['Total Bales', 'Avg Rate', 'Transactions']
            st.dataframe(state_display, use_container_width=True, height=400)

    with tab3:
        top_items = purchase['Item Description'].value_counts().head(10)
        fig_items = go.Figure(go.Bar(
            x=top_items.values,
            y=top_items.index,
            orientation='h',
            marker_color='#667eea',
            text=[f"{v:,}" for v in top_items.values],
            textposition='outside',
        ))
        fig_items.update_layout(
            title='Top 10 Cotton Varieties (by Transaction Count)',
            xaxis_title='Number of Transactions',
            height=450, template='plotly_white',
            margin=dict(l=300),
        )
        st.plotly_chart(fig_items, use_container_width=True)

# ============================================================
# WEATHER STATIONS MAP
# ============================================================
st.markdown('<div class="section-title">Weather Station Network</div>', unsafe_allow_html=True)

stations = pd.DataFrame({
    'lat': [21.15, 22.30, 29.15, 29.91, 22.72],
    'lon': [79.09, 70.78, 75.72, 73.88, 75.86],
    'region': ['Vidarbha, Maharashtra', 'Saurashtra, Gujarat', 'Hisar, Haryana',
               'Sri Ganganagar, Rajasthan', 'Malwa, Madhya Pradesh'],
    'weight': ['39.6%', '30.8%', '11.0%', '9.9%', '8.8%'],
})

fig_map = go.Figure(go.Scattermapbox(
    lat=stations['lat'], lon=stations['lon'],
    mode='markers+text',
    marker=dict(size=[30, 25, 18, 17, 16], color='#e74c3c', opacity=0.8),
    text=stations['region'] + '<br>Weight: ' + stations['weight'],
    textposition='top right',
    hoverinfo='text',
))
fig_map.update_layout(
    mapbox=dict(style='open-street-map', center=dict(lat=25.5, lon=75), zoom=4.3),
    height=500,
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_map, use_container_width=True)

# ============================================================
# METHODOLOGY
# ============================================================
with st.expander("Methodology & Technical Details"):
    st.markdown("""
    ### Hybrid Forecasting Approach

    **Stage 1 — Chronos Zero-Shot Baseline**
    Amazon Chronos-T5-Small generates a probabilistic 12-week forecast purely from price history.
    The model was pretrained on millions of diverse time series and requires no task-specific training.

    **Stage 2 — Weather Correction**
    A GradientBoostingRegressor learns Chronos's prediction errors (residuals) from 19 weather features
    covering temperature, rainfall, humidity, radiation, and their lags/anomalies across 5 cotton-growing
    regions weighted by Vardhaman's actual procurement share.

    **Stage 3 — Walk-Forward Validation**
    The residual model is trained using 14 walk-forward folds (168 samples), preventing data leakage
    and simulating real-time forecasting conditions.

    **Final Output:**
    `Predicted Price = Chronos Baseline + Weather Adjustment`

    ---
    **Unit:** 1 Candy = 355.62 kg of raw cotton. Rs/candy is the standard Indian cotton trading unit.
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#aaa; font-size:0.85rem;">'
    'Vardhaman Cotton Price Forecast Dashboard | '
    'Built with Streamlit | '
    'Model: Amazon Chronos + Weather Regression'
    '</div>',
    unsafe_allow_html=True,
)
