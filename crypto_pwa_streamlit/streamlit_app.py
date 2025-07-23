import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ✅ بهینه‌سازی برای موبایل
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-size: 16px;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .stPlotlyChart, .stChart {
            height: auto !important;
            max-height: 320px;
        }
    </style>
""", unsafe_allow_html=True)

import os

# === Load Data ===
@st.cache_data
def load_data():
    path = "predictions/prediction_summary_multistep.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

df = load_data()

if df.empty:
    st.error("❌ No prediction data found. Please run the pipeline first.")
    st.stop()

# === Sidebar ===
st.sidebar.title("🔎 Filters")
selected_coin = st.sidebar.selectbox("Select Coin", sorted(df['coin'].unique()))
selected_interval = st.sidebar.selectbox("Prediction Interval", ['5min', '10min', '30min', '1h'])

# === Filter Data ===
df_coin = df[df['coin'] == selected_coin].copy()
df_coin['timestamp'] = pd.to_datetime(df_coin['timestamp'])

# === Title ===
st.title("📈 Crypto Multi-step Prediction Dashboard")
st.markdown(f"**Showing results for `{selected_coin}` | Interval: `{selected_interval}`**")


# === Plot 1: Actual Price & Predicted Change with Decisions ===
st.subheader("📊 Price & Predicted Change with Decisions")
fig, ax1 = plt.subplots()

# Actual price
ax1.plot(df_coin['timestamp'], df_coin['actual_price'], label='Actual Price', color='blue')
ax1.set_ylabel("Price (USD)", color='blue')
ax1.tick_params(axis='x', rotation=45)

# Predicted change
ax2 = ax1.twinx()
ax2.plot(df_coin['timestamp'], df_coin[f'pred_{selected_interval}'], label='Predicted Change (%)', color='green')
ax2.set_ylabel("Predicted Change (%)", color='green')

# نقاط خرید/فروش
buy_signals = df_coin[df_coin['decision'] == 'Buy']
sell_signals = df_coin[df_coin['decision'] == 'Sell']

ax1.scatter(buy_signals['timestamp'], buy_signals['actual_price'], marker='^', color='lime', label='Buy', s=100, zorder=5)
ax1.scatter(sell_signals['timestamp'], sell_signals['actual_price'], marker='v', color='red', label='Sell', s=100, zorder=5)

fig.tight_layout()
ax1.legend(loc='upper left')
st.pyplot(fig)
# === Plot 2: RSI (if available) ===
rsi_col = f"{selected_coin}_rsi"
if rsi_col in df_coin.columns:
    st.subheader("📉 RSI Indicator")
    plt.figure(figsize=(10, 2.5))
    plt.plot(df_coin['timestamp'], df_coin[rsi_col], color='orange')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title("Relative Strength Index (RSI)")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.close()

# === Plot 3: Volume (if available) ===
volume_col = f"volume_{selected_coin}"
if volume_col in df_coin.columns:
    st.subheader("📦 Trading Volume")
    plt.figure(figsize=(10, 2.5))
    plt.bar(df_coin['timestamp'], df_coin[volume_col], width=0.01, color='gray')
    plt.ylabel("Volume")
    plt.title("Trading Volume")
    st.pyplot(plt.gcf())
    plt.close()

# === Decision Section ===
# === Profit Simulation Based on Real Decisions ===
st.subheader("💸 Simulated Profit Based on Decisions")

df_coin = df_coin.copy()
df_coin = df_coin.sort_values('timestamp')

position = None
entry_price = 0
profits = []

for _, row in df_coin.iterrows():
    decision = row['decision']
    price = row['actual_price']

    if position is None and decision == 'Buy':
        position = 'long'
        entry_price = price
        profits.append(1)
    elif position == 'long' and decision == 'Sell':
        profit = price / entry_price
        profits.append(profit)
        position = None
        entry_price = 0
    else:
        if profits:
            profits.append(profits[-1])
        else:
            profits.append(1)

df_coin['real_cumulative_return'] = profits

plt.figure(figsize=(10, 3))
plt.plot(df_coin['timestamp'], df_coin['real_cumulative_return'], color='orange')
plt.title("Realistic Cumulative Return (based on Buy/Sell decisions)")
plt.ylabel("Return (x)")
plt.grid(True)
plt.xticks(rotation=45)
st.pyplot(plt.gcf())
plt.close()

st.subheader("🧠 Suggested Action")

latest = df_coin.iloc[-1]
actual_price = latest['actual_price']
predicted_change = latest[f'pred_{selected_interval}']
target_price = actual_price * (1 + predicted_change / 100)

rsi = latest[rsi_col] if rsi_col in latest else None

if predicted_change > 1:
    if rsi is not None and rsi > 70:
        st.warning(f"🔼 Buy signal, but RSI={rsi:.2f} indicates overbought. Be cautious.")
    else:
        st.success(f"**Buy Signal**\n- 💵 Price: **${actual_price:,.2f}**\n- 🎯 Target: **${target_price:,.2f}**\n- Change: **{predicted_change:.2f}%**")
elif predicted_change < -1:
    if rsi is not None and rsi < 30:
        st.warning(f"🔽 Sell signal, but RSI={rsi:.2f} indicates oversold. Be cautious.")
    else:
        st.error(f"**Sell Signal**\n- 💵 Price: **${actual_price:,.2f}**\n- Expected Drop: **${target_price:,.2f}**\n- Change: **{predicted_change:.2f}%**")
else:
    st.info(f"**Hold**\n- Change: **{predicted_change:.2f}%**\n- No strong trend detected.")

# === Table: Latest Predictions ===
st.subheader("📋 Latest Predictions")
st.dataframe(df_coin.tail(10)[['timestamp', 'actual_price', f'pred_{selected_interval}', 'macd', 'macd_signal', 'decision']])

# === Model Evaluation ===
st.subheader("📐 Model Evaluation")
eval_path = f"evaluations/{selected_coin}_evaluation.txt"
if os.path.exists(eval_path):
    with open(eval_path, 'r') as f:
        st.code(f.read())
else:
    st.warning("⚠️ Evaluation metrics not found for this coin.")
