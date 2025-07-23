
def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ===== RSI محاسبه دقیق با EMA =====
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ===== بارگذاری داده‌ها =====
df = pd.read_csv("crypto_prices.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
df = df.dropna(axis=1, how='all')

# ===== تشخیص کوین‌ها =====
price_cols = [col for col in df.columns if col not in ['timestamp'] and not col.startswith('volume_')]
volume_cols = {coin: f"volume_{coin}" for coin in price_cols if f"volume_{coin}" in df.columns}
coins = price_cols

# ===== محاسبات تغییرات، اهداف آینده و RSI =====
future_steps = {'5min': 1, '10min': 2, '30min': 6, '1h': 12}

for coin in coins:
    df[f'{coin}_change'] = df[coin].pct_change() * 100
    df[f'{coin}_rsi'] = compute_rsi(df[coin])
    for label, step in future_steps.items():
        df[f'{coin}_future_{label}'] = (df[coin].shift(-step) - df[coin]) / df[coin] * 100

# ===== حذف داده‌های ناقص و ذخیره =====
df = df.dropna().reset_index(drop=True)

final_cols = ['timestamp']
for coin in coins:
    final_cols.append(f'{coin}_change')
    final_cols.append(f'{coin}_rsi')
    if coin in volume_cols:
        final_cols.append(volume_cols[coin])
    for label in future_steps:
        final_cols.append(f'{coin}_future_{label}')

df_final = df[final_cols]
df_final.to_csv("prepared_data_multistep.csv", index=False)
print("✅ داده نهایی ذخیره شد: prepared_data_multistep.csv")

# ===== نمودار تغییرات کوین =====
charts_dir = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(charts_dir, exist_ok=True)

for coin in coins:
    change_col = f'{coin}_change'
    if change_col in df.columns and df[change_col].dropna().shape[0] >= 2:
        plt.figure(figsize=(12, 5))
        plt.plot(df['timestamp'], df[change_col], label=f'{coin.capitalize()} % change', color='dodgerblue')

        last_time = df['timestamp'].iloc[-1]
        last_change = df[change_col].iloc[-1]
        last_price = df[coin].iloc[-1]
        plt.text(last_time, last_change, f"${last_price:.2f}", fontsize=10, color='green', ha='right')

        plt.title(f'{coin.capitalize()} 1h Change Chart')
        plt.xlabel('Timestamp')
        plt.ylabel('% Change')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=5))

        chart_path = os.path.join(charts_dir, f"{coin}_chart.png")
        plt.savefig(chart_path, dpi=150)
        print(f"✅ نمودار ذخیره شد: {chart_path}")
        plt.close()
