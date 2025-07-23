import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import subprocess

# تنظیمات
SEQUENCE_LENGTH = 12
FUTURE_STEPS = ["5min", "10min", "30min", "1h"]
DATA_PATH = "prepared_data_multistep.csv"
PRICE_PATH = "crypto_prices.csv"
MODEL_DIR = "models_multistep"
PRED_DIR = "predictions"
SUMMARY_PATH = f"{PRED_DIR}/prediction_summary_multistep.csv"
DASHBOARD_PATH = "dashboard.py"

# خروجی
os.makedirs(PRED_DIR, exist_ok=True)

# بارگذاری داده‌ها
df = pd.read_csv(DATA_PATH)
df_price = pd.read_csv(PRICE_PATH)
df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
df_price = df_price.sort_values('timestamp')

# استخراج لیست کوین‌ها
coins = sorted(col.replace("_change", "") for col in df.columns if col.endswith("_change"))

# ✅ محاسبه RSI واقعی
def compute_rsi(prices, window=14):
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi

results = []

for coin in coins:
    try:
        print(f"\n🔮 Predicting for {coin}...")

        change_col = f"{coin}_change"
        if change_col not in df.columns:
            print(f"⚠️ Missing column: {change_col}")
            continue

        model_path = f"{MODEL_DIR}/{coin}_multistep.h5"
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found: {model_path}")
            continue

        coin_data = df[[change_col]].dropna()
        if len(coin_data) < SEQUENCE_LENGTH:
            print(f"⚠️ Not enough data for {coin}")
            continue

        # نرمال‌سازی
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(coin_data)
        X_input = scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)

        # مدل و پیش‌بینی
        model = load_model(model_path)
        prediction_scaled = model.predict(X_input)[0]
        pred_original = scaler.inverse_transform(prediction_scaled.reshape(1, -1))[0]

        actual_price = df_price[coin].iloc[-1]
        timestamp = df_price['timestamp'].iloc[-1]

        # محاسبه RSI
        rsi_series = compute_rsi(df_price[coin])
        current_rsi = round(rsi_series.iloc[-1], 2)

        # ساخت خروجی
        
    # دریافت آخرین مقادیر MACD
    macd_col = f"{coin}_macd"
    signal_col = f"{coin}_macd_signal"
    if macd_col in df.columns and signal_col in df.columns:
        macd_val = df[macd_col].iloc[-1]
        signal_val = df[signal_col].iloc[-1]
    else:
        macd_val = signal_val = None

    # تصمیم‌گیری هوشمند
    decision = "Hold"
    if predicted_change > 1 and current_rsi < 70 and macd_val is not None and macd_val > signal_val:
        decision = "Buy"
    elif predicted_change < -1 and current_rsi > 30 and macd_val is not None and macd_val < signal_val:
        decision = "Sell"

result = {

        "macd": macd_val,
        "macd_signal": signal_val,
        "decision": decision,
            "coin": coin,
            "timestamp": timestamp,
            "actual_price": actual_price,
            "rsi": current_rsi
        }

        for i, label in enumerate(FUTURE_STEPS):
            result[f"pred_{label}"] = round(pred_original[i], 4)

        results.append(result)
        print(f"✅ {coin} predicted successfully.")

    except Exception as e:
        print(f"❌ Error for {coin}: {e}")

pd.DataFrame(results).to_csv(SUMMARY_PATH, index=False)
print(f"\n📊 Predictions saved to: {SUMMARY_PATH}")

if os.path.exists(DASHBOARD_PATH):
    print("\n🚀 Launching Streamlit Dashboard...")
    subprocess.Popen(["streamlit", "run", DASHBOARD_PATH])
else:
    print(f"⚠️ Dashboard script not found: {DASHBOARD_PATH}")
