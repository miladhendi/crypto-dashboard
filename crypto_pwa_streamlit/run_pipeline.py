import subprocess
import os

# === مرحله 1: جمع‌آوری داده با MiladCoin.py ===
print("\n🚀 [1/4] Collecting data from CoinGecko...")
subprocess.run(["python", "MiladCoin.py"], check=True)

# === مرحله 2: آماده‌سازی داده با prepare_data.py ===
print("\n🔧 [2/4] Preparing data...")
subprocess.run(["python", "prepare_data.py"], check=True)

# === مرحله 3: آموزش مدل LSTM با model_lstm.py ===
print("\n📚 [3/4] Training LSTM models...")
subprocess.run(["python", "model_lstm.py"], check=True)

# === مرحله 4: پیش‌بینی و اجرای داشبورد با streamlit_app.py (نسخه نهایی) ===
print("\n🔮 [4/4] Predicting future values & launching dashboard...")
subprocess.run(["python", "pipeline.py"], check=True)

# اجرای داشبورد نهایی
subprocess.Popen(["streamlit", "run", "streamlit_app.py"])

print("\n✅ All steps completed successfully! Dashboard is running...")
