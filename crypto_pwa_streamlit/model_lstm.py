import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# تابع محاسبه خطا
def calculate_errors(real, predicted):
    mae = np.mean(np.abs(real - predicted))
    rmse = np.sqrt(np.mean((real - predicted) ** 2))
    non_zero_mask = real != 0
    mape = np.mean(np.abs((real[non_zero_mask] - predicted[non_zero_mask]) / real[non_zero_mask])) * 100
    return mae, rmse, mape

print("📂 Loading data...")

# بارگذاری داده آماده‌شده
df = pd.read_csv("prepared_data_multistep.csv")

# استخراج نام رمزارزها از ستون‌ها
coins = ['bitcoin', 'ethereum', 'solana', 'cardano', 'dogecoin', 'litecoin', 'polkadot', 'tron', 'chainlink']
future_labels = ['5min', '10min', '30min', '1h']
sequence_length = 12

# ساخت پوشه‌های خروجی در صورت نبود
os.makedirs("models_multistep", exist_ok=True)
os.makedirs("evaluations", exist_ok=True)

# آموزش مدل برای هر رمزارز
for coin in coins:
    print(f"\n📈 Training model for {coin}...")

    try:
        required_columns = [f"{coin}_change"] + [f"{coin}_future_{label}" for label in future_labels]
        if not all(col in df.columns for col in required_columns):
            print(f"⚠️ Skipped {coin} - missing columns")
            continue

        coin_df = df[required_columns].dropna()
        if len(coin_df) < sequence_length + 1:
            print(f"⚠️ Skipped {coin} - not enough data ({len(coin_df)} rows)")
            continue

        # نرمال‌سازی
        scaler = MinMaxScaler()
        coin_scaled = scaler.fit_transform(coin_df)

        # ساخت X و y
        X, y = [], []
        for i in range(sequence_length, len(coin_scaled)):
            X.append(coin_scaled[i-sequence_length:i, 0])  # فقط ستون قیمت تغییرات گذشته
            y.append(coin_scaled[i, 1:])  # تغییرات آینده

        X = np.array(X).reshape(-1, sequence_length, 1)
        y = np.array(y)

        # ساخت مدل
        model = Sequential([
            LSTM(64, input_shape=(sequence_length, 1)),
            Dense(32, activation='relu'),
            Dense(len(future_labels))
        ])
        model.compile(optimizer='adam', loss='mse')
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        # آموزش مدل
        model.fit(X, y, epochs=50, batch_size=8, verbose=0, callbacks=[early_stop])
        model.save(f"models_multistep/{coin}_multistep.h5")
        print(f"✅ Model for {coin} saved.")

        # ارزیابی مدل
        y_pred = model.predict(X)
        mae, rmse, mape = calculate_errors(y, y_pred)

        with open(f"evaluations/{coin}_evaluation.txt", "w") as f:
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"MAPE: {mape:.2f}%\n")

        print(f"📊 Evaluation for {coin} completed: MAPE = {mape:.2f}%")

    except Exception as e:
        print(f"❌ Error training {coin}: {e}")

print("\n✅ All multi-step models trained and evaluated.")
