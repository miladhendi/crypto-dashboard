import os
import requests
import pandas as pd
import time
from datetime import datetime, timezone

# ✅ لیست رمز‌ارزها
top_coins = [
    'bitcoin', 'ethereum', 'solana', 'cardano',
    'dogecoin', 'litecoin', 'polkadot', 'tron', 'chainlink'
]

# ⚙️ تنظیمات
collect_cycles = 288  # تعداد دفعات جمع‌آوری (هر 5 دقیقه = 1 روز کامل)
delay_seconds = 300   # فاصله بین هر بار جمع‌آوری (ثانیه)

# 📁 فایل ذخیره
CSV_FILE = "crypto_prices.csv"

# 🔄 بارگذاری داده‌ی قبلی در صورت وجود
if os.path.exists(CSV_FILE):
    price_data = pd.read_csv(CSV_FILE)
    print(f"📂 Loaded existing data with {len(price_data)} rows.")
else:
    price_data = pd.DataFrame()
    print("📁 No previous data found. Starting fresh...")

# 🎯 شروع از ادامه فایل
start_index = len(price_data)

for i in range(start_index, start_index + collect_cycles):
    print(f"\n⏱️ Collecting {i + 1}/{start_index + collect_cycles}...")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    prices = {}

    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': ','.join(top_coins),
            'vs_currencies': 'usd'
        }
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
    except Exception as e:
        print("❌ API request failed:", e)
        continue

    # استخراج قیمت‌ها
    for coin in top_coins:
        prices[coin] = data.get(coin, {}).get('usd', None)
        if prices[coin] is None:
            print(f"⚠️ Missing price for {coin}")

    prices['timestamp'] = timestamp
    price_data = pd.concat([price_data, pd.DataFrame([prices])], ignore_index=True)

    # ذخیره‌سازی پس از هر بار جمع‌آوری
    price_data.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
    print(f"✔ Saved - {timestamp}")

    # مکث بین جمع‌آوری‌ها
    if i < start_index + collect_cycles - 1:
        time.sleep(delay_seconds)

print(f"\n✅ Collection complete. Total rows: {len(price_data)}. Data saved to {CSV_FILE}")
