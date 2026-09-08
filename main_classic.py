import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

if not os.path.exists('models'): os.makedirs('models')
data_path = 'data/noaa_processed_data.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError("Jalankan preprocess_data.py terlebih dahulu.")

df = pd.read_csv(data_path)

X_time = df[['tahun', 'bulan']]
targets = ['h2_ppb', 'ch4_ppb', 'co2_ppm', 'co_ppb']

print("Melatih Model Baseline Klasik per Gas...")
for target in targets:
    if target in df.columns:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_time, df[target])
        joblib.dump(model, f'models/classic_{target}.pkl')
        print(f"Model klasik untuk '{target}' berhasil disimpan.")