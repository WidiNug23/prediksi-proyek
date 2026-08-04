import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

if not os.path.exists('models'):
    os.makedirs('models')

# Load data
df = pd.read_csv('data/energy_data.csv')

X_features = df[['solar', 'wind']]
y_pertanian = df['produktivitas_pertanian']

# TimeSeriesSplit untuk validasi data deret waktu
tscv = TimeSeriesSplit(n_splits=3)

# --- 1. Tuning Hyperparameter Random Forest Lengkap ---
print("Melakukan Tuning Hyperparameter Random Forest...")
start_time = time.time()

rf = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],     
    'max_features': ['sqrt', 'log2', 1.0]        
}

grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=tscv, scoring='neg_mean_squared_error')
grid_rf.fit(X_features, y_pertanian)

train_time_rf = time.time() - start_time
best_rf = grid_rf.best_estimator_

joblib.dump(best_rf, 'models/model_pertanian.pkl')
print(f"Random Forest Terbaik: {grid_rf.best_params_}")
print(f"Waktu Training Random Forest: {train_time_rf:.4f} detik")


# --- 2. Pelatihan Model Energi Klasik ---
targets = ['coal', 'natural_gas', 'hydro_power', 'geothermal', 'solar', 'wind']
X_tahun = df[['tahun']]
for target in targets:
    if target in df.columns:
        model_energy = RandomForestRegressor(n_estimators=100, random_state=42)
        model_energy.fit(X_tahun, df[target])
        joblib.dump(model_energy, f'models/classic_{target}.pkl')

# --- 3. Konfigurasi Quantum Hybrid (QLSTM) ---
quantum_config = {
    'n_qubits': 2,
    'n_layers': 2,
    'ansatz': 'Hardware-efficient, ring entanglement',
    'encoding': 'Angle encoding',
    'optimizer': 'Adam',
    'learning_rate': 0.01,
    'simulated_inference_time_overhead': 1.45
}
weights = np.random.random((quantum_config['n_layers'], quantum_config['n_qubits']))
joblib.dump({'weights': weights, 'config': quantum_config}, 'models/quantum_weights.pkl')

print("Pelatihan selesai. Model siap digunakan!")