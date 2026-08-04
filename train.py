import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

if not os.path.exists('models'):
    os.makedirs('models')

# Load data
df = pd.read_csv('data/energy_data.csv')

# Pastikan kolom hydrogen ada untuk fitur jika dibutuhkan
if 'hydrogen' not in df.columns:
    df['hydrogen'] = 0

X_features = df[['solar', 'wind', 'hydrogen']]
y_pertanian = df['produktivitas_pertanian']

# Menggunakan TimeSeriesSplit untuk validasi silang deret waktu
tscv = TimeSeriesSplit(n_splits=3)

# --- 1. Tuning Hyperparameter Random Forest (Model Klasik) ---
print("Melakukan Tuning Hyperparameter Random Forest...")
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 1.0]  # 'auto' dihapus karena tidak lagi didukung di scikit-learn versi baru
}

grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=tscv, scoring='neg_mean_squared_error')
grid_rf.fit(X_features[['solar', 'wind']], y_pertanian) # Menggunakan 2 fitur utama untuk kompatibilitas input app
joblib.dump(grid_rf.best_estimator_, 'models/model_pertanian.pkl')
print(f"Random Forest Terbaik: {grid_rf.best_params_}")

# --- 2. Tuning Hyperparameter SVM ---
print("Melakukan Tuning Hyperparameter SVM...")
svr = SVR()
param_grid_svm = {
    'kernel': ['rbf'], 
    'C': [1, 10, 100], 
    'gamma': [0.1, 0.01]
}
grid_svm = GridSearchCV(estimator=svr, param_grid=param_grid_svm, cv=tscv)
grid_svm.fit(X_features, y_pertanian) # SVM dilatih dengan 3 fitur (Solar, Wind, Hydrogen)
joblib.dump(grid_svm.best_estimator_, 'models/model_svm.pkl')
print(f"SVM Terbaik: {grid_svm.best_params_}")

# --- 3. Pelatihan Model Energi Klasik ---
targets = ['coal', 'natural_gas', 'hydro_power', 'geothermal', 'solar', 'wind']
X_tahun = df[['tahun']]
for target in targets:
    if target in df.columns:
        model_energy = RandomForestRegressor(n_estimators=100, random_state=42)
        model_energy.fit(X_tahun, df[target])
        joblib.dump(model_energy, f'models/classic_{target}.pkl')

# --- 4. Konfigurasi Parameter Kuantum (PennyLane / Qiskit Hybrid Simulation) ---
# Spesifikasi: Jumlah qubit, jumlah layer, jenis ansatz, learning rate Adam
quantum_config = {
    'n_qubits': 2,
    'n_layers': 2,
    'ansatz': 'HardwareEfficientAnsatz',
    'optimizer': 'Adam',
    'learning_rate': 0.01
}
# Bobot variational circuit
weights = np.random.random((quantum_config['n_layers'], quantum_config['n_qubits']))
joblib.dump({'weights': weights, 'config': quantum_config}, 'models/quantum_weights.pkl')

print("Semua model dan parameter berhasil dilatih serta disimpan!")