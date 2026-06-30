import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

# 1. Setup & Load Data
if not os.path.exists('models'): os.makedirs('models')
df = pd.read_csv('data/energy_data.csv')

X_tahun = df[['tahun']]
X_energi = df[['solar', 'wind']] 
y_pertanian = df['produktivitas_pertanian']

# 2. Training Model Pertanian dengan Hyperparameter Tuning
print("Melatih Model Pertanian...")
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
model_pertanian = RandomForestRegressor()
grid_search = GridSearchCV(model_pertanian, param_grid, cv=2)
grid_search.fit(X_energi, y_pertanian)
joblib.dump(grid_search.best_estimator_, 'models/model_pertanian.pkl')
print(f"Model Pertanian terbaik: {grid_search.best_params_}")

# 3. Training model energi lainnya
print("Melatih Model Energi...")
targets = ['coal', 'natural_gas', 'hydro_power', 'geothermal', 'solar', 'wind']
for target in targets:
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_tahun, df[target])
    joblib.dump(model, f'models/classic_{target}.pkl')

print("Semua Model berhasil dilatih dan disimpan!")