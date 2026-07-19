import pandas as pd
import joblib
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Load data
df = pd.read_csv('data/energy_data.csv')
X = df[['solar', 'wind']] 
y = df['produktivitas_pertanian']

# --- Optimasi SVM (Gamma, C, Kernel) ---
print("Sedang mencari setelan terbaik untuk SVM...")
svm = SVR()
param_svm = {'kernel': ['rbf'], 'C': [1, 10, 100], 'gamma': [0.1, 0.01, 0.001]}
grid_svm = GridSearchCV(svm, param_svm, cv=5)
grid_svm.fit(X, y)
joblib.dump(grid_svm.best_estimator_, 'models/model_svm.pkl')
print(f"SVM Optimal ditemukan: {grid_svm.best_params_}")

# --- Optimasi Random Forest ---
print("Sedang mencari setelan terbaik untuk Random Forest...")
rf = RandomForestRegressor()
param_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_rf = GridSearchCV(rf, param_rf, cv=5)
grid_rf.fit(X, y)
joblib.dump(grid_rf.best_estimator_, 'models/model_pertanian.pkl')
print(f"Random Forest Optimal ditemukan: {grid_rf.best_params_}")