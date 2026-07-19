import pandas as pd
import joblib
import os
from sklearn.svm import SVR

# 1. Update training data agar menyertakan fitur hidrogen
# Jika data 'hydrogen' belum ada di CSV, tambahkan kolom tersebut sebelum training
df = pd.read_csv('data/energy_data.csv')
# Pastikan ada kolom 'hydrogen' di df. Jika belum, tambahkan dummy/data riil
X = df[['solar', 'wind', 'hydrogen']] 
y = df['produktivitas_pertanian']

# 2. Latih model SVM dengan 3 fitur
model_svm = SVR(kernel='rbf', C=100, gamma=0.1)
model_svm.fit(X, y)

# 3. Simpan model yang baru
joblib.dump(model_svm, 'models/model_svm.pkl')
print("Model SVM dengan 3 fitur (Solar, Wind, Hydrogen) berhasil disimpan!")