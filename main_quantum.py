import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import pandas as pd
import joblib

# 1. Konfigurasi Quantum (Definisikan bobot di sini)
n_qubits = 2 
n_layers = 2
# Inisialisasi bobot sebagai array numpy murni agar aman di-load di app.py
weights = np.random.random((n_layers, n_qubits))

# 2. Simpan sebagai array NumPy agar tidak ada ketergantungan library saat load
joblib.dump(weights, 'models/quantum_weights.pkl')
print("Model Quantum Hybrid berhasil dikonfigurasi.")
print("Bobot model telah disimpan di models/quantum_weights.pkl sebagai array NumPy.")