import numpy as np
import joblib
import os

if not os.path.exists('models'): os.makedirs('models')

n_qubits = 4 
n_layers = 2
weights = np.random.random((n_layers, n_qubits))

joblib.dump(weights, 'models/quantum_weights.pkl')
print("Inisialisasi bobot Quantum berhasil disimpan di models/quantum_weights.pkl.")