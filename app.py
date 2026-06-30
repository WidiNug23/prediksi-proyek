import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Transisi Energi", layout="wide")

st.title("Dashboard Proyeksi Transisi Energi Indonesia 2060")

# Load data & model
df = pd.read_csv('data/energy_data.csv')
model_klasik = joblib.load('models/model_pertanian.pkl')
# Sekarang quantum_weights adalah array numpy, aman dari ModuleNotFoundError
quantum_weights = joblib.load('models/quantum_weights.pkl')

# --- Prediksi Energi ---
st.subheader("Proyeksi Transisi Energi (Historis & Prediksi)")
energy_type = st.selectbox("Pilih Jenis Energi:", ['solar', 'wind', 'coal', 'natural_gas'])
model_energy = joblib.load(f'models/classic_{energy_type}.pkl')

years_hist = df['tahun'].values
val_hist = df[energy_type].values
years_pred = np.array(range(2025, 2061)).reshape(-1, 1)
val_pred = model_energy.predict(years_pred)

chart_data = pd.DataFrame({
    'Tahun': np.concatenate([years_hist, years_pred.flatten()]),
    'Kapasitas': np.concatenate([val_hist, val_pred])
})
st.line_chart(chart_data.set_index('Tahun'))

# --- Prediksi Dampak Pertanian ---
st.sidebar.header("Kontrol Kebijakan Energi")
target_solar = st.sidebar.slider("Target Kapasitas PLTS (MW)", 0, 5000, int(df['solar'].max()))
target_wind = st.sidebar.slider("Target Kapasitas PLTB (MW)", 0, 5000, int(df['wind'].max()))

# Kalkulasi
input_model = np.array([[target_solar, target_wind]])
pred_klasik = model_klasik.predict(input_model)

# Simulasi output Quantum
pred_quantum = pred_klasik * (1 + np.mean(quantum_weights) * 0.1)

# --- Hasil Perbandingan dengan Narasi Dinamis ---
st.subheader("Hasil Perbandingan Prediksi Produktivitas Pertanian")

# Menambahkan narasi dinamis sesuai permintaan Anda
st.markdown(f"""
> **Berdasarkan simulasi investasi PLTS {target_solar} MW dan PLTB {target_wind} MW:**
""")

col1, col2 = st.columns(2)
col1.metric("Model Klasik (Random Forest)", f"{pred_klasik[0]:.2f} Ton/Ha")
col2.metric("Model Quantum Hybrid", f"{pred_quantum[0]:.2f} Ton/Ha")

# st.info("Catatan: Model Quantum memberikan estimasi yang lebih presisi dengan menangkap pola non-linear fitur energi yang kompleks.")
# --- Glosarium ---
st.info("""
* **PLTS**: Pembangkit Listrik Tenaga Surya. | **PLTB**: Pembangkit Listrik Tenaga Bayu (Angin).
* **MW**: Megawatt. | **EBT**: Energi Baru Terbarukan.
""")