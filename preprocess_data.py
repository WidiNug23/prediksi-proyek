import numpy as np
import os
import pandas as pd

if not os.path.exists("data"):
  os.makedirs("data")

input_path = "data/Dataset_gabungan.csv"
if not os.path.exists(input_path):
  raise FileNotFoundError(
      f"File {input_path} tidak ditemukan. Letakkan file dataset di folder"
      " data/."
  )

df = pd.read_csv(input_path)

df["tanggal"] = pd.to_datetime(df["tanggal"])
df = df.sort_values("tanggal").reset_index(drop=True)

df = df.set_index("tanggal").resample("MS").asfreq().reset_index()
df["tahun"] = df["tanggal"].dt.year
df["bulan"] = df["tanggal"].dt.month

gas_cols = ["h2_ppb", "ch4_ppb", "co2_ppm", "co_ppb"]
df[gas_cols] = df[gas_cols].interpolate(method="linear")

# 1. Target Utama sesuai gambar klien: Delta H2(t) = H2(t) - H2(t-1)
df["delta_h2"] = df["h2_ppb"].diff()
df["target_h2_delta"] = df["delta_h2"]

# 2. Target Turunan sesuai gambar klien: H2 berlebih = H2(t) - baseline12(t) (rolling min 12 bulan)
df["baseline_12"] = df["h2_ppb"].rolling(window=12).min()
df["target_h2_berlebih"] = df["h2_ppb"] - df["baseline_12"]

# Fitur Prediktor Pendukung
df["delta_co"] = df["co_ppb"].diff()
df["delta_ch4"] = df["ch4_ppb"].diff()
df["delta_co2"] = df["co2_ppm"].diff()

df["rasio_h2_co"] = np.where(
    df["delta_co"] != 0, df["delta_h2"] / df["delta_co"], 0
)

df["h2_lag1"] = df["h2_ppb"].shift(1)
df["h2_lag2"] = df["h2_ppb"].shift(2)

df["sin_bulan"] = np.sin(2 * np.pi * df["bulan"] / 12)
df["cos_bulan"] = np.cos(2 * np.pi * df["bulan"] / 12)

df_clean = df.dropna().reset_index(drop=True)

output_processed = "data/noaa_processed_data.csv"
df_clean.to_csv(output_processed, index=False)
print(
    "Preprocessing selesai! Menambahkan 'target_h2_delta' dan"
    f" 'target_h2_berlebih'. Data disimpan ke '{output_processed}'."
)