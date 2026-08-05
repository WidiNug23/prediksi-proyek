import pandas as pd

# 1. Load data utama
df_main = pd.read_csv('data/energy_data.csv')

# 2. Load data eksperimen NLR
df_wind = pd.read_csv('data/combined_wind_experiments.csv')
df_solar = pd.read_csv('data/combined_solarPV_experiments.csv')

# 3. Ambil data laju aliran hidrogen dan efisiensi dengan pengecekan nama kolom
# Untuk Wind
wind_flow_col = 'IVAL_f_FM011_Flow' if 'IVAL_f_FM011_Flow' in df_wind.columns else 'IVAL_f_FM011_Flow (kg/hr)'
wind_eff_col = 'Efficiency (kWh/kg)'
wind_flow_avg = df_wind[wind_flow_col].mean()
wind_eff_avg = df_wind[wind_eff_col].mean()

# Untuk Solar (biasanya menggunakan format dengan satuan di belakangnya)
solar_flow_col = 'IVAL_f_FM011_Flow (kg/hr)' if 'IVAL_f_FM011_Flow (kg/hr)' in df_solar.columns else 'IVAL_f_FM011_Flow'
solar_eff_col = 'Efficiency (kWh/kg)'
solar_flow_avg = df_solar[solar_flow_col].mean()
solar_eff_avg = df_solar[solar_eff_col].mean()

# Gabungkan metrik hidrogen dari angin dan surya
avg_h2_flow = (wind_flow_avg + solar_flow_avg) / 2
avg_h2_eff = (wind_eff_avg + solar_eff_avg) / 2

# 4. Masukkan ke dalam tabel utama
df_main['hydrogen_flow_rate'] = None
df_main['hydrogen_efficiency'] = None

# Mengisi data untuk tahun 2023 dan 2024 sebagai simulasi penerapan data NLR
df_main.loc[df_main['tahun'].isin([2023, 2024]), 'hydrogen_flow_rate'] = avg_h2_flow
df_main.loc[df_main['tahun'].isin([2023, 2024]), 'hydrogen_efficiency'] = avg_h2_eff

# Simpan hasilnya
df_main.to_csv('data/energy_data_with_hydrogen.csv', index=False)
print("Data berhasil digabungkan dan disimpan ke 'data/energy_data_with_hydrogen.csv'!")