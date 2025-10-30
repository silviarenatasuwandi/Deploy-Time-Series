# ==========================================================
# app.py (FINAL VERSION - KNN untuk Manyar, versi bersih)
# Sistem Prediksi NO2 berbasis KNN
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from datetime import timedelta
from sklearn.metrics import mean_absolute_percentage_error
from details.utils import load_data

# --- KONFIGURASI STREAMLIT ---
st.set_page_config(
    page_title="Prediksi NO2 Manyar",
    layout="wide"
)

# ==========================================================
# 0. HEADER
# ==========================================================
st.title("🌫️ Sistem Prediksi Konsentrasi NO₂ - Manyar")
st.caption("Menggunakan model KNN untuk memprediksi konsentrasi NO₂ harian berdasarkan data historis.")
st.markdown("---")

# ==========================================================
# 1. LOAD DATA DAN MODEL
# ==========================================================
DATA_FILE = "data/NO2_Manyar.csv"
MODEL_FILE = "model/knn_model.pkl"
SCALER_FILE = "model/scaler.pkl"

# --- Cek dan load file ---
if not os.path.exists(DATA_FILE):
    st.error(f"❌ File data tidak ditemukan: {DATA_FILE}")
    st.stop()

if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    st.error("❌ Model atau scaler belum ditemukan di folder 'model/'. Pastikan file .pkl sudah tersedia.")
    st.stop()

# --- Load data dan model ---
df = load_data(DATA_FILE)
df['time'] = pd.to_datetime(df['time'])
df = df.dropna(subset=['NO2'])

knn = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# ==========================================================
# 2. INPUT TANGGAL PREDIKSI
# ==========================================================
st.header("🗓️ Pilih Periode Prediksi")

last_date = df['time'].iloc[-1].date()
start_date_forecast = last_date + timedelta(days=1)
st.markdown(f"*Tanggal terakhir dalam data:* `{last_date}`")

target_date = st.date_input(
    "Pilih tanggal akhir prediksi:",
    min_value=start_date_forecast,
    value=start_date_forecast + timedelta(days=7),
    max_value=start_date_forecast + timedelta(days=60)
)

days_to_predict = (target_date - last_date).days

if days_to_predict <= 0:
    st.warning("⚠️ Pilih tanggal setelah tanggal terakhir dalam dataset.")
    st.stop()

# ==========================================================
# 3. TOMBOL PREDIKSI
# ==========================================================
if st.button(f"🔮 Prediksi Selama {days_to_predict} Hari (Hingga {target_date})"):
    with st.spinner("⏳ Model sedang memprediksi..."):
        # Gunakan 4 lag terakhir (karena scaler menggunakan 4 fitur)
        n_features = scaler.n_features_in_
        last_values = df['NO2'].values[-n_features:]
        preds = []

        for _ in range(days_to_predict):
            X_input = np.array(last_values[-n_features:]).reshape(1, -1)
            X_scaled = scaler.transform(X_input)
            pred = knn.predict(X_scaled)[0]
            preds.append(pred)
            last_values = np.append(last_values, pred)

        # Buat dataframe hasil prediksi
        future_dates = pd.date_range(df['time'].iloc[-1] + pd.Timedelta(days=1), periods=days_to_predict)
        result = pd.DataFrame({'Tanggal': future_dates, 'Prediksi_NO2': preds})
        result = result.set_index('Tanggal')

        st.success("✅ Prediksi selesai!")

        # ==========================================================
        # 4. HITUNG MAPE (jika data aktual tersedia)
        # ==========================================================
        if len(df) >= days_to_predict:
            actual_values = df['NO2'].values[-days_to_predict:]
            mape = mean_absolute_percentage_error(actual_values, preds[:len(actual_values)]) * 100
        else:
            mape = None

        # ==========================================================
        # 5. GRAFIK PREDIKSI
        # ==========================================================
        st.header("📈 Hasil Prediksi NO₂")
        fig, ax = plt.subplots(figsize=(12, 5))

        # Data historis (90 hari terakhir)
        ax.plot(df['time'].tail(90), df['NO2'].tail(90),
                label="Data Historis (Aktual)", color='#1f77b4', linewidth=2)

        # Garis batas awal prediksi
        ax.axvline(x=future_dates[0], color='black', linestyle='--', linewidth=1, label='Awal Prediksi')

        # Hasil prediksi
        ax.plot(future_dates, preds, label="Prediksi NO₂", color='red', linewidth=2)

        ax.set_title(f"Prediksi Konsentrasi NO₂ Hingga {target_date}", fontsize=14)
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Konsentrasi NO₂ (µg/m³)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown("---")

        # ==========================================================
        # 6. TABEL HASIL PREDIKSI
        # ==========================================================
        st.header("📋 Tabel Hasil Prediksi")
        st.dataframe(result, use_container_width=True)

        # ==========================================================
        # 7. METRIK MODEL
# ==========================================================
        st.header("📊 Kinerja Model")
        if mape:
            st.metric(label="MAPE (Mean Absolute Percentage Error)", value=f"{mape:.2f}%")
        else:
            st.info("Tidak cukup data aktual untuk menghitung MAPE.")

        # ==========================================================
        # 8. UNDUH HASIL
        # ==========================================================
        csv = result.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Unduh Hasil Prediksi (.csv)",
            data=csv,
            file_name=f"Prediksi_NO2_Manyar_Hingga_{target_date}.csv",
            mime="text/csv"
        )
