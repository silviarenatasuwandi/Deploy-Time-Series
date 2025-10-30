import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os

def create_lag_features(df, target_col='NO2', n_lags=3):
    for i in range(1, n_lags + 1):
        df[f'{target_col}_lag{i}'] = df[target_col].shift(i)
    df = df.dropna()
    return df

def train_knn_model(data_path, n_lags=3):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File data tidak ditemukan: {data_path}")

    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.dropna(subset=['NO2'])

    # Buat fitur lag untuk memperhitungkan autokorelasi
    df = create_lag_features(df, 'NO2', n_lags)

    # Fitur: NO2_lag1, NO2_lag2, ...
    feature_cols = [col for col in df.columns if 'lag' in col]
    X = df[feature_cols].values
    y = df['NO2'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Normalisasi
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model KNN
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn.fit(X_train_scaled, y_train)

    # Prediksi dan evaluasi MAPE
    y_pred = knn.predict(X_test_scaled)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Simpan model dan scaler
    os.makedirs("model", exist_ok=True)
    joblib.dump(knn, "model/knn_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    return 1 - mape  # semakin tinggi semakin bagus (mirip akurasi)
