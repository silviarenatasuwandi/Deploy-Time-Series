import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os

# === 1. Baca dataset ===
data_path = "data/NO2_Manyar.csv"
df = pd.read_csv(data_path)
df['time'] = pd.to_datetime(df['time'])
df = df.dropna(subset=['NO2'])

# === 2. Buat fitur lag (autokorelasi) ===
def create_lag_features(df, target_col='NO2', n_lags=3):
    for i in range(1, n_lags + 1):
        df[f'{target_col}_lag{i}'] = df[target_col].shift(i)
    df = df.dropna()
    return df

df = create_lag_features(df, 'NO2', n_lags=3)

X = df[['NO2_lag1', 'NO2_lag2', 'NO2_lag3']]
y = df['NO2']

# === 3. Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === 4. Normalisasi ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 5. Latih model KNN ===
model = KNeighborsRegressor(n_neighbors=5, weights='distance')
model.fit(X_train_scaled, y_train)

# === 6. Evaluasi model ===
y_pred = model.predict(X_test_scaled)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Model berhasil dilatih ✅\nMAPE: {mape:.4f}")

# === 7. Simpan model & scaler ===
joblib.dump(model, "model_no2.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model dan scaler telah disimpan!")
