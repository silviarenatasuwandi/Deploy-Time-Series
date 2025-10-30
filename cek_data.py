import pandas as pd

# Baca file CSV
df = pd.read_csv("data/NO2_Manyar.csv")

# Tampilkan 5 baris pertama data
print("==== 5 Baris Pertama Data ====")
print(df.head())

# Tampilkan tipe data tiap kolom
print("\n==== Tipe Data Tiap Kolom ====")
print(df.dtypes)

# Tampilkan informasi umum tentang dataset
print("\n==== Info Dataset ====")
print(df.info())
