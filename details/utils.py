import pandas as pd
import os

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File data tidak ditemukan: {path}")
    return pd.read_csv(path)
