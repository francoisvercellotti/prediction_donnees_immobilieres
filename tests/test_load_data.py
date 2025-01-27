import pytest
from src.load_data import load_data
import pandas as pd
import os

@pytest.fixture
def create_test_files():
    os.makedirs("data/raw", exist_ok=True)
    df_sample = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4, 5, 6],
        "col3": [7, 8, 9]
    })
    df_sample.to_csv("data/raw/test_file.csv", index=False)
    df_sample.to_parquet("data/raw/test_file.parquet")
    df_sample.to_excel("data/raw/test_file.xlsx", index=False)

def test_load_csv(create_test_files):
    data = load_data("data/raw/test_file.csv")
    assert not data.empty, "Le fichier CSV n'a pas été chargé correctement."

def test_load_parquet(create_test_files):
    data = load_data("data/raw/test_file.parquet")
    assert not data.empty, "Le fichier Parquet n'a pas été chargé correctement."

def test_load_excel(create_test_files):
    data = load_data("data/raw/test_file.xlsx")
    assert not data.empty, "Le fichier Excel n'a pas été chargé correctement."