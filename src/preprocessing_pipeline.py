import os
import sys
import joblib
import logging
import pandas as pd

# Ajouter le dossier parent au chemin pour importer les paramètres
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from split_and_encode import split_and_encode
from settings import CLASSIFICATION_TARGET, REGRESSION_TARGET

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Chemins pour sauvegarde
ENCODER_PATH = "models/encoders.joblib"

def preprocess_data(df: pd.DataFrame, columns_to_encode: list) -> tuple:
    """
    Applique le prétraitement sur les données sans standardisation ni imputation.
    """
    logging.info("Début du prétraitement des données.")

    # Séparation et encodage
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls, encoders = split_and_encode(
        df, CLASSIFICATION_TARGET, REGRESSION_TARGET, columns_to_encode
    )

    # Sauvegarde des encodeurs
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoders, ENCODER_PATH)
    logging.info(f"Encodeurs sauvegardés dans {ENCODER_PATH}")

    return X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls

if __name__ == "__main__":
    logging.info("Chargement du DataFrame.")
    df = pd.read_parquet('data/interim/prepared_dataset.parquet')

    # Nettoyage des colonnes
    df.columns = df.columns.str.strip()
    logging.info(f"Colonnes disponibles après nettoyage : {list(df.columns)}.")

    # Colonnes à encoder
    columns_to_encode = ['type_batiment', 'nom_region']

    # Exécution du prétraitement
    results = preprocess_data(df, columns_to_encode)

    logging.info("Prétraitement terminé avec succès.")
