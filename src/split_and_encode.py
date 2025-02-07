"""
Module pour le prétraitement des données, incluant le découpage train/test et l'encodage des variables catégorielles.
Ce module fournit des fonctionnalités pour préparer les données avant l'entraînement des modèles.
"""

import logging
import os
import sys
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Ajouter le dossier parent au chemin pour importer les paramètres
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from settings import CLASSIFICATION_TARGET, REGRESSION_TARGET, random_state

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def encode_column(
    data: pd.DataFrame,
    column: str,
    encoder: OneHotEncoder = None
) -> Tuple[pd.DataFrame, Dict[str, OneHotEncoder]]:
    """
    Encode une colonne spécifique avec OneHotEncoder et retourne des valeurs binaires.

    Args:
        data: DataFrame d'entrée
        column: Nom de la colonne à encoder
        encoder: Un encodeur déjà entraîné. Si None, l'encodeur sera créé et entraîné

    Returns:
        DataFrame avec la colonne encodée et le dictionnaire contenant l'encodeur
    """
    logging.info("Encodage de la colonne '%s' avec OneHotEncoder.", column)

    if encoder is None:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        encoder.fit(data[[column]])
        logging.info("Encoder créé et entraîné pour la colonne '%s'.", column)

    encoded = encoder.transform(data[[column]])
    feature_names = encoder.get_feature_names_out([column])
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=data.index)
    encoded_df = encoded_df.astype(int)

    result = pd.concat([data.drop(columns=[column]), encoded_df], axis=1)
    logging.info(
        "Colonne '%s' encodée avec succès. Dimensions du DataFrame: %s.",
        column,
        result.shape
    )

    return result, {column: encoder}


def split_and_encode(
    data: pd.DataFrame,
    target_class: str,
    target_reg: str,
    cols_to_encode: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, Dict[str, OneHotEncoder]]:
    """
    Sépare le DataFrame en train et test, et encode des colonnes catégorielles.

    Args:
        data: DataFrame d'entrée
        target_class: Nom de la colonne cible pour la classification
        target_reg: Nom de la colonne cible pour la régression
        cols_to_encode: Liste des colonnes à encoder

    Returns:
        Tuple contenant:
        - X_train: Features d'entraînement
        - X_test: Features de test
        - y_train_reg: Cible régression d'entraînement
        - y_test_reg: Cible régression de test
        - y_train_class: Cible classification d'entraînement
        - y_test_class: Cible classification de test
        - encoders: Dictionnaire des encodeurs utilisés
    """
    logging.info("Début de la séparation du DataFrame en ensembles d'entraînement et de test.")

    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=random_state
    )

    logging.info("Train set : %s, Test set : %s.", train_data.shape, test_data.shape)

    # Séparer les features et les cibles
    features_train = train_data.drop([target_reg, target_class], axis=1)
    target_train_reg = train_data[target_reg]
    target_train_class = train_data[target_class]

    features_test = test_data.drop([target_reg, target_class], axis=1)
    target_test_reg = test_data[target_reg]
    target_test_class = test_data[target_class]

    logging.info("Valeurs uniques de 'nom_region' : %s", data['nom_region'].unique())

    encoders = {}

    if cols_to_encode:
        for col in cols_to_encode:
            if col not in data.columns:
                msg = f"La colonne {col} n'existe pas dans le DataFrame."
                logging.error(msg)
                raise ValueError(msg)

            logging.info("Encodage de la colonne '%s'.", col)
            features_train, encoder = encode_column(features_train, col)
            features_test, _ = encode_column(features_test, col, encoder[col])
            encoders[col] = encoder[col]

    logging.info(
        "X_train: %s, X_test: %s.",
        features_train.shape,
        features_test.shape
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    return (
        features_train,
        features_test,
        target_train_reg,
        target_test_reg,
        target_train_class,
        target_test_class,
        encoders
    )


if __name__ == "__main__":
    logging.info("Chargement du DataFrame.")
    input_df = pd.read_parquet('data/interim/prepared_dataset.parquet')
    input_df.columns = input_df.columns.str.strip()

    logging.info("Colonnes disponibles après nettoyage : %s", list(input_df.columns))

    COLS_TO_ENCODE = ['type_batiment', 'nom_region']

    results = split_and_encode(
        data=input_df,
        target_class=CLASSIFICATION_TARGET,
        target_reg=REGRESSION_TARGET,
        cols_to_encode=COLS_TO_ENCODE
    )

    logging.info("split_and_encode terminé avec succès.")