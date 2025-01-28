import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import logging
import sys
import os

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # Affichage des logs dans la console
    ]
)

# Ajouter le dossier parent au chemin pour importer les paramètres
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from settings import CLASSIFICATION_TARGET, REGRESSION_TARGET, random_state


# Fonction pour encoder une colonne catégorielle avec OneHotEncoder
def encode_column(df: pd.DataFrame, column: str, encoder: OneHotEncoder = None) -> Tuple[pd.DataFrame, Dict[str, OneHotEncoder]]:
    """
    Encode une colonne spécifique avec OneHotEncoder et retourne des valeurs binaires.

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        column (str): Nom de la colonne à encoder.
        encoder (OneHotEncoder, optionnel): Un encodeur déjà entraîné. Si None, l'encodeur sera créé et entraîné.

    Returns:
        Tuple[pd.DataFrame, Dict[str, OneHotEncoder]]: DataFrame avec la colonne encodée et le dictionnaire contenant l'encodeur.
    """
    logging.info(f"Encodage de la colonne '{column}' avec OneHotEncoder.")
    if encoder is None:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        encoder.fit(df[[column]])
        logging.info(f"Encoder créé et entraîné pour la colonne '{column}'.")

    encoded = encoder.transform(df[[column]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]), index=df.index)
    encoded_df = encoded_df.astype(int)

    df = pd.concat([df.drop(columns=[column]), encoded_df], axis=1)
    logging.info(f"Colonne '{column}' encodée avec succès. Dimensions du DataFrame: {df.shape}.")
    return df, {column: encoder}


# Fonction principale pour la séparation et l'encodage
def split_and_encode(df: pd.DataFrame, target_classification: str, target_regression: str, columns_to_encode: list = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, Dict[str, OneHotEncoder]]:
    """
    Sépare le DataFrame en train et test, et encode des colonnes catégorielles, si spécifié.
    """
    logging.info("Début de la séparation du DataFrame en ensembles d'entraînement et de test.")

    # Séparation du DataFrame
    trainset, testset = train_test_split(df, test_size=0.2, random_state=random_state)
    logging.info(f"Train set : {trainset.shape}, Test set : {testset.shape}.")

    # Séparer les features et les cibles
    X_train = trainset.drop([target_regression, target_classification], axis=1)
    y_train_regression = trainset[target_regression]
    y_train_classification = trainset[target_classification]

    X_test = testset.drop([target_regression, target_classification], axis=1)
    y_test_regression = testset[target_regression]
    y_test_classification = testset[target_classification]

    logging.info(f"Valeurs uniques de 'nom_region' avant encodage : {df['nom_region'].unique()}")


    encoders = {}

    # Encodage des colonnes catégorielles
    if columns_to_encode:
        for col in columns_to_encode:
            if col not in df.columns:
                logging.error(f"La colonne '{col}' n'existe pas dans le DataFrame.")
                raise ValueError(f"La colonne {col} n'existe pas dans le DataFrame.")

            logging.info(f"Encodage de la colonne '{col}'.")
            X_train, encoder = encode_column(X_train, col)
            X_test, _ = encode_column(X_test, col, encoder.get(col))
            encoders[col] = encoder[col]

    logging.info("Encodage terminé. Dimensions finales:")
    logging.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}.")

    # Afficher les 5 premières lignes de chaque DataFrame en sortie
    logging.info("5 premières lignes de X_train:")
    logging.info(X_train.head())

    logging.info("5 premières lignes de X_test:")
    logging.info(X_test.head())

    return X_train, X_test, y_train_regression, y_test_regression, y_train_classification, y_test_classification, encoders


if __name__ == "__main__":
    logging.info("Chargement du DataFrame.")
    df = pd.read_parquet('data/interim/prepared_dataset.parquet')

    # Nettoyage des colonnes
    df.columns = df.columns.str.strip()
    logging.info(f"Colonnes disponibles après nettoyage : {list(df.columns)}.")

    # Paramètres pour la fonction
    target_classification = CLASSIFICATION_TARGET
    target_regression = REGRESSION_TARGET
    columns_to_encode = ['type_batiment', 'nom_region']

    logging.info("Début de la fonction split_and_encode.")
    results = split_and_encode(
        df=df,
        target_classification=target_classification,
        target_regression=target_regression,
        columns_to_encode=columns_to_encode
    )

    logging.info("split_and_encode terminé avec succès.")
