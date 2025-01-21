import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, Dict

def standardize_data(X, numerical_features, standardize=False):
    """
    Applique la standardisation uniquement aux colonnes spécifiées par `numerical_features`.

    Arguments :
    X : DataFrame
        Le jeu de données contenant les variables numériques.
    numerical_features : liste
        Liste des noms de colonnes à standardiser.
    standardize : bool
        Si True, applique la standardisation ; sinon, ne fait rien.

    Retourne :
    DataFrame : Le jeu de données avec les colonnes standardisées (si applicable).
    """
    if standardize:
        scaler = StandardScaler()
        X[numerical_features] = scaler.fit_transform(X[numerical_features])

    return X


def encode_categorial_features (X: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, OneHotEncoder]]:
    """
    Encode une colonne spécifique avec OneHotEncoder et retourne des valeurs binaires.

    Args:
        X (pd.DataFrame): DataFrame d'entrée.
        column (str): Nom de la colonne à encoder.

    Returns:
        Tuple[pd.DataFrame, Dict[str, OneHotEncoder]]:
        - DataFrame avec la colonne encodée en valeurs binaires (0, 1).
        - Dictionnaire contenant la colonne et son encodeur.
    """
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    # Fit and transform the column
    encoded = encoder.fit_transform(X[[column]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]), index=X.index)

    # Convertir les float en int
    encoded_df = encoded_df.astype(int)

    # Ajouter les colonnes encodées et supprimer la colonne d'origine
    X = pd.concat([X.drop(columns=[column]), encoded_df], axis=1)

    # Return the modified DataFrame and a dictionary mapping the column to its encoder
    return X, {column: encoder}