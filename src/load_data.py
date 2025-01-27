"""
Module load_data.py :
Contient les fonctions nécessaires pour charger les données selon le type d'extension.
"""

import os
import sys
import pandas as pd


def load_data(filepath, delimiter=';', preview=True, **kwargs):
    """
    Charge les données en fonction de l'extension du fichier.

    Args:
        filepath (str): Chemin vers le fichier à charger.
        delimiter (str, optional): Délimiteur utilisé pour les fichiers CSV ou TXT. Par défaut, ';'.
        preview (bool, optional): Afficher un aperçu des données si True. Par défaut, True.
        **kwargs: Arguments supplémentaires passés aux fonctions de chargement de pandas.

    Returns:
        pd.DataFrame: Un DataFrame contenant les données chargées.
    """
    # Vérifier si le fichier existe
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier '{filepath}' n'existe pas.")

    # Obtenir l'extension du fichier
    _, file_extension = os.path.splitext(filepath)

    # Charger le fichier en fonction de son extension
    try:
        if file_extension == '.csv':
            data = pd.read_csv(filepath, delimiter=delimiter, **kwargs)
        elif file_extension in ['.xls', '.xlsx']:
            data = pd.read_excel(filepath, **kwargs)
        elif file_extension == '.json':
            data = pd.read_json(filepath, **kwargs)
        elif file_extension == '.parquet':
            data = pd.read_parquet(filepath, **kwargs)
        elif file_extension == '.txt':
            data = pd.read_csv(filepath, delimiter=delimiter, **kwargs)
        else:
            raise ValueError(f"L'extension de fichier '{file_extension}' n'est pas supportée. "
                             "Extensions supportées : .csv, .xls, .xlsx, .json, .parquet, .txt.")

        # Afficher un aperçu et les dimensions des données
        if preview:
            print(f"Fichier chargé avec succès. Dimensions : {data.shape[0]} lignes, {data.shape[1]} colonnes.")
            print("Aperçu des données :")
            print(data.head())

        return data

    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return None


if __name__ == "__main__":
    # Vérifier si un chemin de fichier a été fourni comme argument
    if len(sys.argv) < 2:
        print("Erreur : il faut spécifier le chemin du fichier à charger.")
        sys.exit(1)

    # Charger le fichier et afficher le résultat
    file_path = sys.argv[1]
    dataset = load_data(file_path)
