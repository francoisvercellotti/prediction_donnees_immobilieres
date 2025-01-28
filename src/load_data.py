"""
Module load_data.py :
Contient les fonctions nécessaires pour charger les données selon le type d'extension.
"""


import os
import sys
import pandas as pd


def load_data(filepath, delimiter=';', **kwargs):
    """
    Charge les données en fonction de l'extension du fichier.

    Args:
        filepath (str): Chemin vers le fichier à charger.
        delimiter (str, optional): Délimiteur utilisé pour les fichiers CSV ou TXT. Par défaut, ';'.
        **kwargs: Arguments supplémentaires passés aux fonctions de chargement de pandas.

    Returns:
        pd.DataFrame: Un DataFrame contenant les données chargées.
    """
    # Obtenir l'extension du fichier
    _, file_extension = os.path.splitext(filepath)

    # Charger le fichier en fonction de son extension
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
        raise ValueError(f"Extension de fichier '{file_extension}' non supportée.")

    # Vérifier les premières lignes pour confirmer le chargement
    print("Les premières lignes du fichier chargé :")
    print(data.head())

    # Colonnes nécessaires pour l'entraînement
    columns_needed = ['prix',  'id_ville',
       'ville',  'type_batiment', 'vefa', 'n_pieces',
       'surface_habitable', 'latitude', 'longitude',
       'mois_transaction', 'annee_transaction', 'prix_m2', 'prix_m2_moyen',
       'nb_transactions_mois', 'en_dessous_du_marche',
       'nom_departement','nom_region']


    # Garder uniquement les colonnes nécessaires si elles existent
    data = data[[col for col in columns_needed if col in data.columns]]

    # Vérifier les colonnes après filtrage
    print("Colonnes après filtrage :", data.columns.tolist())

    return data


if __name__ == "__main__":
    # Vérifier si un chemin de fichier a été fourni comme argument
    if len(sys.argv) < 2:
        print("Erreur : il faut spécifier le chemin du fichier à charger.")
        sys.exit(1)  # Sortir du programme en cas d'erreur

    # Le chemin du fichier est passé comme argument
    file_path_arg = sys.argv[1]

    # Charger les données à partir du fichier donné
    df = load_data(file_path_arg)

    # Sauvegarder les données nettoyées sous un autre nom ou emplacement
    OUTPUT_FILE = 'data/loaded/loaded_dataset.parquet'
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Le fichier '{file_path_arg}' a été chargé et sauvegardé sous '{OUTPUT_FILE}'.")
