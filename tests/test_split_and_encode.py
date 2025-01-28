import os
import sys
import pytest
import pandas as pd
from src.split_and_encode import split_and_encode, encode_column
# Ajouter le dossier parent au chemin pour importer les paramètres
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from settings import CLASSIFICATION_TARGET, REGRESSION_TARGET, random_state

# Préparer les données pour les tests
@pytest.fixture
def df():
    data = {
        'prix': [190000.0, 90000.0, 65000.0, 250000.0, 150000.0, 300000.0, 120000.0, 200000.0, 175000.0, 220000.0],
        'type_batiment': ['Appartement', 'Appartement', 'Appartement', 'Maison', 'Maison', 'Appartement', 'Maison', 'Appartement', 'Maison', 'Appartement'],
        'vefa': [0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
        'surface_habitable': [139, 60, 87, 120, 95, 130, 80, 100, 110, 105],
        'latitude': [46.198306, 46.199362, 46.204469, 46.208736, 46.207895, 46.201538, 46.215493, 46.210073, 46.213596, 46.222321],
        'longitude': [5.228741, 5.218384, 5.235174, 5.230167, 5.240093, 5.217563, 5.213489, 5.220495, 5.235634, 5.222078],
        'mois_transaction': [2, 2, 2, 3, 3, 2, 1, 1, 3, 3],
        'annee_transaction': [2018, 2018, 2018, 2019, 2019, 2018, 2020, 2020, 2020, 2021],
        'en_dessous_du_marche': [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        'nom_region': ['Auvergne-Rhône-Alpes', 'Nouvelle-Aquitaine', 'Occitanie', 'Provence-Alpes-Côte d\'Azur', 'Provence-Alpes-Côte d\'Azur', 'Nouvelle-Aquitaine', 'Île-de-France', 'Île-de-France', 'Provence-Alpes-Côte d\'Azur', 'Auvergne-Rhône-Alpes'],
        'prix_m2_moyen_mois_precedent': [1229.76, 1229.76, 1229.76, 1350.50, 1350.50, 1229.76, 1400.20, 1400.20, 1350.50, 1300.80],
        'nb_transactions_mois_precedent': [17, 17, 17, 20, 20, 17, 25, 25, 20, 23],
        'ville_demandee': [1, 1, 1, 0, 0, 1, 1, 0, 1, 1]
    }
    return pd.DataFrame(data)

# Test principal pour split_and_encode
def test_split_and_encode(df):
    columns_to_encode = ['type_batiment', 'nom_region']

    # Appeler la fonction
    X_train, X_test, y_train_regression, y_test_regression, y_train_classification, y_test_classification, encoders = split_and_encode(
        df=df,
        target_classification=CLASSIFICATION_TARGET,
        target_regression=REGRESSION_TARGET,
        columns_to_encode=columns_to_encode
    )

    # Vérifications de la taille des ensembles d'entraînement et de test
    assert X_train.shape[0] == 8  # 80% des 10 lignes dans l'ensemble d'entraînement
    assert X_test.shape[0] == 2   # 20% des 10 lignes dans l'ensemble de test

    # Vérifier que les colonnes encodées ont bien été créées dans l'ensemble d'entraînement
    assert 'type_batiment_Maison' in X_train.columns


    # Vérifier que les colonnes encodées de la région sont bien présentes dans X_train

    assert 'nom_region_Nouvelle-Aquitaine' in X_train.columns
    assert 'nom_region_Occitanie' in X_train.columns
    assert 'nom_region_Provence-Alpes-Côte d\'Azur' in X_train.columns
    assert 'nom_region_Île-de-France' in X_train.columns

    # Vérifier que les encodeurs ont bien été créés pour chaque colonne à encoder
    assert 'type_batiment' in encoders
    assert 'nom_region' in encoders

    # Vérifier la taille des cibles de régression et de classification
    assert len(y_train_regression) == 8
    assert len(y_test_regression) == 2
    assert len(y_train_classification) == 8
    assert len(y_test_classification) == 2



# Test pour une colonne invalide
def test_invalid_column(df):
    with pytest.raises(ValueError):
        split_and_encode(
            df=df,
            target_classification=CLASSIFICATION_TARGET,
            target_regression=REGRESSION_TARGET,
            columns_to_encode=['colonne_inexistante']  # Colonne invalide
        )
