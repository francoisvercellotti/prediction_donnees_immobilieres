"""
Module de tests pour la fonction `split_and_encode` du module `src.split_and_encode`.

Ce module teste le bon fonctionnement de la fonction `split_and_encode`, qui sépare un jeu
de données en ensembles d'entraînement et de test, et qui encode les variables catégorielles
spécifiées. Les tests vérifient les points suivants :
- La répartition correcte des données entre les ensembles d'entraînement et de test (80/20).
- La création correcte des colonnes encodées pour les variables catégorielles.
- La gestion des erreurs lorsqu'une colonne invalide est fournie.

Fonctions :
-----------
- test_split_and_encode() : Teste la fonction `split_and_encode` en vérifiant la répartition
  des données, l'encodage des variables catégorielles et la taille correcte des ensembles
  résultants.
- test_invalid_column() : Teste le comportement de la fonction lorsque des colonnes invalides
  sont spécifiées, en vérifiant que l'erreur appropriée (`ValueError`) est levée.

Exécution :
-----------
Si ce script est exécuté directement, il lance les tests via `pytest`.

Exemple d'exécution :
---------------------
    pytest tests/test_split_and_encode.py
"""


import os
import sys
import pytest
import pandas as pd
from settings import CLASSIFICATION_TARGET, REGRESSION_TARGET
from src.split_and_encode import split_and_encode
# Ajouter le dossier parent au chemin pour importer les paramètres
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))




@pytest.fixture
def dataframe():
    """
    Prépare un jeu de données synthétique pour les tests unitaires.
    """
    data = {
        'prix': [190000.0, 90000.0, 65000.0, 250000.0, 150000.0,
                 300000.0, 120000.0, 200000.0, 175000.0, 220000.0],
        'type_batiment': ['Appartement', 'Appartement', 'Appartement', 'Maison', 'Maison',
                          'Appartement', 'Maison', 'Appartement', 'Maison', 'Appartement'],
        'vefa': [0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
        'surface_habitable': [139, 60, 87, 120, 95, 130, 80, 100, 110, 105],
        'latitude': [46.1983, 46.1993, 46.2044, 46.2087, 46.2078,
                     46.2015, 46.2154, 46.2100, 46.2135, 46.2223],
        'longitude': [5.2287, 5.2183, 5.2351, 5.2301, 5.2400,
                      5.2175, 5.2134, 5.2204, 5.2356, 5.2220],
        'mois_transaction': [2, 2, 2, 3, 3, 2, 1, 1, 3, 3],
        'annee_transaction': [2018, 2018, 2018, 2019, 2019, 2018, 2020, 2020, 2020, 2021],
        'en_dessous_du_marche': [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        'nom_region': ['Auvergne-Rhône-Alpes', 'Nouvelle-Aquitaine', 'Occitanie',
                       'Provence-Alpes-Côte d\'Azur', 'Provence-Alpes-Côte d\'Azur',
                       'Nouvelle-Aquitaine', 'Île-de-France', 'Île-de-France',
                       'Provence-Alpes-Côte d\'Azur', 'Auvergne-Rhône-Alpes'],
        'prix_m2_moyen_mois_precedent': [1229.76, 1229.76, 1229.76, 1350.50, 1350.50,
                                         1229.76, 1400.20, 1400.20, 1350.50, 1300.80],
        'nb_transactions_mois_precedent': [17, 17, 17, 20, 20, 17, 25, 25, 20, 23],
        'ville_demandee': [1, 1, 1, 0, 0, 1, 1, 0, 1, 1]
    }
    return pd.DataFrame(data)


def test_split_and_encode(dataframe):
    """
    Teste la fonction split_and_encode avec un jeu de données synthétique.
    """
    cols_to_encode = ['type_batiment', 'nom_region']

    X_train, X_test, y_train_regression, y_test_regression, y_train_classification, y_test_classification, encoders = \
        split_and_encode(data=dataframe, target_class=CLASSIFICATION_TARGET,
                         target_reg=REGRESSION_TARGET, cols_to_encode=cols_to_encode)

    assert X_train.shape[0] == 8, "L'ensemble d'entraînement doit contenir 8 lignes"
    assert X_test.shape[0] == 2, "L'ensemble de test doit contenir 2 lignes"
    assert 'type_batiment_Maison' in X_train.columns, "Colonne 'type_batiment_Maison' manquante"

    region_columns = [
        'nom_region_Nouvelle-Aquitaine', 'nom_region_Occitanie',
        'nom_region_Provence-Alpes-Côte d\'Azur', 'nom_region_Île-de-France'
    ]
    for col in region_columns:
        assert col in X_train.columns, f"Colonne '{col}' manquante"

    assert 'type_batiment' in encoders, "Encodeur 'type_batiment' absent"
    assert 'nom_region' in encoders, "Encodeur 'nom_region' absent"

    assert len(y_train_regression) == 8, "Les cibles de régression d'entraînement doivent contenir 8 éléments"
    assert len(y_test_regression) == 2, "Les cibles de régression de test doivent contenir 2 éléments"
    assert len(y_train_classification) == 8, "Les cibles de classification d'entraînement doivent contenir 8 éléments"
    assert len(y_test_classification) == 2, "Les cibles de classification de test doivent contenir 2 éléments"


def test_invalid_column(dataframe):
    """
    Teste le comportement de split_and_encode lorsqu'une colonne invalide est fournie.
    """
    with pytest.raises(ValueError,
                       match="La colonne colonne_inexistante n'existe pas dans le DataFrame."):
        split_and_encode(data=dataframe, target_class=CLASSIFICATION_TARGET,
                         target_reg=REGRESSION_TARGET, cols_to_encode=['colonne_inexistante'])
