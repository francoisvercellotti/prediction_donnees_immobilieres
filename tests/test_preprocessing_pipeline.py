

"""
Module de tests pour la fonction `preprocess_data` du module `src.preprocessing_pipeline`.

Ce module teste le bon fonctionnement du pipeline de prétraitement des données, notamment :
- La séparation des données en ensembles d'entraînement et de test.
- L'encodage des variables catégorielles spécifiées.
- La vérification que les ensembles résultants ne sont pas vides
et conservent bien les dimensions initiales.

Fonctions :
-----------
- test_preprocess_data() : Teste la fonction `preprocess_data`
en vérifiant la présence des données transformées
  et des colonnes encodées, sans créer de fichiers grâce à `unittest.mock.patch`.

Exécution :
-----------
Si ce script est exécuté directement, il lance les tests via `pytest`.

Exemple d'exécution :
---------------------
    pytest test_preprocessing_pipeline.py

"""
from unittest.mock import patch
import pytest
import pandas as pd
from src.preprocessing_pipeline import preprocess_data

# Données d'exemple (identiques à votre précédent exemple)
data = {
    'prix': [190000.0, 90000.0, 65000.0, 250000.0, 150000.0, 300000.0,
             120000.0, 200000.0, 175000.0, 220000.0],
    'type_batiment': ['Appartement', 'Appartement', 'Appartement', 'Maison',
                      'Maison', 'Appartement', 'Maison', 'Appartement', 'Maison', 'Appartement'],
    'vefa': [0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
    'surface_habitable': [139, 60, 87, 120, 95, 130, 80, 100, 110, 105],
    'latitude': [46.198306, 46.199362, 46.204469, 46.208736, 46.207895, 46.201538,
                 46.215493, 46.210073, 46.213596, 46.222321],
    'longitude': [5.228741, 5.218384, 5.235174, 5.230167, 5.240093,
                  5.217563, 5.213489, 5.220495, 5.235634, 5.222078],
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

df = pd.DataFrame(data)
columns_to_encode = ['type_batiment', 'nom_region']

def test_preprocess_data():
    """Test du pipeline de prétraitement des données sans créer de fichiers"""
    # Utiliser patch pour éviter la création de fichiers
    with patch('joblib.dump'), patch('os.makedirs'):
        # Prétraiter les données
        X_train, X_test, y_train_reg, y_test_reg,y_train_cls, y_test_cls = preprocess_data(df, columns_to_encode)

        # Vérifier que les DataFrames ne sont pas vides
        assert not X_train.empty, "X_train est vide"
        assert not X_test.empty, "X_test est vide"
        assert not y_train_reg.empty, "y_train_reg est vide"
        assert not y_test_reg.empty, "y_test_reg est vide"
        assert not y_train_cls.empty, "y_train_cls est vide"
        assert not y_test_cls.empty, "y_test_cls est vide"

        # Vérifier les colonnes encodées
        for col in columns_to_encode:
            encoded_columns = [c for c in X_train.columns if c.startswith(f"{col}_")]
            assert len(encoded_columns) > 0, f"Colonnes encodées pour {col} manquantes"

        # Vérification supplémentaire des dimensions
        assert len(X_train) + len(X_test) == len(df), "La répartition des données est incorrecte"

if __name__ == "__main__":
    pytest.main()
