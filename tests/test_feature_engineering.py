import os
import pytest
import pandas as pd
from io import StringIO
from src.feature_engineering import feature_enginneering, compute_city_features, compute_features_price_per_m2

# Exemple de données pour les tests
@pytest.fixture
def sample_data():
    data = """
    prix,id_ville,ville,type_batiment,vefa,n_pieces,surface_habitable,latitude,longitude,mois_transaction,annee_transaction,prix_m2,prix_m2_moyen,nb_transactions_mois,en_dessous_du_marche,nom_departement,nom_region
    98000.0,53,BOURG-EN-BRESSE,Appartement,0,3,70,46.204952,5.225964,1,2018,1400.000000,1229.764459,17,0,Ain,Auvergne-Rhône-Alpes
    225000.0,143,DIVONNE-LES-BAINS,Appartement,0,2,56,46.355345,6.137488,1,2018,4017.857143,4512.419826,9,1,Ain,Auvergne-Rhône-Alpes
    67000.0,53,BOURG-EN-BRESSE,Appartement,0,1,45,46.201122,5.237210,1,2018,1488.888889,1229.764459,17,0,Ain,Auvergne-Rhône-Alpes
    110000.0,53,BOURG-EN-BRESSE,Appartement,0,3,70,46.204952,5.225964,2,2018,1571.428571,1229.764459,12,0,Ain,Auvergne-Rhône-Alpes
    220000.0,143,DIVONNE-LES-BAINS,Appartement,0,2,56,46.355345,6.137488,2,2018,3928.571429,4512.419826,8,1,Ain,Auvergne-Rhône-Alpes
    65000.0,53,BOURG-EN-BRESSE,Appartement,0,1,45,46.201122,5.237210,2,2018,1444.444444,1229.764459,15,0,Ain,Auvergne-Rhône-Alpes
    100000.0,143,DIVONNE-LES-BAINS,Appartement,0,2,56,46.355345,6.137488,3,2018,3571.428571,4512.419826,11,1,Ain,Auvergne-Rhône-Alpes
    195000.0,53,BOURG-EN-BRESSE,Appartement,0,3,70,46.204952,5.225964,3,2018,2785.714286,1229.764459,14,0,Ain,Auvergne-Rhône-Alpes
    69000.0,143,DIVONNE-LES-BAINS,Appartement,0,1,45,46.355345,6.137488,3,2018,1533.333333,4512.419826,10,1,Ain,Auvergne-Rhône-Alpes
    """
    return pd.read_csv(StringIO(data), delimiter=",")

# Test pour la fonction compute_city_features
def test_compute_city_features(sample_data):
    # Appliquer compute_city_features
    df_city = compute_city_features(sample_data)

    # Vérifier que les colonnes attendues existent
    assert 'prix_m2_moyen' in df_city.columns
    assert 'ratio_transaction' in df_city.columns
    assert 'ville_demandee' in df_city.columns

    # Vérifier le format des données retournées
    assert df_city['prix_m2_moyen'].dtype == float
    assert df_city['ratio_transaction'].dtype == float
    assert df_city['ville_demandee'].dtype == int

# Test pour la fonction compute_features_price_per_m2
def test_compute_features_price_per_m2(sample_data):
    # Calculer les caractéristiques du prix au m² par ville et mois
    df_city_agg = compute_features_price_per_m2(sample_data)

    # Vérifier que les colonnes attendues existent
    assert 'prix_m2_moyen_mois_precedent' in df_city_agg.columns
    assert 'nb_transactions_mois_precedent' in df_city_agg.columns

    # Vérifier le format des données retournées
    assert df_city_agg['prix_m2_moyen_mois_precedent'].dtype == float
    assert df_city_agg['nb_transactions_mois_precedent'].dtype == int

# Test pour la fonction feature_enginneering
def test_feature_enginneering(sample_data):
    # Vérifier si le répertoire data/interim existe, sinon le créer
    if not os.path.exists('data/interim'):
        os.makedirs('data/interim')

    # Appliquer le prétraitement sur l'exemple de données
    df_prepared = feature_enginneering(sample_data)

    # Vérifier que le DataFrame résultant contient les colonnes attendues après prétraitement
    required_columns = ['annee_transaction', 'mois_transaction', 'prix_m2_moyen_mois_precedent',
                        'nb_transactions_mois_precedent', 'ville_demandee']

    for col in required_columns:
        assert col in df_prepared.columns

    # Vérifier que les colonnes inutiles ont été supprimées
    assert 'nom_departement' not in df_prepared.columns
    assert 'id_ville' not in df_prepared.columns
    assert 'ville' not in df_prepared.columns
    assert 'prix_m2' not in df_prepared.columns
    assert 'prix_m2_moyen' not in df_prepared.columns
    assert 'nb_transactions_mois' not in df_prepared.columns
    assert 'n_pieces' not in df_prepared.columns

    # Vérifier que les résultats ne sont pas vides
    assert not df_prepared.empty

# Exécution des tests
if __name__ == "__main__":
    pytest.main()
