import mlflow.pyfunc
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from mlflow.pyfunc import PyFuncModel

def load_encoders(encoder_path):
    """
    Charge les encoders depuis un fichier pickle
    """
    with open(encoder_path, "rb") as f:
        return pickle.load(f)

def shorten_feature_names(feature_names):
    """
    Raccourcit les noms des caractéristiques pour l'affichage SHAP
    """
    replacements = {
        'nom_region_Provence-Alpes-Côte d\'Azur': 'PACA',
        'nom_region_Nouvelle-Aquitaine': 'Nouvelle-Aquit.',
        'nom_region_Ile-de-France': 'IDF',
        'prix_m2_moyen_mois_precedent': 'prix_m2_moy',
        'nb_transactions_mois_precedent': 'nb_trans',
        'surface_habitable': 'surface',
        'type_batiment': 'type_bat',
        'ville_demandee': 'ville',
        'mois_transaction': 'mois',
        'annee_transaction': 'annee'
    }

    shortened_names = []
    for name in feature_names:
        for old, new in replacements.items():
            name = name.replace(old, new)
        shortened_names.append(name)
    return shortened_names

def predict_prix_immobilier(
    vefa: bool,
    surface_habitable: int,
    latitude: float,
    longitude: float,
    mois_transaction: int,
    annee_transaction: int,
    prix_m2_moyen_mois_precedent: float,
    nb_transactions_mois_precedent: int,
    ville_demandee: bool,
    type_batiment: str,
    region: str,
    model_uri: str = "../mlruns/203965422485053459/ede52d5f729548649d6f25ec255a2f1e/artifacts/model",
    encoder_path: str = "../models/encoders.pickle",
    train_data_path: str = "../data/interim/preprocessed/X_train.parquet",
    show_plots: bool = True
    ):
    """
    Prédit le prix d'un bien immobilier et affiche les détails de la prédiction
    """
    try:
        # Chargement des données d'entraînement
        X_train = pd.read_parquet(train_data_path)
        train_columns = X_train.columns.tolist()

        # Chargement des encodeurs
        encoders = load_encoders(encoder_path)
        type_batiment_encoder = encoders["type_batiment"]
        region_encoder = encoders["nom_region"]

        # Chargement du modèle
        loaded_model = mlflow.sklearn.load_model(model_uri)

        # Encodage des variables catégorielles
        type_batiment_encoded = type_batiment_encoder.transform([[type_batiment]])[0]
        region_encoded = region_encoder.transform([[region]])[0]

        # Création du DataFrame d'entrée
        input_data = pd.DataFrame({
            "vefa": [np.int32(vefa)],
            "surface_habitable": [np.int32(surface_habitable)],
            "latitude": [np.float64(latitude)],
            "longitude": [np.float64(longitude)],
            "mois_transaction": [np.int32(mois_transaction)],
            "annee_transaction": [np.int32(annee_transaction)],
            "prix_m2_moyen_mois_precedent": [np.float64(prix_m2_moyen_mois_precedent)],
            "nb_transactions_mois_precedent": [np.int64(nb_transactions_mois_precedent)],
            "ville_demandee": [np.int64(ville_demandee)]
        })

        # Ajout des colonnes encodées pour le type de bâtiment
        for col, value in zip(type_batiment_encoder.get_feature_names_out(["type_batiment"]), type_batiment_encoded):
            input_data[col] = np.int64(value)

        # Ajout des colonnes encodées pour la région
        for col, value in zip(region_encoder.get_feature_names_out(["nom_region"]), region_encoded):
            input_data[col] = np.int64(value)

        # Affichage du DataFrame
        print("\nDonnées d'entrée :")
        print(input_data.to_string())

        # S'assurer que input_data a les mêmes colonnes que X_train dans le même ordre
        input_data_aligned = pd.DataFrame(columns=train_columns)
        for col in train_columns:
            if col in input_data.columns:
                input_data_aligned[col] = input_data[col]
            else:
                input_data_aligned[col] = 0

        # Prédiction
        prediction = loaded_model.predict(input_data_aligned)

        if show_plots:
            # Création de l'explainer SHAP avec les données d'entraînement
            explainer = shap.Explainer(loaded_model, X_train)

            # Calcul des valeurs SHAP pour les données d'entrée
            shap_values = explainer(input_data_aligned)

            # Raccourcir les noms des caractéristiques pour l'affichage
            shortened_names = shorten_feature_names(input_data_aligned.columns)

            # Configuration de la figure avec un écart plus important
            plt.figure(figsize=(14, 10))  # Augmenter la taille de la figure

            # Création du waterfall plot avec les noms raccourcis
            shap_values.feature_names = shortened_names
            shap.plots.waterfall(
                shap_values[0],
                show=False,
                max_display=15
            )

            # Ajustement du layout pour augmenter l'espace entre le texte et les valeurs
            plt.subplots_adjust(left=0.5)  # Augmente l'écart entre les noms et le diagramme
            plt.title("Impact des caractéristiques sur la prédiction", fontsize=16, pad=30)
            plt.show()


        return prediction[0]

    except Exception as e:
        raise Exception(f"Erreur lors de la prédiction : {str(e)}")

# Exemple d'utilisation
if __name__ == "__main__":
    params = {
        "vefa": False,
        "surface_habitable": 20,
        "latitude": 48.8566,
        "longitude": 2.3522,
        "mois_transaction": 3,
        "annee_transaction": 2024,
        "prix_m2_moyen_mois_precedent": 6000,
        "nb_transactions_mois_precedent": 50,
        "ville_demandee": True,
        "type_batiment": "Appartement",
        "region": "Île-de-France"
    }

    try:
        prix_predit = predict_prix_immobilier(**params)
        print(f"\nPrix prédit : {prix_predit:,.2f} €")
    except Exception as e:
        print(f"Erreur : {e}")