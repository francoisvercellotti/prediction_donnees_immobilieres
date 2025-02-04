import streamlit as st
import requests
import pandas as pd
import pickle
import mlflow.pyfunc

# Charger les encodeurs
with open("models/encoders.pickle", "rb") as f:
    encoders = pickle.load(f)

# Fonction pour récupérer les coordonnées GPS
def get_coordinates(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    response = requests.get(url, params=params)

    if response.status_code == 200 and response.json():
        data = response.json()[0]
        return float(data["lat"]), float(data["lon"])
    return None, None

# Fonction pour récupérer la région à partir des coordonnées GPS
def get_region_from_coordinates(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json"}
    response = requests.get(url, params=params)

    if response.status_code == 200 and response.json():
        address_data = response.json().get("address", {})
        return address_data.get("state", "Région inconnue")
    return "Région inconnue"

# Fonction pour charger le modèle MLflow
def load_model(model_uri):
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        st.success(f"✅ Modèle chargé depuis {model_uri}")
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

# Interface Streamlit
st.title("🏡 Prédiction de Prix Immobilier avec Adresse 📍")

# Entrée de l'URI du modèle
model_uri = st.text_input("Entrez l'URI du modèle MLflow", "runs:/<run_id>/model")
if st.button("Charger le modèle"):
    model = load_model(model_uri)
    st.session_state.model = model

# Saisie de l'adresse
address = st.text_input("📍 Entrez l'adresse du bien immobilier")

if st.button("Obtenir les coordonnées GPS"):
    latitude, longitude = get_coordinates(address)

    if latitude and longitude:
        st.success(f"📌 Coordonnées GPS : {latitude}, {longitude}")
        region = get_region_from_coordinates(latitude, longitude)
        st.success(f"🌍 Région détectée : {region}")

        # Encodage de la région
        if region in encoders["nom_region"]:
            encoded_region = encoders["nom_region"].transform([[region]])[0]
        else:
            encoded_region = [0, 0, 0, 0]  # Si la région n'est pas connue

        # Demande d'autres caractéristiques
        vefa = st.selectbox("Vente en l'état futur d'achèvement (VEFA)", [0, 1])
        surface_habitable = st.number_input("Surface habitable (m²)")
        mois_transaction = st.slider("Mois de transaction", 1, 12)
        annee_transaction = st.number_input("Année de transaction", min_value=2000, max_value=2030, step=1)
        prix_m2_moyen_mois_precedent = st.number_input("Prix moyen au m² le mois précédent")
        nb_transactions_mois_precedent = st.number_input("Nombre de transactions le mois précédent")
        ville_demandee = st.selectbox("Bien situé dans une ville demandée", [0, 1])

        # Sélection et encodage du type de bien
        type_batiment = st.selectbox("Type de bâtiment", ["Appartement", "Maison"])
        encoded_type_batiment = 1 if type_batiment == "Maison" else 0

        # Création du DataFrame final
        input_data = pd.DataFrame([{
            "vefa": vefa,
            "surface_habitable": surface_habitable,
            "latitude": latitude,
            "longitude": longitude,
            "mois_transaction": mois_transaction,
            "annee_transaction": annee_transaction,
            "prix_m2_moyen_mois_precedent": prix_m2_moyen_mois_precedent,
            "nb_transactions_mois_precedent": nb_transactions_mois_precedent,
            "ville_demandee": ville_demandee,
            "type_batiment_Maison": encoded_type_batiment,
            "nom_region_Nouvelle-Aquitaine": encoded_region[0],
            "nom_region_Occitanie": encoded_region[1],
            "nom_region_Provence-Alpes-Côte d'Azur": encoded_region[2],
            "nom_region_Île-de-France": encoded_region[3]
        }])

        st.write("📊 **Données formatées pour la prédiction** :", input_data)

        # Lancer la prédiction si le modèle est chargé
        if "model" in st.session_state:
            model = st.session_state.model
            if st.button("Effectuer la prédiction"):
                predictions = model.predict(input_data)
                st.write("💰 **Prédiction du prix du bien** :", predictions[0])
        else:
            st.warning("⚠ Veuillez charger un modèle avant de faire une prédiction.")

