import mlflow.pyfunc
import pandas as pd
import streamlit as st
import pickle
import requests
import unicodedata
import numpy as np

# Configuration de la page Streamlit
st.set_page_config(layout="wide", page_title="Prédiction Prix Immobilier", page_icon="🏠")

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .stNumberInput, .stSelectbox {
        background-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

# Charger les encodeurs sauvegardés
with open("models/encoders.pickle", "rb") as f:
    encoders = pickle.load(f)

type_batiment_encoder = encoders["type_batiment"]
region_encoder = encoders["nom_region"]

# Définir l'URI du modèle MLflow
LOGGED_MODEL_URI = "runs:/dbe54e3229dc471fbf49aac749a20477/model"

# Charger automatiquement le modèle
@st.cache_resource
def load_model():
    try:
        model = mlflow.pyfunc.load_model(LOGGED_MODEL_URI)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

def normalize_text(text):
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                  if unicodedata.category(c) != 'Mn')
    text = ' '.join(text.split())
    return text

def get_coordinates(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {'User-Agent': 'PrixImmobilierApp/1.0'}
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200 and response.json():
        data = response.json()[0]
        return float(data["lat"]), float(data["lon"])
    return None, None

def get_region_from_coordinates(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json"}
    headers = {'User-Agent': 'PrixImmobilierApp/1.0'}
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200 and response.json():
        address_data = response.json().get("address", {})
        region = address_data.get("state", "Région inconnue")
        return normalize_text(region)
    return "region inconnue"

# Régions autorisées
REGIONS_AUTORISÉES = {normalize_text(region) for region in [
    "Nouvelle-Aquitaine",
    "Occitanie",
    "Provence-Alpes-Côte d'Azur",
    "Île-de-France",
    "Auvergne-Rhône-Alpes"
]}

# Interface utilisateur Streamlit
st.title("🏠 Prédiction de Prix Immobilier")

model = load_model()

# Création des colonnes principales
col1, col2 = st.columns([1, 2])

# Initialiser les variables d'état
if 'region' not in st.session_state:
    st.session_state.region = None
if 'lat' not in st.session_state:
    st.session_state.lat = None
if 'lon' not in st.session_state:
    st.session_state.lon = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

with col1:
    st.markdown("### 📍 Localisation")
    with st.container():
        address = st.text_input("Adresse du bien", placeholder="ex: 10 Rue de Rivoli, Paris")
        if st.button("🔍 Rechercher", key="search_button"):
            if address:
                with st.spinner("Recherche de l'adresse..."):
                    st.session_state.lat, st.session_state.lon = get_coordinates(address)
                    if st.session_state.lat and st.session_state.lon:
                        region_found = get_region_from_coordinates(st.session_state.lat, st.session_state.lon)
                        normalized_region = normalize_text(region_found)

                        st.success(f"📍 Coordonnées trouvées")
                        st.info(f"🌍 Région : {region_found}")

                        if normalized_region in REGIONS_AUTORISÉES:
                            st.session_state.region = region_found
                        else:
                            st.error("❌ Région non prise en charge")
                            st.session_state.region = None
                    else:
                        st.error("❌ Adresse non trouvée")

    if st.session_state.region:
        st.markdown("### 🏡 Caractéristiques du bien")

        # Utilisation de select_box pour VEFA
        vefa = st.selectbox("VEFA",
                           options=["Non", "Oui"],
                           index=0)
        vefa = 1 if vefa == "Oui" else 0

        # Slider pour la surface
        surface_habitable = st.slider("Surface habitable (m²)",
                                    min_value=10,
                                    max_value=500,
                                    value=100,
                                    step=5)

        # Select box pour le type de bâtiment
        type_batiment_selection = st.selectbox("Type de bâtiment",
                                             options=["Appartement", "Maison"])

        # Select box pour ville demandée
        ville_demandee = st.selectbox("Ville demandée",
                                    options=["Non", "Oui"])
        ville_demandee = 1 if ville_demandee == "Oui" else 0

        # Slider pour le prix
        prix_m2_moyen_mois_precedent = st.slider("Prix moyen au m² (mois précédent)",
                                                min_value=1000,
                                                max_value=15000,
                                                value=3000,
                                                step=100)

        # Nombre de transactions avec number_input
        nb_transactions_mois_precedent = st.number_input("Transactions (mois précédent)",
                                                        min_value=0,
                                                        max_value=1000,
                                                        value=100,
                                                        step=1)

        # Date en deux colonnes
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            mois_transaction = st.selectbox("Mois", range(1, 13))
        with col_date2:
            annee_transaction = st.selectbox("Année", range(2023, 2026))

        # Bouton de prédiction
        if st.button("🎯 Effectuer la prédiction"):
            # Préparation des données
            type_batiment_encoded = type_batiment_encoder.transform([[type_batiment_selection]])[0]
            region_encoded = region_encoder.transform([[st.session_state.region]])[0]

            input_data = pd.DataFrame({
                "vefa": [np.int32(vefa)],
                "surface_habitable": [np.int32(surface_habitable)],
                "latitude": [np.float64(st.session_state.lat)],
                "longitude": [np.float64(st.session_state.lon)],
                "mois_transaction": [np.int32(mois_transaction)],
                "annee_transaction": [np.int32(annee_transaction)],
                "prix_m2_moyen_mois_precedent": [np.float64(prix_m2_moyen_mois_precedent)],
                "nb_transactions_mois_precedent": [np.int64(nb_transactions_mois_precedent)],
                "ville_demandee": [np.int64(ville_demandee)]
            })

            for col, value in zip(type_batiment_encoder.get_feature_names_out(["type_batiment"]), type_batiment_encoded):
                input_data[col] = np.int64(value)

            for col, value in zip(region_encoder.get_feature_names_out(["nom_region"]), region_encoded):
                input_data[col] = np.int64(value)

            try:
                predictions = model.predict(input_data)
                st.session_state.prediction = predictions[0]
                st.session_state.current_params = {
                    'surface': surface_habitable,
                    'type': type_batiment_selection,
                    'vefa': vefa,
                    'region': st.session_state.region,
                    'prix_m2': prix_m2_moyen_mois_precedent,
                    'ville_demandee': ville_demandee,
                    'transactions': nb_transactions_mois_precedent,
                    'mois': mois_transaction,
                    'annee': annee_transaction
                }
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {str(e)}")

with col2:
    if st.session_state.region and st.session_state.lat and st.session_state.lon and hasattr(st.session_state, 'prediction'):
        st.markdown("### 📊 Résultats de la prédiction")

        # Affichage du prix prédit
        st.markdown(f"""
        <div class="prediction-box">
            <h2>Prix estimé</h2>
            <h1 style='color: #1e88e5; font-size: 2.5em;'>{st.session_state.prediction:,.0f} €</h1>
            <p>Soit environ {(st.session_state.prediction/st.session_state.current_params['surface']):,.0f} €/m²</p>
        </div>
        """, unsafe_allow_html=True)

        # Récapitulatif des caractéristiques principales
        st.markdown("### 📋 Récapitulatif")
        recap_col1, recap_col2 = st.columns(2)

        with recap_col1:
            st.write("**Caractéristiques principales:**")
            st.write(f"- Surface: {st.session_state.current_params['surface']} m²")
            st.write(f"- Type: {st.session_state.current_params['type']}")
            st.write(f"- VEFA: {'Oui' if st.session_state.current_params['vefa'] else 'Non'}")
            st.write(f"- Date: {st.session_state.current_params['mois']}/{st.session_state.current_params['annee']}")

        with recap_col2:
            st.write("**Localisation et marché:**")
            st.write(f"- Région: {st.session_state.current_params['region']}")
            st.write(f"- Prix moyen du marché: {st.session_state.current_params['prix_m2']} €/m²")
            st.write(f"- Ville demandée: {'Oui' if st.session_state.current_params['ville_demandee'] else 'Non'}")
            st.write(f"- Transactions: {st.session_state.current_params['transactions']}")