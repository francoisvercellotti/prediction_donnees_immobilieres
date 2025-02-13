
"""
Module de prétraitement et d'évaluation de modèles de
classification pour la détection des faux billets.

Ce module comprend plusieurs fonctions destinées à :
- Encoder les variables catégorielles
- Standardiser les données
- Imputer les valeurs manquantes à l'aide de modèles de régression
- Appliquer la transformation polynomiale et la sélection de caractéristiques
- Effectuer une analyse en composantes principales (PCA) et un clustering K-Means
- Évaluer les modèles à l'aide de courbes d'apprentissage,
matrices de confusion et métriques de classification
- Gérer le déséquilibre des classes à l'aide de SMOTE

Fonctions principales :
----------------------
- `train_regression_model(X_train, y_train)`:
Entraîne un modèle de régression pour imputer les valeurs manquantes.
- `impute_missing_values(model, X)`: Impute les valeurs manquantes dans un jeu de données donné.
- `preprocessing(X_train, X_test, y_train, smote=True)`:
Effectue l'encodage, la standardisation, l'imputation et le rééquilibrage des classes.
- `evaluation(model, X_train, y_train, X_test, y_test)`:
Évalue un modèle en affichant les métriques de classification et la courbe d'apprentissage.
- `Kmean_pipeline(X_train, X_test)`: Applique PCA et K-Means pour identifier les groupes de billets.

Exemple d'utilisation :
----------------------
```python
from preprocessing_module import preprocessing, evaluation, Kmean_pipeline

X_train, X_test, y_train, y_test = preprocessing(X_train, X_test, y_train)
model = Kmean_pipeline(X_train, X_test)
evaluation(model, X_train, y_train, X_test, y_test)
```

"""
import joblib
import pandas as pd
import streamlit as st
import pickle
import requests
import unicodedata
import numpy as np
import shap
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration de la page Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Prédiction Prix Immobilier", page_icon="🏠")

# -----------------------------------------------------------------------------
# Ajout d'un style CSS personnalisé
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .prediction-box {
        background-color:rgb(24, 22, 22);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .stNumberInput, .stSelectbox {
        background-color: transparent !important;
    }
    .shap-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Chargement des encodeurs sauvegardés
# -----------------------------------------------------------------------------
with open("models/encoders.pickle", "rb") as f:
    encoders = pickle.load(f)

# On récupère les encodeurs pour les variables catégorielles
type_batiment_encoder = encoders["type_batiment"]
region_encoder = encoders["nom_region"]

# -----------------------------------------------------------------------------
# Définition du chemin du modèle MLflow
# -----------------------------------------------------------------------------
MODEL_PATH = "models/model.pkl"

# -----------------------------------------------------------------------------
# Fonctions utilitaires
# -----------------------------------------------------------------------------
def normalize_text(text):
    """
    Normalise un texte en le mettant en minuscules et en supprimant les accents.

    Paramètres
    ----------
    text : str
        Le texte à normaliser.

    Retourne
    -------
    str
        Le texte normalisé.
    """
    text = text.lower()
    # Supprime les accents en utilisant unicodedata
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    # Supprime les espaces superflus
    text = ' '.join(text.split())
    return text

def get_coordinates(address):
    """
    Récupère les coordonnées (latitude et longitude) d'une adresse via l'API Nominatim.

    Paramètres
    ----------
    address : str
        L'adresse à rechercher.

    Retourne
    -------
    tuple of float or (None, None)
        Un tuple (latitude, longitude) si l'adresse est trouvée, sinon (None, None).
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {'User-Agent': 'PrixImmobilierApp/1.0'}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200 and response.json():
        data = response.json()[0]
        return float(data["lat"]), float(data["lon"])
    return None, None

def get_region_from_coordinates(lat, lon):
    """
    Récupère la région à partir de coordonnées géographiques via l'API Nominatim.

    Paramètres
    ----------
    lat : float
        Latitude.
    lon : float
        Longitude.

    Retourne
    -------
    str
        La région (normalisée) ou "region inconnue" en cas d'échec.
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json"}
    headers = {'User-Agent': 'PrixImmobilierApp/1.0'}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200 and response.json():
        # Extraction des informations d'adresse
        address_data = response.json().get("address", {})
        region = address_data.get("state", "Région inconnue")
        return normalize_text(region)
    return "region inconnue"

# Ensemble des régions autorisées, normalisées
REGIONS_AUTORISÉES = {normalize_text(region) for region in [
    "Nouvelle-Aquitaine",
    "Occitanie",
    "Provence-Alpes-Côte d'Azur",
    "Île-de-France",
    "Auvergne-Rhône-Alpes"
]}

def shorten_feature_names(feature_names):
    """
    Raccourcit les noms des caractéristiques pour l'affichage dans le diagramme SHAP.

    Paramètres
    ----------
    feature_names : list of str
        Liste des noms de caractéristiques à transformer.

    Retourne
    -------
    list of str
        Liste des noms raccourcis.

    Remplacements effectués :
      - 'nom_region_Provence-Alpes-Côte d\'Azur' -> 'PACA'
      - 'nom_region_Nouvelle-Aquitaine' -> 'Nouvelle-Aquit.'
      - 'nom_region_Ile-de-France' -> 'IDF'
      - 'prix_m2_moyen_mois_precedent' -> 'prix_m2_moy'
      - 'nb_transactions_mois_precedent' -> 'nb_trans'
      - 'surface_habitable' -> 'surface'
      - 'type_batiment' -> 'type_bat'
      - 'ville_demandee' -> 'ville'
      - 'mois_transaction' -> 'mois'
      - 'annee_transaction' -> 'annee'
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
    # Pour chaque nom, appliquer tous les remplacements
    for name in feature_names:
        for old, new in replacements.items():
            name = name.replace(old, new)
        shortened_names.append(name)
    return shortened_names

def plot_regression_predictions(y_true, y_pred, title, user_pred=None):
    """
    Crée un graphique comparant les prédictions aux valeurs réelles, avec conversion en k€.

    Les valeurs des axes sont converties en milliers d'euros (k€) en divisant par 1000.
    De plus, sur la ligne idéale (y=x), une croix (marqueur 'X' en rouge) est affichée pour
    indiquer la prédiction utilisateur après avoir multiplié cette valeur par 10.

    Paramètres
    ----------
    y_true : array-like
        Valeurs réelles.
    y_pred : array-like
        Prédictions générées par le modèle.
    title : str
        Titre du graphique.
    filename : str
        Nom du fichier dans lequel enregistrer le graphique.
    output_dir : str
        Répertoire où sauvegarder le graphique.
    user_pred : float, optionnel
        Prédiction utilisateur qui sera mise en évidence sur la ligne idéale.

    Retourne
    -------
    matplotlib.figure.Figure
        La figure du graphique de régression générée.
    """
    plt.figure(figsize=(8, 6))
    # Conversion des valeurs en milliers d'euros (k€) pour les axes
    conversion_factor = 10000.0
    y_true_k = y_true / conversion_factor
    y_pred_k = y_pred / conversion_factor

    # Tracer les points de données
    plt.scatter(y_true_k, y_pred_k, alpha=0.5, edgecolors="k", s=30)

    # Tracer la ligne idéale (y=x)
    x_vals = [y_true_k.min(), y_true_k.max()]
    plt.plot(x_vals, x_vals, color="red", linestyle="--", label="Idéal")

    # Si une prédiction utilisateur est fournie, ajuster sa valeur et la marquer
    if user_pred is not None:
        # Multiplier par 10 avant conversion pour la mise en évidence
        user_pred_adjusted = user_pred * 10
        user_pred_k = user_pred_adjusted / conversion_factor
        plt.plot(user_pred_k, user_pred_k, marker='X', markersize=12, markeredgewidth=3,
                 color='red', label="Prédiction actuelle")

    plt.xlabel("Valeurs réelles (k€)")
    plt.ylabel("Prédictions (k€)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    plt.close(fig)
    return fig

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
    show_plots: bool = True,
    plot_reg: bool = True,
):
    """
        Prédit le prix d'un bien immobilier en fonction de plusieurs caractéristiques et génère des visualisations explicatives.

        Paramètres :
        -----------
        vefa : bool
            Indique si le bien est vendu en état futur d'achèvement (VEFA).
        surface_habitable : int
            Surface habitable du bien en mètres carrés.
        latitude : float
            Latitude de l'emplacement du bien.
        longitude : float
            Longitude de l'emplacement du bien.
        mois_transaction : int
            Mois de la transaction (1-12).
        annee_transaction : int
            Année de la transaction.
        prix_m2_moyen_mois_precedent : float
            Prix moyen du mètre carré dans la zone géographique le mois précédent.
        nb_transactions_mois_precedent : int
            Nombre de transactions immobilières enregistrées le mois précédent.
        ville_demandee : bool
            Indique si la ville est considérée comme demandée (1 pour oui, 0 pour non).
        type_batiment : str
            Type de bâtiment (ex : "Appartement", "Maison").
        region : str
            Région où se situe le bien immobilier.
        show_plots : bool, optionnel (par défaut True)
            Indique si l'explication SHAP doit être générée et retournée.
        plot_reg : bool, optionnel (par défaut True)
            Indique si un graphique de régression doit être généré.

        Retours :
        --------
        prediction : float
            Prix prédit du bien immobilier.
        shap_fig : matplotlib.figure.Figure ou None
            Figure SHAP expliquant la contribution des variables à la prédiction, ou None si `show_plots` est False.
        regression_fig : matplotlib.figure.Figure ou None
            Graphique de régression comparant les prédictions et les valeurs réelles, ou None si `plot_reg` est False.

        Exceptions :
        -----------
        En cas d'erreur lors du chargement des fichiers, de l'encodage des variables ou de la prédiction, une exception est levée.

        Notes :
        ------
        - Le modèle est chargé depuis `MODEL_PATH` via `joblib.load`.
        - Les encodeurs pour `type_batiment` et `region` sont appliqués pour transformer ces variables catégorielles en numériques.
        - L'alignement des colonnes est effectué pour garantir la compatibilité avec le modèle entraîné.
        - L'explication SHAP est générée uniquement si `show_plots` est activé.
        - Un graphique de régression est généré si `plot_reg` est activé.
        """

    try:
        X_train = pd.read_parquet("data/interim/preprocessed/X_train.parquet")
        train_columns = X_train.columns.tolist()

        # Load model using joblib instead of MLflow
        loaded_model = joblib.load(MODEL_PATH)

        type_batiment_encoded = type_batiment_encoder.transform([[type_batiment]])[0]
        region_encoded = region_encoder.transform([[region]])[0]

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

        for col, value in zip(type_batiment_encoder.get_feature_names_out(["type_batiment"]), type_batiment_encoded):
            input_data[col] = np.int64(value)
        for col, value in zip(region_encoder.get_feature_names_out(["nom_region"]), region_encoded):
            input_data[col] = np.int64(value)

        input_data_aligned = pd.DataFrame(columns=train_columns)
        for col in train_columns:
            if col in input_data.columns:
                input_data_aligned[col] = input_data[col]
            else:
                input_data_aligned[col] = 0

        prediction = loaded_model.predict(input_data_aligned)

        shap_fig = None
        if show_plots:
            explainer = shap.Explainer(loaded_model, X_train)
            shap_values = explainer(input_data_aligned)
            shortened_names = shorten_feature_names(input_data_aligned.columns)
            plt.figure(figsize=(14, 10))
            shap_values.feature_names = shortened_names
            shap.plots.waterfall(
                shap_values[0],
                show=False,
                max_display=6,
            )
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.subplots_adjust(left=0.5)
            plt.title("Impact des caractéristiques sur la prédiction", fontsize=16, pad=30)
            shap_fig = plt.gcf()
            plt.close(shap_fig)

        regression_fig = None
        if plot_reg:
            X_test = pd.read_parquet("data/interim/preprocessed/X_test.parquet")
            y_test = pd.read_parquet("data/interim/preprocessed/y_test.parquet")
            if isinstance(y_test, pd.DataFrame):
                y_test = y_test.iloc[:, 0]
            y_test_pred = loaded_model.predict(X_test)
            regression_fig = plot_regression_predictions(
                y_true=y_test,
                y_pred=y_test_pred,
                title="Tendances des erreurs du modèle (en k€)",
                user_pred=prediction[0]
            )

        return prediction[0], shap_fig, regression_fig

    except Exception as e:
        raise Exception(f"Erreur lors de la prédiction : {str(e)}")


# -----------------------------------------------------------------------------
# Interface utilisateur Streamlit
# -----------------------------------------------------------------------------

# Centrer le titre avec du HTML et du CSS
st.markdown(
    "<h1 style='text-align: center; font-size: 60px;'>🏠 Prédiction de Prix Immobilier</h1>",
    unsafe_allow_html=True
)

# Création de deux colonnes pour organiser l'interface (entrée et affichage)
col1, col2 = st.columns([1, 2])

# Initialisation des variables d'état dans la session Streamlit
if 'region' not in st.session_state:
    st.session_state.region = None
if 'lat' not in st.session_state:
    st.session_state.lat = None
if 'lon' not in st.session_state:
    st.session_state.lon = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'shap_fig' not in st.session_state:
    st.session_state.shap_fig = None
if 'regression_fig' not in st.session_state:
    st.session_state.regression_fig = None

# -----------------------------------------------------------------------------
# Colonne 1 : Saisie des paramètres par l'utilisateur
# -----------------------------------------------------------------------------
with col1:
    st.markdown("### 📍 Localisation")
    with st.container():
        # Saisie de l'adresse du bien
        address = st.text_input("Adresse du bien", placeholder="ex: 10 Rue de Rivoli, Paris")
        if st.button("🔍 Rechercher", key="search_button"):
            if address:
                with st.spinner("Recherche de l'adresse..."):
                    # Récupération des coordonnées à partir de l'adresse
                    st.session_state.lat, st.session_state.lon = get_coordinates(address)
                    if st.session_state.lat and st.session_state.lon:
                        # Récupération et normalisation de la région à partir des coordonnées
                        region_found = get_region_from_coordinates(st.session_state.lat, st.session_state.lon)
                        normalized_region = normalize_text(region_found)
                        st.success("📍 Coordonnées trouvées")
                        st.info(f"🌍 Région : {region_found}")
                        # Vérifier si la région est autorisée
                        if normalized_region in REGIONS_AUTORISÉES:
                            st.session_state.region = region_found
                        else:
                            st.error("❌ Région non prise en charge")
                            st.session_state.region = None
                    else:
                        st.error("❌ Adresse non trouvée")
    # Si une région valide est sélectionnée, afficher les autres paramètres
    if st.session_state.region:
        st.markdown("### 🏡 Caractéristiques du bien")
        # Sélection de la VEFA (Oui/Non)
        vefa = st.selectbox("VEFA", options=["Non", "Oui"], index=0)
        vefa = 1 if vefa == "Oui" else 0
        # Saisie de la surface habitable
        surface_habitable = st.slider("Surface habitable (m²)",
                                      min_value=10, max_value=500,
                                      value=100, step=5)
        # Sélection du type de bâtiment
        type_batiment_selection = st.selectbox("Type de bâtiment", options=["Appartement", "Maison"])
        # Sélection de la ville demandée
        ville_demandee = st.selectbox("Ville demandée", options=["Non", "Oui"])
        ville_demandee = 1 if ville_demandee == "Oui" else 0
        # Saisie du prix moyen au m² (mois précédent)
        prix_m2_moyen_mois_precedent = st.slider("Prix moyen au m² (mois précédent)",
                                                 min_value=1000, max_value=15000,
                                                 value=3000, step=100)
        # Saisie du nombre de transactions (mois précédent)
        nb_transactions_mois_precedent = st.slider("Transactions (mois précédent)",
                                                   min_value=0, max_value=1000,
                                                   value=20, step=1)
        # Sélection de la date via deux colonnes
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            mois_transaction = st.selectbox("Mois", range(1, 13))
        with col_date2:
            annee_transaction = st.selectbox("Année", range(2023, 2026))
        # Bouton pour lancer la prédiction
        if st.button("🎯 Effectuer la prédiction"):
            try:
                prediction, shap_fig, regression_fig = predict_prix_immobilier(
                    vefa=bool(vefa),
                    surface_habitable=surface_habitable,
                    latitude=st.session_state.lat,
                    longitude=st.session_state.lon,
                    mois_transaction=mois_transaction,
                    annee_transaction=annee_transaction,
                    prix_m2_moyen_mois_precedent=prix_m2_moyen_mois_precedent,
                    nb_transactions_mois_precedent=nb_transactions_mois_precedent,
                    ville_demandee=bool(ville_demandee),
                    type_batiment=type_batiment_selection,
                    region=st.session_state.region,
                    show_plots=True,
                    plot_reg=True
                )
                st.session_state.prediction = prediction
                st.session_state.shap_fig = shap_fig
                st.session_state.regression_fig = regression_fig
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

# -----------------------------------------------------------------------------
# Colonne 2 : Affichage des résultats de la prédiction
# -----------------------------------------------------------------------------
with col2:
    if st.session_state.region and st.session_state.lat and st.session_state.lon and st.session_state.prediction is not None:
        st.markdown("### 📊 Résultats de la prédiction")
        # Affichage du prix estimé
        st.markdown(f"""
        <div class="prediction-box">
            <h2>Prix estimé</h2>
            <h1 style='color: #1e88e5; font-size: 2.5em;'>{st.session_state.prediction:,.0f} €</h1>
            <p>Soit environ {(st.session_state.prediction/st.session_state.current_params['surface']):,.0f} €/m²</p>
        </div>
        """, unsafe_allow_html=True)
        # Affichage d'un récapitulatif des caractéristiques saisies
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
        # Affichage du diagramme SHAP si disponible
        if st.session_state.shap_fig is not None:
            st.markdown("### 🔍 Explication de la prédiction (SHAP)")
            st.pyplot(st.session_state.shap_fig)
        # Affichage du graphique de régression si disponible
        if st.session_state.regression_fig is not None:
            st.markdown("### 📈 Tendances des erreurs du modèle (en k€)")
            st.pyplot(st.session_state.regression_fig)