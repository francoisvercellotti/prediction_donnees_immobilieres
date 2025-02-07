"""
Module de prétraitement des données immobilières.

Ce module contient des fonctions pour calculer des caractéristiques
des villes et des prix au m²,
effectuer des agrégations et des transformations de données,
et prétraiter les données pour une analyse ultérieure.
Les données sont ensuite sauvegardées dans un fichier parquet pour un usage futur.

Fonctions disponibles :
- compute_city_features : Calcul des caractéristiques liées
aux villes (ratio de transaction et demandée).
- compute_features_price_per_m2 : Calcul des caractéristiques dérivées pour les prix au m².
- feature_enginneering : Fonction principale de prétraitement
des données (calcul et fusion des caractéristiques).
"""

import logging
import os
import pandas as pd

# S'assurer que le répertoire 'data' existe
os.makedirs('data', exist_ok=True)

# Configuration du journal de bord (logging)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Niveau DEBUG pour capter plus d'informations

# Créer un handler pour afficher les logs dans la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Afficher INFO dans la console

# Créer un format commun pour les logs
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Ajouter les handlers au logger
logger.addHandler(console_handler)

# Fonction de calcul des caractéristiques de la ville
def compute_city_features(
    transaction_per_city: pd.DataFrame,
    aggregation_columns: list = ['nom_departement', 'ville', 'id_ville',
                                 'annee_transaction', 'mois_transaction',
                                 'nb_transactions_mois'],
    threshold: float = 0.8
):
    """
    Calcule les caractéristiques des villes en fonction du ratio
    de transactions et identifie les villes demandées.

    Cette fonction agrège les données des transactions par ville,
    puis calcule un ratio de transactions et
    attribue une étiquette 'ville_demandee' en fonction d'un seuil spécifié.

    Args:
        transaction_per_city (pd.DataFrame): DataFrame contenant
        les informations de transaction par ville.
        aggregation_columns (list, optional): Colonnes utilisées pour l'agrégation.
        Par défaut, c'est une liste des principales variables.
        threshold (float, optional): Seuil pour déterminer
        si une ville est demandée (par défaut 0.8).

    Returns:
        pd.DataFrame: DataFrame avec les caractéristiques calculées
        par ville, y compris le ratio et la ville demandée.
    """
    logging.info("Début du calcul des caractéristiques par ville.")
    try:
        transaction_per_city = transaction_per_city.groupby(aggregation_columns).agg(
            prix_m2_moyen=('prix_m2_moyen', 'mean')).reset_index()

        nb_transactions_departement = transaction_per_city.groupby(
            ['nom_departement', 'annee_transaction', 'mois_transaction']
        )['nb_transactions_mois'].sum().reset_index().rename(columns={
            'nb_transactions_mois': 'nb_transactions_departement'})

        transaction_per_city = transaction_per_city.merge(
            nb_transactions_departement, on=['nom_departement', 'annee_transaction',
                                             'mois_transaction'], how='outer'
        )

        transaction_per_city['ratio_transaction'] = (transaction_per_city['nb_transactions_mois'] /
                                                     transaction_per_city['nb_transactions_departement']) * 100

        seuil_tension = transaction_per_city['ratio_transaction'].quantile(threshold)

        transaction_per_city['ville_demandee'] = transaction_per_city['ratio_transaction'].apply(
            lambda x: 1 if x > seuil_tension else 0
        )

        logging.info("Caractéristiques par ville calculées avec succès.")
        return transaction_per_city
    except Exception as e:
        logging.error(f"Erreur lors du calcul des caractéristiques par ville : {e}")
        raise

# Fonction pour calculer les caractéristiques du prix par m2
def compute_features_price_per_m2(
    average_per_month_per_city: pd.DataFrame,
    sort_columns: list = ["nom_departement", "ville", 'id_ville',
                          "annee_transaction", "mois_transaction"],
    aggregation_columns: list = ["nom_departement", "ville", "id_ville"]
):
    """
    Calcule les caractéristiques dérivées des prix au m² par mois et par ville.

    Cette fonction ajoute des colonnes pour les prix au m² du mois précédent
    et le nombre de transactions pour le mois précédent.

    Args:
        average_per_month_per_city (pd.DataFrame): DataFrame contenant
        les moyennes des prix au m² par mois et par ville.
        sort_columns (list, optional): Colonnes pour trier les données avant l'agrégation.
        aggregation_columns (list, optional): Colonnes pour l'agrégation des données.

    Returns:
        pd.DataFrame: DataFrame avec les caractéristiques dérivées des
        prix au m² et du nombre de transactions des mois précédents.
    """
    logging.info("Début du calcul des caractéristiques dérivées pour les prix au m².")
    try:
        average_per_month_per_city = average_per_month_per_city.sort_values(by=sort_columns)

        average_per_month_per_city["prix_m2_moyen_mois_precedent"] = (
            average_per_month_per_city.groupby(aggregation_columns)["prix_m2_moyen"].shift()
        )
        average_per_month_per_city["nb_transactions_mois_precedent"] = (
            average_per_month_per_city.groupby(aggregation_columns)['nb_transactions_mois'].shift()
        )

        average_per_month_per_city = average_per_month_per_city.dropna()
        average_per_month_per_city["nb_transactions_mois_precedent"] =\
            average_per_month_per_city["nb_transactions_mois_precedent"].astype(int)

        logging.info("Caractéristiques dérivées calculées avec succès.")
        return average_per_month_per_city
    except Exception as e:
        logging.error(f"Erreur lors du calcul des caractéristiques dérivées : {e}")
        raise

# Fonction de prétraitement principal
def feature_enginneering(df) -> pd.DataFrame:
    """
    Fonction principale de prétraitement des données,
    incluant le calcul et la fusion des caractéristiques.

    Cette fonction calcule les caractéristiques par ville,
    puis les caractéristiques liées aux prix au m²,
    et les fusionne avec les données d'entrée.
    Elle sauvegarde également le DataFrame préparé dans un fichier parquet.

    Args:
        df (pd.DataFrame): DataFrame contenant les données brutes des transactions immobilières.

    Returns:
        pd.DataFrame: DataFrame préparé avec les nouvelles caractéristiques calculées et fusionnées.
    """
    logging.info("Début du prétraitement des données.")
    try:
        df_city = df.groupby(
            ['nom_departement', 'ville', 'id_ville', 'annee_transaction',
             'mois_transaction', 'nb_transactions_mois']
        ).agg(prix_m2_moyen=('prix_m2_moyen', 'mean')).round(2).reset_index()

        df_city = compute_city_features(transaction_per_city=df_city, threshold=0.8)

        if df_city is None or df_city.empty:
            logging.error("Le DataFrame df_city est vide après\
                          le calcul des caractéristiques de la ville.")
            raise ValueError("Le calcul des caractéristiques de la ville a échoué.")

        df_city_agg = compute_features_price_per_m2(df_city,
                                                    aggregation_columns=["nom_departement",
                                                                         "ville",
                                                                         "id_ville"])

        if df_city_agg is None or df_city_agg.empty:
            logging.error("Le DataFrame df_city_agg est vide après \
                le calcul des caractéristiques dérivées.")
            raise ValueError("Le calcul des caractéristiques dérivées a échoué.")

        required_columns = ['nom_departement', 'ville', 'id_ville',
                            'annee_transaction',
                            'mois_transaction', 'prix_m2_moyen_mois_precedent',
                            'nb_transactions_mois_precedent', 'ville_demandee']
        if not all(col in df_city_agg.columns for col in required_columns):
            missing_cols = set(required_columns) - set(df_city_agg.columns)
            logging.error(f"Colonnes manquantes pour la fusion : {missing_cols}")
            raise ValueError("Colonnes nécessaires manquantes dans df_city_agg.")

        df_prepared = df.merge(
            df_city_agg[required_columns],
            on=['nom_departement', 'ville', 'id_ville',
                'annee_transaction', 'mois_transaction'], how='inner'
        )

        df_prepared = df_prepared.drop(['nom_departement', 'id_ville', 'ville', 'prix_m2',
                                        'prix_m2_moyen', 'nb_transactions_mois',
                                        'n_pieces'], axis=1)

        # Affichage des dimensions et des 5 premières lignes
        logging.info(f"Dimensions du DataFrame préparé : {df_prepared.shape}")
        logging.info(f"Les 5 premières lignes du DataFrame préparé :\n{df_prepared.head()}")

        # Vérification des permissions et création du répertoire si nécessaire
        OUTPUT_FILE = 'data/interim/prepared_dataset.parquet'
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

        print(f"Sauvegarde du fichier sous '{OUTPUT_FILE}'.")
        logging.info(f"Le fichier a été sauvegardé sous '{OUTPUT_FILE}'.")

        # Sauvegarde du fichier
        df_prepared.to_parquet(OUTPUT_FILE, index=False)

        logging.info(f"Le fichier a été sauvegardé sous '{OUTPUT_FILE}'.")

        return df_prepared
    except Exception as e:
        logging.error(f"Erreur dans la fonction de prétraitement : {e}")
        raise

if __name__ == "__main__":
    try:
        data = pd.read_parquet('data/loaded/loaded_dataset.parquet')
        feature_enginneering(data)
    except Exception as e:
        logging.critical(f"Erreur critique lors de l'exécution du script principal : {e}")
