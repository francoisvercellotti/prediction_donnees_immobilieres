import pandas as pd
import logging
import os

# Configuration du journal de bord (logging)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Niveau DEBUG pour capter plus d'informations

# Créer un handler pour la sortie dans un fichier
file_handler = logging.FileHandler('data/preprocessing_log.log')
file_handler.setLevel(logging.DEBUG)

# Créer un handler pour afficher les logs dans la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Afficher INFO dans la console

# Créer un format commun pour les logs
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Ajouter les handlers au logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Fonction de calcul des caractéristiques de la ville
def compute_city_features(
    transaction_per_city: pd.DataFrame,
    aggregation_columns: list = ['nom_departement', 'ville', 'id_ville', 'annee_transaction', 'mois_transaction', 'nb_transactions_mois'],
    threshold: float = 0.8
):
    logging.info("Début du calcul des caractéristiques par ville.")
    try:
        transaction_per_city = transaction_per_city.groupby(aggregation_columns).agg(
            prix_m2_moyen=('prix_m2_moyen', 'mean')).reset_index()

        nb_transactions_departement = transaction_per_city.groupby(
            ['nom_departement', 'annee_transaction', 'mois_transaction']
        )['nb_transactions_mois'].sum().reset_index().rename(columns={'nb_transactions_mois': 'nb_transactions_departement'})

        transaction_per_city = transaction_per_city.merge(
            nb_transactions_departement, on=['nom_departement', 'annee_transaction', 'mois_transaction'], how='outer'
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
    sort_columns: list = ["nom_departement", "ville", 'id_ville', "annee_transaction", "mois_transaction"],
    aggregation_columns: list = ["nom_departement", "ville", "id_ville"]
):
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
        average_per_month_per_city["nb_transactions_mois_precedent"] = average_per_month_per_city["nb_transactions_mois_precedent"].astype(int)

        logging.info("Caractéristiques dérivées calculées avec succès.")
        return average_per_month_per_city
    except Exception as e:
        logging.error(f"Erreur lors du calcul des caractéristiques dérivées : {e}")
        raise

# Fonction de prétraitement principal
def preprocessing(df) -> pd.DataFrame:
    logging.info("Début du prétraitement des données.")
    try:
        df_city = df.groupby(
            ['nom_departement', 'ville', 'id_ville', 'annee_transaction', 'mois_transaction', 'nb_transactions_mois']
        ).agg(prix_m2_moyen=('prix_m2_moyen', 'mean')).round(2).reset_index()

        df_city = compute_city_features(transaction_per_city=df_city, threshold=0.8)

        if df_city is None or df_city.empty:
            logging.error("Le DataFrame df_city est vide après le calcul des caractéristiques de la ville.")
            raise ValueError("Le calcul des caractéristiques de la ville a échoué.")

        df_city_agg = compute_features_price_per_m2(df_city, aggregation_columns=["nom_departement", "ville", "id_ville"])

        if df_city_agg is None or df_city_agg.empty:
            logging.error("Le DataFrame df_city_agg est vide après le calcul des caractéristiques dérivées.")
            raise ValueError("Le calcul des caractéristiques dérivées a échoué.")

        required_columns = ['nom_departement', 'ville', 'id_ville', 'annee_transaction',
                            'mois_transaction', 'prix_m2_moyen_mois_precedent',
                            'nb_transactions_mois_precedent', 'ville_demandee']
        if not all(col in df_city_agg.columns for col in required_columns):
            missing_cols = set(required_columns) - set(df_city_agg.columns)
            logging.error(f"Colonnes manquantes pour la fusion : {missing_cols}")
            raise ValueError("Colonnes nécessaires manquantes dans df_city_agg.")

        df_prepared = df.merge(
            df_city_agg[required_columns],
            on=['nom_departement', 'ville', 'id_ville', 'annee_transaction', 'mois_transaction'], how='inner'
        )

        df_prepared = df_prepared.drop(['nom_departement', 'id_ville', 'ville', 'prix_m2',
                                        'prix_m2_moyen', 'nb_transactions_mois', 'n_pieces'], axis=1)

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
        preprocessing(data)
    except Exception as e:
        logging.critical(f"Erreur critique lors de l'exécution du script principal : {e}")
