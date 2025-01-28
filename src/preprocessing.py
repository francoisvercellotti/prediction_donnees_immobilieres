"""
Module de prétraitement pour les données immobilières.

Ce module contient des fonctions permettant de charger, traiter et transformer
les données liées aux transactions immobilières. Les principales étapes incluent
le calcul des caractéristiques des villes, l'encodage des colonnes catégorielles,
et la sauvegarde des résultats prétraités.

Fonctions principales :
- compute_city_features : Calcule les caractéristiques liées aux
transactions immobilières par ville.
- compute_features_price_per_m2 : Calcule les caractéristiques
dérivées des prix au m² et des transactions.
- encode_column : Encode une colonne catégorielle spécifique en
variables binaires avec OneHotEncoder.
- preprocessing : Fonction principale pour charger, prétraiter et transformer les données.

"""


import pandas as pd


# Fonction de calcul des caractéristiques de la ville
def compute_city_features(
    transaction_per_city: pd.DataFrame,
    aggregation_columns: list = ['nom_departement', 'ville','id_ville', 'annee_transaction', 'mois_transaction', 'nb_transactions_mois'],
    threshold: float = 0.8
):
    """
    Cette fonction calcule plusieurs caractéristiques liées aux transactions immobilières par ville.

    1. Elle agrège les données de transactions par ville et mois pour obtenir le prix moyen au m2.
    2. Elle calcule le ratio des transactions de chaque ville par rapport au département.
    3. Elle applique un seuil basé sur le quantile spécifié pour identifier les villes en tension.

    Paramètres :
    -----------
    transaction_per_city : pd.DataFrame
        Un DataFrame contenant les informations des transactions immobilières, y compris le prix au m2,
        le nombre de transactions, ainsi que des colonnes de regroupement par département, ville et mois.

    aggregation_columns : list, optionnel
        Liste des colonnes utilisées pour l'agrégation des données (par défaut : ['nom_departement', 'ville', 'annee_transaction', 'mois_transaction', 'nb_transactions_mois']).

    threshold : float, optionnel
        Quantile (par défaut à 0.8) utilisé pour déterminer le seuil au-dessus duquel une ville est considérée en tension.

    Retour :
    --------
    pd.DataFrame
        Un DataFrame enrichi avec :
        - `prix_m2_moyen` : Prix moyen au m2 par ville et mois,
        - `ratio_transaction` : Ratio des transactions de la ville par rapport au département,
        - `ville_demandee` : Indicateur binaire indiquant si la ville est en tension (1 si en tension, 0 sinon).
    """

    # Agréger les données par ville, année, mois et nombre de transactions
    transaction_per_city = transaction_per_city.groupby(aggregation_columns).agg(prix_m2_moyen=('prix_m2_moyen', 'mean')).reset_index()

    # Calculer le nombre total de transactions par département et par mois
    nb_transactions_departement = transaction_per_city.groupby(['nom_departement', 'annee_transaction', 'mois_transaction'])['nb_transactions_mois'].sum().reset_index().rename(columns={'nb_transactions_mois': 'nb_transactions_departement'})

    # Fusionner les données pour avoir le nombre total de transactions par département pour chaque ligne de ville
    transaction_per_city = transaction_per_city.merge(nb_transactions_departement, on=['nom_departement', 'annee_transaction', 'mois_transaction'], how='outer')

    # Calculer le ratio des transactions de la ville par rapport au département
    transaction_per_city['ratio_transaction'] = (transaction_per_city['nb_transactions_mois'] / transaction_per_city['nb_transactions_departement']) * 100

    # Déterminer le seuil de tension en fonction du quantile spécifié
    seuil_tension = transaction_per_city['ratio_transaction'].quantile(threshold)

    # Appliquer un indicateur binaire pour savoir si la ville est en tension
    transaction_per_city['ville_demandee'] = transaction_per_city['ratio_transaction'].apply(lambda x: 1 if x > seuil_tension else 0)

    return transaction_per_city



# Fonction pour calculer les caractéristiques du prix par m2
def compute_features_price_per_m2(
    average_per_month_per_city: pd.DataFrame,
    sort_columns: list = [
        "nom_departement",
        "ville",
        'id_ville',
        "annee_transaction",
        "mois_transaction"
    ],
    aggregation_columns: list = [
        "nom_departement",
        "ville",
        "id_ville",
    ]
):
    """
    Cette fonction calcule des caractéristiques dérivées pour les prix au m² et le nombre de transactions
    pour chaque ville et département, avec des moyennes des valeurs précédentes.

    Paramètres:
    - average_per_month_per_city (pd.DataFrame): DataFrame contenant les données des prix au m²
      et des transactions par mois et ville.
    - sort_columns (list, optionnel): Liste des colonnes par lesquelles trier les données avant de
      calculer les caractéristiques dérivées. Par défaut, cela inclut "nom_departement", "ville", "annee_transaction",
      et "mois_transaction".
    - aggregation_columns (list, optionnel): Colonnes utilisées pour le groupby. Par défaut, cela inclut
      "nom_departement", "ville" et "id_ville".

    Retour:
    - pd.DataFrame: DataFrame avec les nouvelles caractéristiques dérivées, incluant les prix et
      le nombre de transactions des mois précédents.

    Processus:
    1. Trie les données selon les colonnes spécifiées.
    2. Calcule les prix au m² et le nombre de transactions des mois précédents pour chaque groupe.
    3. Supprime les lignes contenant des valeurs manquantes (NaN).


    # Trier les données selon les colonnes spécifiées
    average_per_month_per_city = average_per_month_per_city.sort_values(by=sort_columns)

    # Calculer les colonnes dérivées pour le mois précédent (shift)
    average_per_month_per_city["prix_m2_moyen_mois_precedent"] = (
        average_per_month_per_city.groupby(aggregation_columns)["prix_m2_moyen"]
        .shift()
    )
    average_per_month_per_city["nb_transactions_mois_precedent"] = (
        average_per_month_per_city.groupby(aggregation_columns)['nb_transactions_mois']
        .shift()
    )

    # Supprimer les NaN engendrés par les opérations de shift
    average_per_month_per_city = average_per_month_per_city.dropna()

    # Retourner le DataFrame avec les nouvelles caractéristiques calculées
    return average_per_month_per_city
"""

# Trier les données selon les colonnes spécifiées
    average_per_month_per_city = average_per_month_per_city.sort_values(by=sort_columns)

    # Calculer les colonnes dérivées pour le mois précédent (shift)
    average_per_month_per_city["prix_m2_moyen_mois_precedent"] = (
        average_per_month_per_city.groupby(aggregation_columns)["prix_m2_moyen"]
        .shift()
    )
    average_per_month_per_city["nb_transactions_mois_precedent"] = (
        average_per_month_per_city.groupby(aggregation_columns)['nb_transactions_mois']
        .shift()
    )


    # Supprimer les NaN engendrés par les opérations de shift
    average_per_month_per_city = average_per_month_per_city.dropna()

    # Convertir en entier
    average_per_month_per_city["nb_transactions_mois_precedent"] = average_per_month_per_city["nb_transactions_mois_precedent"].astype(int)

    # Retourner le DataFrame avec les nouvelles caractéristiques calculées
    return average_per_month_per_city


# Fonction de prétraitement principal
def preprocessing(df) -> pd.DataFrame:
    """
    Fonction principale pour charger et prétraiter.
    """

    # Agrégation par ville et mois
    df_city = df.groupby(['nom_departement', 'ville', 'id_ville',
                          'annee_transaction', 'mois_transaction',
                          'nb_transactions_mois']).agg(
                              prix_m2_moyen=('prix_m2_moyen', 'mean')).round(2).reset_index()

    # Calcul des caractéristiques de la ville
    df_city = compute_city_features(transaction_per_city=df_city, threshold=0.8)

    if df_city is None or df_city.empty:
        raise ValueError("Le calcul des caractéristiques de la ville a échoué. df_city est vide ou None.")

    df_city_agg = compute_features_price_per_m2(df_city, aggregation_columns=["nom_departement", "ville", "id_ville"])

    if df_city_agg is None or df_city_agg.empty:
        raise ValueError("Le calcul des caractéristiques dérivées a échoué. df_city_agg est vide ou None.")

    # Vérification des colonnes avant fusion
    required_columns = ['nom_departement', 'ville', 'id_ville', 'annee_transaction',
                        'mois_transaction', 'prix_m2_moyen_mois_precedent',
                        'nb_transactions_mois_precedent', 'ville_demandee']
    if not all(col in df_city_agg.columns for col in required_columns):
        raise ValueError(f"Les colonnes nécessaires pour la fusion ne sont pas présentes dans df_city_agg. "
                         f"Colonnes trouvées : {df_city_agg.columns}")

    # Fusionner les nouvelles caractéristiques avec les données originales
    df_prepared = df.merge(df_city_agg[required_columns],
                           on=['nom_departement', 'ville', 'id_ville',
                               'annee_transaction', 'mois_transaction'], how='inner')

    df_prepared = df_prepared.drop(['nom_departement','id_ville','ville','prix_m2',
                                    'prix_m2_moyen','nb_transactions_mois','n_pieces'],axis=1)

    # Sauvegarder les transactions prétraitées en parquet
    OUTPUT_FILE = 'data/interim/prepared_dataset.parquet'
    df_prepared.to_parquet(OUTPUT_FILE, index=False)
    print(f"Le fichier a été préparé et sauvegardé sous '{OUTPUT_FILE}'.")

    # Retourner les résultats prétraités
    return df_prepared

if __name__ == "__main__":
    # Charger le DataFrame
    data = pd.read_parquet( 'data/loaded/loaded_dataset.parquet')
    preprocessing(data)