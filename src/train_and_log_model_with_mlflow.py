import os
import sys
# Ajouter le dossier parent au chemin pour importer les paramètres
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import cross_validate, TimeSeriesSplit
import numpy as np
import pandas as pd
import logging
from src.preprocessing_pipeline import preprocess_data

def perform_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    cross_val_type,
    scoring_metrics: tuple,
    groupes=None,
):
    """
    Effectue une validation croisée pour un modèle donné et retourne les scores d'entraînement et de test.

    Retourne :
    ---------
    scores_dict : dict
        Moyennes et écarts-types des scores pour chaque métrique.
    """
    scores = cross_validate(
        model,
        X.values,  # Convertir DataFrame en tableau NumPy
        y.values,  # Convertir Series en tableau NumPy
        cv=cross_val_type,
        return_train_score=True,
        return_estimator=True,
        scoring=scoring_metrics,
        groups=groupes,
    )

    # Calculer les moyennes et écarts-types pour chaque métrique
    scores_dict = {}
    for metrique in scoring_metrics:
        scores_dict["moyenne_train_" + metrique] = np.mean(scores["train_" + metrique])
        scores_dict["ecart_type_train_" + metrique] = np.std(scores["train_" + metrique])
        scores_dict["moyenne_test_" + metrique] = np.mean(scores["test_" + metrique])
        scores_dict["ecart_type_test_" + metrique] = np.std(scores["test_" + metrique])

    return scores_dict


def train_and_log_model_with_mlflow(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    experiment_name,
    run_name,
    model_params,
    tags,
    cross_val_type,
    scoring_metrics,
    groups=None,
):
    """
    Entraîne un modèle de régression, effectue une validation croisée,
    enregistre les résultats et le modèle avec MLflow.

    Paramètres :
    ----------
    model : object
        Modèle de régression (ex : LinearRegression, Ridge, etc.).
    X_train : pd.DataFrame
        Données d'entraînement.
    y_train : pd.Series
        Étiquettes d'entraînement.
    X_test : pd.DataFrame
        Données de test.
    y_test : pd.Series
        Étiquettes de test.
    experiment_name : str
        Nom de l'expérience MLflow.
    run_name : str
        Nom du run dans MLflow.
    model_params : dict
        Paramètres du modèle.
    tags : dict
        Tags pour le run MLflow.
    cross_val_type : int, générateur ou objet de cross-validation
        Stratégie de validation croisée (ex : KFold, StratifiedKFold).
    scoring_metrics : tuple
        Liste des métriques à utiliser pour la validation croisée.
    groups : array-like, optionnel
        Groupes utilisés pour certaines stratégies comme GroupKFold.

    Retourne :
    ---------
    trained_model : object
        Le modèle ajusté sur l'ensemble des données (X_train, y_train).
    scores_dict : dict
        Les scores de validation croisée.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        experiment_id = client.create_experiment(name=experiment_name)
        print(f"Nouvelle expérience créée avec l'ID : {experiment_id}")
    except Exception:
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(tags)

        # Validation croisée
        scores_dict = perform_cross_validation(
            X=X_train,
            y=y_train,
            model=model,
            cross_val_type=cross_val_type,
            scoring_metrics=scoring_metrics,
            groupes=groups,
        )

        # Enregistrement des paramètres du modèle
        for param_name, param_value in model_params.items():
            mlflow.log_param(param_name, param_value)

        # Enregistrement des scores de validation croisée
        mlflow.log_metrics(scores_dict)

        preprocessed_dir = "data/interim/preprocessed"
        os.makedirs(preprocessed_dir, exist_ok=True)  # Créer le dossier s'il n'existe pas

        X_train_file = os.path.join(preprocessed_dir, "X_train.parquet")
        X_test_file = os.path.join(preprocessed_dir, "X_test.parquet")
        y_train_file = os.path.join(preprocessed_dir, "y_train.parquet")
        y_test_file = os.path.join(preprocessed_dir, "y_test.parquet")

        # Enregistrement des DataFrames en fichiers parquet dans le dossier spécifié
        X_train.to_parquet(X_train_file, index=False)
        X_test.to_parquet(X_test_file, index=False)
        y_train.to_frame().to_parquet(y_train_file, index=False)
        y_test.to_frame().to_parquet(y_test_file, index=False)

        # Log des artefacts dans MLflow
        mlflow.log_artifact(X_train_file)
        mlflow.log_artifact(X_test_file)
        mlflow.log_artifact(y_train_file)
        mlflow.log_artifact(y_test_file)

        # Ajuster le modèle final sur l'ensemble des données
        model.fit(X_train, y_train)

        # Signature et exemple d'entrée pour MLflow
        signature = infer_signature(X_train, y_train)
        input_example = X_train.head(1)
        mlflow.sklearn.log_model(model, "model", input_example=input_example, signature=signature)

        return model, scores_dict


if __name__ == "__main__":
    logging.info("Chargement du DataFrame.")
    df = pd.read_parquet('data/interim/prepared_dataset.parquet')

    # Colonnes à encoder
    columns_to_encode = ['type_batiment', 'nom_region']

    # Prétraitement des données : Séparation en train/test
    X_train, X_test, y_train_reg, y_test_reg, _, _, = preprocess_data(df, columns_to_encode)

    # Définir le modèle GradientBoostingRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(
        learning_rate=0.07,
        max_depth=5,
        min_samples_leaf=2,
        min_samples_split=4,
        n_estimators=150,
        random_state=42,
        subsample=0.9
    )

    # Paramètres et tags pour MLFlow
    model_params = {
        'learning_rate': 0.07,
        'max_depth': 5,
        'min_samples_leaf': 2,
        'min_samples_split': 4,
        'n_estimators': 150,
        'random_state': 42,
        'subsample': 0.9
    }
    tags = {
        'project': 'immobilier_prediction',
        'CV_Folds' : '5',
        'CV_Method' : 'TimeSeriesSplit',
        'Dataset' : 'trainset_v1',
        'Experiment_Type' : 'regression_model',
        'Model_Type' : 'Gradient Boosting Regressor',
        'Phase' : 'production',
        'Run_Type' : 'model_training',
        'Solver': 'Gradient Boosting',
        'Task' : 'Regression'
    }

    # Paramètres pour la validation croisée
    cross_val_type = TimeSeriesSplit(n_splits=5)
    scoring_metrics = ('neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2')

    # Nom de l'expérience et du run MLFlow
    experiment_name = "regression_model_prediction_immobilières"
    run_name = "run_train_model_v1_dataset_v1"

    # Vérification des valeurs manquantes
    print("Nombre total de valeurs manquantes dans X_train :", X_train.isna().sum().sum())
    print("Nombre total de valeurs manquantes dans y_train_reg :", y_train_reg.isna().sum().sum())

    # Entraînement du modèle avec enregistrement MLFlow
    trained_model, scores_dict = train_and_log_model_with_mlflow(
        model,
        X_train,
        y_train_reg,
        X_test,
        y_test_reg,
        experiment_name,
        run_name,
        model_params,
        tags,
        cross_val_type,
        scoring_metrics,
    )

    logging.info("Entraînement et validation croisée terminés.")
    logging.info(f"Scores de validation croisée : {scores_dict}")
