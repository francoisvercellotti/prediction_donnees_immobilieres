"""
Module pour l'entraînement et l'enregistrement de modèles de machine learning avec MLflow.

Ce module fournit des fonctions pour effectuer une validation croisée,
entraîner des modèles et les enregistrer avec MLflow.
"""

import logging
import os
import sys
from typing import Dict, Tuple, Optional, Union
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate, TimeSeriesSplit
# Ajouter le dossier parent au chemin pour importer les paramètres
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing_pipeline import preprocess_data

def perform_cross_validation(
    features: pd.DataFrame,
    target: pd.Series,
    model: object,
    cross_validation_strategy: Union[int, TimeSeriesSplit],
    scoring_metrics: Tuple[str, ...],
    groups: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Effectue une validation croisée pour un modèle donné et retourne les scores.
    """
    cv_results = cross_validate(
        model,
        features.values,
        target.values,
        cv=cross_validation_strategy,
        return_train_score=True,
        return_estimator=True,
        scoring=scoring_metrics,
        groups=groups
    )

    scores_dict = {}
    for metric in scoring_metrics:
        scores_dict[f"moyenne_train_{metric}"] = np.mean(cv_results[f"train_{metric}"])
        scores_dict[f"ecart_type_train_{metric}"] = np.std(cv_results[f"train_{metric}"])
        scores_dict[f"moyenne_test_{metric}"] = np.mean(cv_results[f"test_{metric}"])
        scores_dict[f"ecart_type_test_{metric}"] = np.std(cv_results[f"test_{metric}"])

    return scores_dict

def get_or_create_parent_run(experiment_name: str, run_name: str) -> str:
    """
    Récupère le run parent actif ou en crée un nouveau si nécessaire.
    """
    # Configuration de MLflow
    mlflow.set_tracking_uri("file:" + os.path.abspath("mlruns"))
    mlflow.set_experiment(experiment_name)

    active_run = mlflow.active_run()
    if active_run:
        return active_run.info.run_id

    # Si pas de run actif, chercher le dernier run parent créé
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName LIKE 'training_%'",
            order_by=["start_time DESC"],
            max_results=1
        )
        if runs:
            return runs[0].info.run_id

    # Si aucun run trouvé, créer un nouveau
    with mlflow.start_run(run_name=run_name) as run:
        return run.info.run_id

def train_and_log_model_with_mlflow(
    model: object,
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    experiment_name: str,
    run_name: str,
    model_parameters: Dict[str, Union[int, float]],
    run_tags: Dict[str, str],
    cross_validation_strategy: Union[int, TimeSeriesSplit],
    scoring_metrics: Tuple[str, ...],
    groups: Optional[np.ndarray] = None
) -> Tuple[object, Dict[str, float]]:
    """
    Entraîne un modèle, effectue une validation croisée et enregistre les résultats avec MLflow.
    """
    # Récupérer ou créer le run parent
    parent_run_id = get_or_create_parent_run(experiment_name, run_name)

    with mlflow.start_run(run_id=parent_run_id):
        mlflow.set_tags(run_tags)

        # Effectuer la validation croisée
        cross_validation_scores = perform_cross_validation(
            features=features_train,
            target=target_train,
            model=model,
            cross_validation_strategy=cross_validation_strategy,
            scoring_metrics=scoring_metrics,
            groups=groups
        )

        # Logger les paramètres et métriques
        for param_name, param_value in model_parameters.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_metrics(cross_validation_scores)

        # Sauvegarder les données prétraitées
        preprocessed_dir = "data/interim/preprocessed"
        os.makedirs(preprocessed_dir, exist_ok=True)
        data_files = {
            "features_train": os.path.join(preprocessed_dir, "features_train.parquet"),
            "features_test": os.path.join(preprocessed_dir, "features_test.parquet"),
            "target_train": os.path.join(preprocessed_dir, "target_train.parquet"),
            "target_test": os.path.join(preprocessed_dir, "target_test.parquet")
        }

        features_train.to_parquet(data_files["features_train"], index=False)
        features_test.to_parquet(data_files["features_test"], index=False)
        target_train.to_frame().to_parquet(data_files["target_train"], index=False)
        target_test.to_frame().to_parquet(data_files["target_test"], index=False)

        for file_path in data_files.values():
            mlflow.log_artifact(file_path)

        # Entraîner et sauvegarder le modèle
        model.fit(features_train, target_train)
        signature = infer_signature(features_train, target_train)
        input_example = features_train.head(1)
        mlflow.sklearn.log_model(model, "model", input_example=input_example, signature=signature)

        return model, cross_validation_scores

def main():
    """
    Fonction principale pour l'entraînement du modèle et l'enregistrement avec MLflow.
    """
    logging.info("Chargement du DataFrame.")
    data_frame = pd.read_parquet('data/interim/prepared_dataset.parquet')

    # Colonnes à encoder
    COLUMNS_TO_ENCODE = ['type_batiment', 'nom_region']

    # Prétraitement des données
    features_train, features_test, target_train_reg, target_test_reg, _, _ = preprocess_data(
        data_frame,
        COLUMNS_TO_ENCODE
    )

    # Définition du modèle
    model = GradientBoostingRegressor(
        learning_rate=0.07,
        max_depth=5,
        min_samples_leaf=2,
        min_samples_split=4,
        n_estimators=150,
        random_state=42,
        subsample=0.9
    )

    # Paramètres et tags pour MLflow
    MODEL_PARAMETERS = {
        'learning_rate': 0.07,
        'max_depth': 5,
        'min_samples_leaf': 2,
        'min_samples_split': 4,
        'n_estimators': 150,
        'random_state': 42,
        'subsample': 0.9
    }
    RUN_TAGS = {
        'project': 'immobilier_prediction',
        'CV_Folds': '5',
        'CV_Method': 'TimeSeriesSplit',
        'Dataset': 'trainset_v1',
        'Experiment_Type': 'regression_model',
        'Model_Type': 'Gradient Boosting Regressor',
        'Phase': 'production',
        'Run_Type': 'model_training',
        'Solver': 'Gradient Boosting',
        'Task': 'Regression'
    }

    # Configuration de la validation croisée
    CROSS_VALIDATION_STRATEGY = TimeSeriesSplit(n_splits=5)
    SCORING_METRICS = ('neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2')

    EXPERIMENT_NAME = "regression_model_prediction_immobilières"
    RUN_NAME = "run_train_model_v1_dataset_v1"

    # Vérification des valeurs manquantes
    logging.info("Valeurs manquantes dans X_train : %s", features_train.isna().sum().sum())
    logging.info("Valeurs manquantes dans y_train_reg : %s", target_train_reg.isna().sum().sum())

    # Entraînement du modèle avec enregistrement MLflow
    trained_model, cross_validation_scores = train_and_log_model_with_mlflow(
        model=model,
        features_train=features_train,
        target_train=target_train_reg,
        features_test=features_test,
        target_test=target_test_reg,
        experiment_name=EXPERIMENT_NAME,
        run_name=RUN_NAME,
        model_parameters=MODEL_PARAMETERS,
        run_tags=RUN_TAGS,
        cross_validation_strategy=CROSS_VALIDATION_STRATEGY,
        scoring_metrics=SCORING_METRICS
    )

    logging.info("Entraînement et validation croisée terminés.")
    logging.info("Scores de validation croisée : %s", cross_validation_scores)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()