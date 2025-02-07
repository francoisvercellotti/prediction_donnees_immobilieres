"""
Module pour l'entraînement et l'enregistrement de modèles de machine learning avec MLflow.

Ce module fournit des fonctions pour effectuer une validation croisée,
entraîner des modèles et les enregistrer avec MLflow.
"""

import logging
import os
from typing import Dict, Tuple, Optional, Union

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate, TimeSeriesSplit

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

    Args:
        features (pd.DataFrame): Données de features.
        target (pd.Series): Données cibles.
        model (object): Modèle à évaluer.
        cross_validation_strategy (Union[int, TimeSeriesSplit]): Stratégie de validation croisée.
        scoring_metrics (Tuple[str, ...]): Métriques de scoring.
        groups (Optional[np.ndarray], optional): Groupes pour validation croisée groupée.

    Returns:
        Dict[str, float]: Dictionnaire des moyennes et écarts-types des scores.
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

    Args:
        model (object): Modèle à entraîner.
        features_train (pd.DataFrame): Features d'entraînement.
        target_train (pd.Series): Cible d'entraînement.
        features_test (pd.DataFrame): Features de test.
        target_test (pd.Series): Cible de test.
        experiment_name (str): Nom de l'expérience MLflow.
        run_name (str): Nom du run MLflow.
        model_parameters (Dict[str, Union[int, float]]): Paramètres du modèle.
        run_tags (Dict[str, str]): Tags pour le run MLflow.
        cross_validation_strategy (Union[int, TimeSeriesSplit]): Stratégie de validation croisée.
        scoring_metrics (Tuple[str, ...]): Métriques de scoring.
        groups (Optional[np.ndarray], optional): Groupes pour validation croisée groupée.

    Returns:
        Tuple[object, Dict[str, float]]: Modèle entraîné et scores de validation.
    """
    mlflow_client = MlflowClient()
    try:
        experiment_id = mlflow_client.create_experiment(name=experiment_name)
        logging.info("Nouvelle expérience créée avec l'ID : %s", experiment_id)
    except Exception:
        experiment = mlflow_client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(run_tags)

        # Validation croisée
        cross_validation_scores = perform_cross_validation(
            features=features_train,
            target=target_train,
            model=model,
            cross_validation_strategy=cross_validation_strategy,
            scoring_metrics=scoring_metrics,
            groups=groups
        )

        # Enregistrement des paramètres et métriques
        for param_name, param_value in model_parameters.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_metrics(cross_validation_scores)

        # Préparation et sauvegarde des données
        preprocessed_dir = "data/interim/preprocessed"
        os.makedirs(preprocessed_dir, exist_ok=True)

        data_files = {
            "features_train": os.path.join(preprocessed_dir, "features_train.parquet"),
            "features_test": os.path.join(preprocessed_dir, "features_test.parquet"),
            "target_train": os.path.join(preprocessed_dir, "target_train.parquet"),
            "target_test": os.path.join(preprocessed_dir, "target_test.parquet")
        }

        # Enregistrement des DataFrames
        features_train.to_parquet(data_files["features_train"], index=False)
        features_test.to_parquet(data_files["features_test"], index=False)
        target_train.to_frame().to_parquet(data_files["target_train"], index=False)
        target_test.to_frame().to_parquet(data_files["target_test"], index=False)

        # Log des artefacts
        for file_path in data_files.values():
            mlflow.log_artifact(file_path)

        # Ajustement et enregistrement du modèle
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

    # Paramètres et tags pour MLFlow
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

    # Entraînement du modèle avec enregistrement MLFlow
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