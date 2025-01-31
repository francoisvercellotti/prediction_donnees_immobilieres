import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from src.train_and_log_model_with_mlflow import train_and_log_model_with_mlflow

def generate_mock_data(n_samples=100):
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(n_samples, 5), columns=[f"feature_{i}" for i in range(5)])
    y = pd.Series(np.random.rand(n_samples), name="target")
    return X, y

def test_train_and_log_model_with_mlflow():
    X_train, y_train = generate_mock_data()
    X_test, y_test = generate_mock_data(n_samples=20)

    model = GradientBoostingRegressor(
        learning_rate=0.07,
        max_depth=5,
        min_samples_leaf=2,
        min_samples_split=4,
        n_estimators=150,
        random_state=42,
        subsample=0.9
    )

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
        'project': 'test_project',
        'Model_Type': 'Gradient Boosting Regressor'
    }

    cross_val_type = TimeSeriesSplit(n_splits=3)
    scoring_metrics = ('neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2')

    experiment_name = "test_experiment"
    run_name = "test_run"

    # Mock MLflow functions et désactiver la génération de fichiers
    with patch("mlflow.start_run"), \
         patch("mlflow.set_experiment"), \
         patch("mlflow.log_param"), \
         patch("mlflow.log_metrics"), \
         patch("mlflow.log_artifact"), \
         patch("mlflow.sklearn.log_model"), \
         patch("mlflow.set_tags"), \
         patch("mlflow.tracking.MlflowClient"), \
         patch('pandas.DataFrame.to_parquet'):

        trained_model, scores_dict = train_and_log_model_with_mlflow(
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
            scoring_metrics
        )

    # Vérifications identiques au test précédent
    assert trained_model is not None, "Le modèle entraîné ne doit pas être None"
    assert isinstance(scores_dict, dict), "Les scores doivent être retournés sous forme de dictionnaire"
    assert "moyenne_test_r2" in scores_dict, "Les scores de R2 doivent être inclus dans le dictionnaire de sortie"

    for metric in scoring_metrics:
        assert f"moyenne_test_{metric}" in scores_dict, f"La métrique {metric} doit être présente dans les scores"
        assert f"moyenne_train_{metric}" in scores_dict, f"La métrique {metric} doit être présente pour l'entraînement"

if __name__ == "__main__":
    pytest.main(["-v", "test_train_model.py"])