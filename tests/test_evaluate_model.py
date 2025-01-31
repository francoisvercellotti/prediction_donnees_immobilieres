import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from contextlib import nullcontext
from sklearn.linear_model import LinearRegression
from src.evaluate_model import error_analysis

def create_test_data():
    """Crée des données synthétiques pour les tests"""
    np.random.seed(42)
    X_train = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })
    X_test = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 30),
        'feature2': np.random.normal(0, 1, 30)
    })
    y_train = pd.Series(X_train['feature1'] * 2 + X_train['feature2'] + np.random.normal(0, 0.1, 100))
    y_test = pd.Series(X_test['feature1'] * 2 + X_test['feature2'] + np.random.normal(0, 0.1, 30))
    model = LinearRegression()
    model.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, model

def test_error_analysis_metrics():
    """Teste la fonction error_analysis sans générer de fichiers ni de runs MLflow"""
    X_train, X_test, y_train, y_test, model = create_test_data()

    # Désactive tous les effets secondaires de MLflow et des fichiers
    with patch('mlflow.start_run', return_value=nullcontext()), \
         patch('mlflow.log_metrics'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_artifacts'), \
         patch('matplotlib.pyplot.savefig'), \
         patch('os.makedirs'), \
         patch('matplotlib.pyplot.close'):

        # Forcer un répertoire de sortie temporaire
        metrics = error_analysis(model, X_train, y_train, X_test, y_test, output_dir="/tmp")

        # Vérifications des métriques
        expected_metrics = ['Train RMSE', 'Train MAE', 'Train R2',
                            'Test RMSE', 'Test MAE', 'Test R2']

        assert all(metric in metrics for metric in expected_metrics), "Métriques manquantes"
        assert all(isinstance(metrics[metric], float) for metric in expected_metrics), "Métriques invalides"

        # Vérifications de plausibilité
        assert 0 <= metrics['Train R2'] <= 1, "Train R2 hors limites"
        assert 0 <= metrics['Test R2'] <= 1, "Test R2 hors limites"
        assert all(metrics[metric] > 0 for metric in ['Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE']), "Métriques négatives"

def test_empty_data():
    """Teste la gestion des données vides"""
    empty_df = pd.DataFrame()
    model = LinearRegression()

    with pytest.raises(ValueError):
        error_analysis(model, empty_df, empty_df, empty_df, empty_df)

if __name__ == "__main__":
    pytest.main([__file__])