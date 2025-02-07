"""
Module de test pour la fonction error_analysis du fichier evaluate_model.

Ce module teste les performances et la robustesse de la fonction error_analysis,
qui analyse les erreurs d'un modèle de régression. Il inclut des tests pour :
- La vérification des métriques retournées.
- La gestion des données vides.
- L'intégration avec MLflow (mockée).
"""

import os
import sys
import unittest.mock as mock
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.evaluate_model import error_analysis

# Ajout du chemin du projet pour l'importation des modules locaux
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_test_data():
    """Crée des données synthétiques pour tester le modèle de régression."""
    np.random.seed(42)
    X_train = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })
    X_test = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 30),
        'feature2': np.random.normal(0, 1, 30)
    })
    y_train = X_train['feature1'] * 2 + X_train['feature2'] + np.random.normal(0, 0.1, 100)
    y_test = X_test['feature1'] * 2 + X_test['feature2'] + np.random.normal(0, 0.1, 30)

    model = LinearRegression()
    model.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, model

def test_error_analysis_metrics():
    """Teste error_analysis sans interactions avec le système de fichiers ni exécutions MLflow."""
    X_train, X_test, y_train, y_test, model = create_test_data()

    # Simulation d'une expérience MLflow fictive
    FakeExperiment = type("FakeExperiment", (), {"experiment_id": "fake_experiment_id"})
    fake_experiment = FakeExperiment()

    # Patcher les appels à MLflow et les fonctions liées aux fichiers
    with mock.patch('mlflow.get_experiment_by_name', return_value=fake_experiment), \
         mock.patch('mlflow.start_run'), \
         mock.patch('mlflow.log_metrics'), \
         mock.patch('mlflow.log_params'), \
         mock.patch('mlflow.log_artifacts'), \
         mock.patch('matplotlib.pyplot.savefig'), \
         mock.patch('os.makedirs'), \
         mock.patch('matplotlib.pyplot.close'):

        metrics = error_analysis(model, X_train, y_train, X_test, y_test)

        # Vérification des métriques attendues
        expected_metrics = ['Train RMSE', 'Train MAE', 'Train R2',
                            'Test RMSE', 'Test MAE', 'Test R2']

        assert all(metric in metrics for metric in expected_metrics), "Métriques manquantes"
        assert all(isinstance(metrics[metric], float) for metric in expected_metrics), \
            "Métriques invalides"

        # Vérifications de plausibilité des valeurs des métriques
        assert 0 <= metrics['Train R2'] <= 1, "Train R2 hors limites"
        assert 0 <= metrics['Test R2'] <= 1, "Test R2 hors limites"
        assert all(metrics[metric] > 0 for metric in [
            'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE']), "Métriques négatives"

def test_empty_data():
    """Teste la gestion des erreurs avec des données vides."""
    empty_df = pd.DataFrame()
    model = LinearRegression()

    with pytest.raises(ValueError):
        error_analysis(model, empty_df, empty_df, empty_df, empty_df)

if __name__ == "__main__":
    pytest.main([__file__])
