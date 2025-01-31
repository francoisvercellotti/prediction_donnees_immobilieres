import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import shutil
from src.evaluate_model import error_analysis, load_model_from_mlflow, load_data_from_mlflow

def create_test_data():
    """Crée des données synthétiques pour les tests"""
    np.random.seed(42)

    # Création des features
    X_train = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })
    X_test = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 30),
        'feature2': np.random.normal(0, 1, 30)
    })

    # Création des variables cibles avec un peu de bruit
    y_train = pd.DataFrame(X_train['feature1'] * 2 + X_train['feature2'] + np.random.normal(0, 0.1, 100))
    y_test = pd.DataFrame(X_test['feature1'] * 2 + X_test['feature2'] + np.random.normal(0, 0.1, 30))

    # Création et entraînement du modèle
    model = LinearRegression()
    model.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, model

def test_error_analysis_output_structure():
    """Teste la structure de sortie de l'analyse d'erreurs"""
    print("\n🧪 Test de la structure des métriques...")

    # Préparation
    X_train, X_test, y_train, y_test, model = create_test_data()
    output_dir = "test_artifacts"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Exécution
        metrics = error_analysis(model, X_train, y_train, X_test, y_test, output_dir)

        # Vérification
        expected_metrics = ['Train RMSE', 'Train MAE', 'Train R2',
                          'Test RMSE', 'Test MAE', 'Test R2']

        all_metrics_present = all(metric in metrics for metric in expected_metrics)
        all_metrics_float = all(isinstance(metrics[metric], float) for metric in expected_metrics)

        assert all_metrics_present, "❌ Certaines métriques sont manquantes"
        assert all_metrics_float, "❌ Certaines métriques ne sont pas des nombres"

        print("✅ Test de structure réussi")
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

def test_file_creation():
    """Teste la création des fichiers de visualisation"""
    print("\n🧪 Test de la création des fichiers...")

    # Préparation
    X_train, X_test, y_train, y_test, model = create_test_data()
    output_dir = "test_artifacts"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Exécution
        error_analysis(model, X_train, y_train, X_test, y_test, output_dir)

        # Vérification
        expected_files = [
            'train_predictions.png',
            'test_predictions.png',
            'train_error_dist.png',
            'test_error_dist.png'
        ]

        for file in expected_files:
            assert os.path.exists(os.path.join(output_dir, file)), f"❌ Le fichier {file} n'existe pas"

        print("✅ Test de création des fichiers réussi")
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

def test_empty_data():
    """Teste la gestion des données vides"""
    print("\n🧪 Test avec données vides...")

    empty_df = pd.DataFrame()
    output_dir = "test_artifacts"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Exécution
        try:
            error_analysis(LinearRegression(), empty_df, empty_df, empty_df, empty_df, output_dir)
            print("❌ Le test aurait dû lever une exception")
            assert False
        except ValueError:
            print("✅ Test avec données vides réussi (exception levée comme prévu)")
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

def test_metrics_values():
    """Teste les valeurs des métriques"""
    print("\n🧪 Test des valeurs des métriques...")

    # Préparation
    X_train, X_test, y_train, y_test, model = create_test_data()
    output_dir = "test_artifacts"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Exécution
        metrics = error_analysis(model, X_train, y_train, X_test, y_test, output_dir)

        # Vérification
        assert 0 <= metrics['Train R2'] <= 1, "❌ Train R2 hors limites"
        assert 0 <= metrics['Test R2'] <= 1, "❌ Test R2 hors limites"
        assert metrics['Train RMSE'] > 0, "❌ Train RMSE négatif"
        assert metrics['Test RMSE'] > 0, "❌ Test RMSE négatif"
        assert metrics['Train MAE'] > 0, "❌ Train MAE négatif"
        assert metrics['Test MAE'] > 0, "❌ Test MAE négatif"

        print("✅ Test des valeurs des métriques réussi")
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

if __name__ == '__main__':
    print("🚀 Démarrage des tests...")

    try:
        test_error_analysis_output_structure()
        test_file_creation()
        test_empty_data()
        test_metrics_values()
        print("\n✨ Tous les tests ont réussi !")
    except AssertionError as e:
        print(f"\n❌ Échec des tests : {str(e)}")
    except Exception as e:
        print(f"\n❌ Erreur inattendue : {str(e)}")