import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_latest_training_run(experiment_name: str):
    """
    Récupère le dernier run d'entraînement de l'expérience.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Expérience '{experiment_name}' non trouvée")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.Run_Type = 'model_training'",
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("Aucun run d'entraînement trouvé")

    return runs[0]

def load_data_from_mlflow(run_id, dataset_name="train"):
    """
    Récupère les jeux de données depuis MLflow.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            X_path = client.download_artifacts(run_id, f"features_{dataset_name}.parquet", temp_dir)
            y_path = client.download_artifacts(run_id, f"target_{dataset_name}.parquet", temp_dir)
            X = pd.read_parquet(X_path)
            y = pd.read_parquet(y_path)
            logger.info(f"📊 {dataset_name.capitalize()} chargé depuis MLflow.")
            return X, y
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lors du téléchargement des données: {e}")

def format_metrics(metrics, train_errors, test_errors):
    """
    Formate les métriques d'évaluation du modèle.
    """
    pd.set_option('display.float_format', lambda x: '%.4f' % x)

    train_stats = pd.Series(train_errors).describe()
    test_stats = pd.Series(test_errors).describe()

    mlflow_metrics = {
        "Train RMSE": metrics["Train RMSE"],
        "Train MAE": metrics["Train MAE"],
        "Train R2": metrics["Train R2"],
        "Test RMSE": metrics["Test RMSE"],
        "Test MAE": metrics["Test MAE"],
        "Test R2": metrics["Test R2"],

        # Statistiques des erreurs
        "Train_Error_Mean": float(train_stats['mean']),
        "Train_Error_Std": float(train_stats['std']),
        "Test_Error_Mean": float(test_stats['mean']),
        "Test_Error_Std": float(test_stats['std'])
    }

    return mlflow_metrics

def plot_regression_predictions(y_true, y_pred, title, filename, output_dir):
    """
    Crée un graphique des prédictions vs réalité.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", s=30)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
             color="red", linestyle="--", label="Idéal")
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_regression_error(y_true, y_pred, title, filename, output_dir):
    """
    Crée un histogramme des erreurs.
    """
    errors = y_true - y_pred
    error_std = np.std(errors)
    error_mean = np.mean(errors)

    plt.figure(figsize=(8, 6))
    sns.kdeplot(errors, color="blue", fill=True, alpha=0.3)
    plt.axvline(x=0, color='red', linestyle='--', label='Zero Error')
    plt.xlim(error_mean - 4 * error_std, error_mean + 4 * error_std)
    plt.xlabel("Erreur de prédiction")
    plt.ylabel("Densité")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def error_analysis(parent_run_id, model, X_train, y_train, X_test, y_test):
    """
    Effectue une analyse des erreurs et enregistre les résultats dans un run enfant MLflow.
    """
    with tempfile.TemporaryDirectory() as output_dir:
        if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
            raise ValueError("❌ Les jeux de données sont vides!")

        # Démarrer un run enfant
        with mlflow.start_run(run_id=parent_run_id):
            with mlflow.start_run(run_name="model_evaluation",
                                nested=True,
                                description="Analyse détaillée des erreurs du modèle"):

                logger.info("🔍 Début de l'analyse des erreurs...")

                # Prédictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Conversion en arrays 1D
                y_train = y_train.values.ravel()
                y_test = y_test.values.ravel()
                y_train_pred = y_train_pred.ravel()
                y_test_pred = y_test_pred.ravel()

                # Calcul des métriques
                metrics = {
                    "Train RMSE": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
                    "Train MAE": float(mean_absolute_error(y_train, y_train_pred)),
                    "Train R2": float(r2_score(y_train, y_train_pred)),
                    "Test RMSE": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
                    "Test MAE": float(mean_absolute_error(y_test, y_test_pred)),
                    "Test R2": float(r2_score(y_test, y_test_pred))
                }

                # Calcul des erreurs absolues
                absolute_errors_train = np.abs(y_train - y_train_pred)
                absolute_errors_test = np.abs(y_test - y_test_pred)

                # Format des métriques pour MLflow
                mlflow_metrics = format_metrics(metrics, absolute_errors_train, absolute_errors_test)

                # Log des métriques
                mlflow.log_metrics(mlflow_metrics)

                # Génération des visualisations
                plot_regression_predictions(y_train, y_train_pred,
                                         "Prédictions vs Réel (Train)",
                                         "train_predictions.png",
                                         output_dir)
                plot_regression_predictions(y_test, y_test_pred,
                                         "Prédictions vs Réel (Test)",
                                         "test_predictions.png",
                                         output_dir)
                plot_regression_error(y_train, y_train_pred,
                                    "Distribution des Erreurs (Train)",
                                    "train_error_dist.png",
                                    output_dir)
                plot_regression_error(y_test, y_test_pred,
                                    "Distribution des Erreurs (Test)",
                                    "test_error_dist.png",
                                    output_dir)

                # Log des visualisations
                mlflow.log_artifacts(output_dir)

                logger.info("✅ Analyse des erreurs terminée et résultats enregistrés dans MLflow")

                return metrics

def main():
    try:
        # Configuration de MLflow
        EXPERIMENT_NAME = "regression_model_prediction_immobilières"
        mlflow.set_tracking_uri("file:" + os.path.abspath("mlruns"))
        mlflow.set_experiment(EXPERIMENT_NAME)

        # Récupérer le dernier run d'entraînement
        parent_run = get_latest_training_run(EXPERIMENT_NAME)
        parent_run_id = parent_run.info.run_id
        logger.info(f"Run parent trouvé: {parent_run_id}")

        # Chargement du modèle
        model_uri = f"runs:/{parent_run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Modèle chargé depuis MLflow")

        # Chargement des données
        X_train, y_train = load_data_from_mlflow(parent_run_id, "train")
        X_test, y_test = load_data_from_mlflow(parent_run_id, "test")

        # Analyse des erreurs
        error_analysis(parent_run_id, model, X_train, y_train, X_test, y_test)

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'exécution : {e}")
        raise

if __name__ == "__main__":
    main()