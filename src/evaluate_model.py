import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration de MLflow
mlflow.set_tracking_uri("http://localhost:5000")
EXPERIMENT_NAME = "regression_model_prediction_immobilières"
RUN_NAME = "run_train_model_v1_dataset_v1"

def load_model_from_mlflow(experiment_name, run_name):
    """ Récupère le modèle sauvegardé dans MLflow pour un run donné. """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        raise ValueError(f"Expérience '{experiment_name}' introuvable dans MLflow.")

    runs = client.search_runs(experiment.experiment_id, filter_string=f"tags.mlflow.runName = '{run_name}'")

    if not runs:
        raise ValueError(f"Aucun run trouvé avec le nom '{run_name}'.")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    print(f"✅ Modèle chargé depuis MLflow - Run ID: {run_id}")
    return model, run_id

def load_data_from_mlflow(run_id, dataset_name="train"):
    """ Récupère le jeu de données sauvegardé dans MLflow. """
    client = mlflow.tracking.MlflowClient()

    try:
        X_path = client.download_artifacts(run_id, f"X_{dataset_name}.parquet")
        y_path = client.download_artifacts(run_id, f"y_{dataset_name}.parquet")
        X = pd.read_parquet(X_path)
        y = pd.read_parquet(y_path)
        print(f"📊 {dataset_name.capitalize()} chargé depuis MLflow.")
        return X, y
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lors du téléchargement des données depuis MLflow: {e}")

def format_metrics(metrics, train_errors, test_errors):
    """Formate les métriques sans notation scientifique."""
    # Configuration de pandas pour éviter la notation scientifique
    pd.set_option('display.float_format', lambda x: '%.4f' % x)

    # Statistiques descriptives des erreurs
    train_stats = pd.Series(train_errors).describe()
    test_stats = pd.Series(test_errors).describe()

    # Création du dictionnaire de métriques pour MLflow
    mlflow_metrics = {
        # Métriques principales
        "Train RMSE": metrics["Train RMSE"],
        "Train MAE": metrics["Train MAE"],
        "Train R2": metrics["Train R2"],
        "Test RMSE": metrics["Test RMSE"],
        "Test MAE": metrics["Test MAE"],
        "Test R2": metrics["Test R2"],

        # Statistiques des erreurs d'entraînement
        "Train_Error_Count": float(train_stats['count']),
        "Train_Error_Mean": float(train_stats['mean']),
        "Train_Error_Std": float(train_stats['std']),
        "Train_Error_Min": float(train_stats['min']),
        "Train_Error_25": float(train_stats['25%']),
        "Train_Error_50": float(train_stats['50%']),
        "Train_Error_75": float(train_stats['75%']),
        "Train_Error_Max": float(train_stats['max']),

        # Statistiques des erreurs de test
        "Test_Error_Count": float(test_stats['count']),
        "Test_Error_Mean": float(test_stats['mean']),
        "Test_Error_Std": float(test_stats['std']),
        "Test_Error_Min": float(test_stats['min']),
        "Test_Error_25": float(test_stats['25%']),
        "Test_Error_50": float(test_stats['50%']),
        "Test_Error_75": float(test_stats['75%']),
        "Test_Error_Max": float(test_stats['max'])
    }

    # Création du rapport formaté pour l'affichage
    report_lines = [
        "--- Metrics ---",
        f"Train RMSE: {metrics['Train RMSE']:.4f}",
        f"Train MAE: {metrics['Train MAE']:.4f}",
        f"Train R2: {metrics['Train R2']:.4f}",
        f"Test RMSE: {metrics['Test RMSE']:.4f}",
        f"Test MAE: {metrics['Test MAE']:.4f}",
        f"Test R2: {metrics['Test R2']:.4f}",
        "",
        "--- Absolute Errors (Train) ---",
        f"count    {train_stats['count']:.4f}",
        f"mean     {train_stats['mean']:.4f}",
        f"std      {train_stats['std']:.4f}",
        f"min      {train_stats['min']:.4f}",
        f"25%      {train_stats['25%']:.4f}",
        f"50%      {train_stats['50%']:.4f}",
        f"75%      {train_stats['75%']:.4f}",
        f"max      {train_stats['max']:.4f}",
        "",
        "--- Absolute Errors (Test) ---",
        f"count    {test_stats['count']:.4f}",
        f"mean     {test_stats['mean']:.4f}",
        f"std      {test_stats['std']:.4f}",
        f"min      {test_stats['min']:.4f}",
        f"25%      {test_stats['25%']:.4f}",
        f"50%      {test_stats['50%']:.4f}",
        f"75%      {test_stats['75%']:.4f}",
        f"max      {test_stats['max']:.4f}"
    ]

    return mlflow_metrics, "\n".join(report_lines)

def plot_regression_predictions(y_true, y_pred, title, filename, output_dir):
    """Crée un graphique des prédictions vs valeurs réelles."""
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
    """Crée un histogramme des erreurs de prédiction."""
    errors = y_true - y_pred
    error_std = np.std(errors)
    error_mean = np.mean(errors)
    x_min = error_mean - 4 * error_std
    x_max = error_mean + 4 * error_std

    plt.figure(figsize=(8, 6))
    sns.kdeplot(errors, color="blue", fill=True, alpha=0.3)  # Courbe de densité lissée
    plt.axvline(x=0, color='red', linestyle='--', label='Zero Error')
    plt.xlim(x_min, x_max)
    plt.xlabel("Erreur de prédiction")
    plt.ylabel("Densité")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def error_analysis(model, X_train, y_train, X_test, y_test, output_dir="artifacts"):
    """Effectue l'analyse des erreurs et enregistre les résultats dans MLflow."""
    os.makedirs(output_dir, exist_ok=True)

    if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
        raise ValueError("❌ Les jeux de données d'entraînement ou de test sont vides !")

    new_run_name = RUN_NAME.replace("train", "evaluate")

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=new_run_name, experiment_id=experiment.experiment_id):
        # Prédictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Conversion en numpy arrays 1D
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        y_train_pred = y_train_pred.ravel()
        y_test_pred = y_test_pred.ravel()

        # Calcul des métriques de base
        metrics = {
            "Train RMSE": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "Train MAE": float(mean_absolute_error(y_train, y_train_pred)),
            "Train R2": float(r2_score(y_train, y_train_pred)),
            "Test RMSE": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "Test MAE": float(mean_absolute_error(y_test, y_test_pred)),
            "Test R2": float(r2_score(y_test, y_test_pred)),
        }

        # Calcul des erreurs absolues
        absolute_errors_train = np.abs(y_train - y_train_pred)
        absolute_errors_test = np.abs(y_test - y_test_pred)

        # Génération des métriques et du rapport
        mlflow_metrics, report_text = format_metrics(metrics, absolute_errors_train, absolute_errors_test)

        # Log des métriques dans MLflow
        mlflow.log_metrics(mlflow_metrics)

        # Affichage du rapport
        print(report_text)

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

        # Log des visualisations dans MLflow
        mlflow.log_artifacts(output_dir)

        return metrics

if __name__ == "__main__":
    print("📥 Chargement des données et modèle...")

    try:
        model, train_run_id = load_model_from_mlflow(EXPERIMENT_NAME, RUN_NAME)
        X_train, y_train = load_data_from_mlflow(train_run_id, "train")
        X_test, y_test = load_data_from_mlflow(train_run_id, "test")

        print("🚀 Analyse des erreurs et évaluation du modèle...")
        metrics = error_analysis(model, X_train, y_train, X_test, y_test)
        print("✔️ Analyse terminée !")
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution : {e}")