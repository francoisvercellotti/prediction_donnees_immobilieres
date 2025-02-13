"""
Module pour l'analyse des erreurs dans un modèle de régression, en calculant/
diverses métriques de performance,
en générant des graphiques de prédictions et d'erreurs,
et en enregistrant ces résultats dans MLflow.

Ce module contient les fonctions suivantes :
- `error_analysis`: Effectue une analyse complète des erreurs pour un modèle de régression, en calculant des métriques
  de performance, en générant des graphiques de prédictions et en enregistrant les résultats dans MLflow.
- `format_metrics`: Formate les métriques de performance pour l'affichage et l'enregistrement dans MLflow.
- `plot_regression_predictions`: Génère un graphique des prédictions par rapport aux valeurs réelles.
- `plot_regression_error`: Génère un graphique de la distribution des erreurs.

Les graphiques générés sont enregistrés dans un répertoire temporaire et ensuite ajoutés au run MLflow en tant qu'artéfacts.

Les métriques calculées incluent :
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

Ce module est conçu pour fournir une évaluation détaillée de la performance du modèle, aider à la visualisation des erreurs
et assurer le suivi des résultats via MLflow.

Il est recommandé d'utiliser ce module après l'entraînement d'un modèle pour analyser ses erreurs et son ajustement
sur les jeux de données d'entraînement et de test.
"""


import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# Configuration de MLflow
mlflow.set_tracking_uri("http://localhost:5000")
EXPERIMENT_NAME = "regression_model_prediction_immobilières"
RUN_NAME = "run_train_model_v1_dataset_v1"

def load_model_from_mlflow(experiment_name, run_name):
    """
    Récupère un modèle sauvegardé dans MLflow pour un run donné.

    Args:
        experiment_name (str): Le nom de l'expérience dans MLflow.
        run_name (str): Le nom du run dont on souhaite récupérer le modèle.

    Returns:
        model: Le modèle sauvegardé dans MLflow.
        run_id (str): L'ID du run correspondant au modèle chargé.

    Raises:
        ValueError: Si l'expérience ou le run n'est pas trouvé dans MLflow.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        raise ValueError(f"Expérience '{experiment_name}' introuvable dans MLflow.")

    runs = client.search_runs(experiment.experiment_id,
                              filter_string=f"tags.mlflow.runName = '{run_name}'")

    if not runs:
        raise ValueError(f"Aucun run trouvé avec le nom '{run_name}'.")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    print(f"✅ Modèle chargé depuis MLflow - Run ID: {run_id}")
    return model, run_id

def load_data_from_mlflow(run_id, dataset_name="train"):
    """
    Récupère les jeux de données (X et y) sauvegardés dans MLflow pour un run donné.

    Args:
        run_id (str): L'ID du run pour lequel on veut charger les jeux de données.
        dataset_name (str): Le nom du jeu de données à charger ('train' ou 'test').

    Returns:
        tuple: Un tuple contenant les données X et y chargées depuis MLflow.

    Raises:
        RuntimeError: Si un problème survient lors du téléchargement des données depuis MLflow.
    """
    client = mlflow.tracking.MlflowClient()

    try:
        X_path = client.download_artifacts(run_id, f"features_{dataset_name}.parquet")
        y_path = client.download_artifacts(run_id, f"target_{dataset_name}.parquet")
        X = pd.read_parquet(X_path)
        y = pd.read_parquet(y_path)
        print(f"📊 {dataset_name.capitalize()} chargé depuis MLflow.")
        return X, y
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lors du téléchargement des données depuis MLflow: {e}")

def format_metrics(metrics, train_errors, test_errors):
    """
    Formate les métriques d'évaluation du modèle et génère un rapport sous forme de texte.

    Args:
        metrics (dict): Un dictionnaire contenant les métriques de performance du modèle.
        train_errors (list): Liste des erreurs absolues pour l'ensemble d'entraînement.
        test_errors (list): Liste des erreurs absolues pour l'ensemble de test.

    Returns:
        tuple: Un tuple contenant un dictionnaire de métriques formatées pour MLflow et un texte de rapport.
    """
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
    """
    Crée un graphique comparant les prédictions du modèle aux valeurs réelles.

    Args:
        y_true (array-like): Valeurs réelles à comparer.
        y_pred (array-like): Prédictions du modèle.
        title (str): Le titre du graphique.
        filename (str): Nom du fichier pour sauvegarder le graphique.
        output_dir (str): Répertoire où sauvegarder le graphique.
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
    Crée un histogramme de la distribution des erreurs de prédiction.

    Args:
        y_true (array-like): Valeurs réelles.
        y_pred (array-like): Prédictions du modèle.
        title (str): Titre du graphique.
        filename (str): Nom du fichier pour sauvegarder le graphique.
        output_dir (str): Répertoire où sauvegarder le graphique.
    """
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

def error_analysis(model, X_train, y_train, X_test, y_test):
    """
    Effectue une analyse des erreurs en termes de prédiction et enregistre les résultats dans MLflow.

    Args:
        model (sklearn model): Modèle entraîné pour effectuer les prédictions.
        x_train (DataFrame): Données d'entrée d'entraînement.
        y_train (DataFrame): Cibles d'entraînement.
        x_test (DataFrame): Données d'entrée de test.
        y_test (DataFrame): Cibles de test.
    """
    # Utilisation d'un répertoire temporaire
    with tempfile.TemporaryDirectory() as output_dir:
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
            mlflow_metrics, report_text = format_metrics(metrics, absolute_errors_train,
                                                         absolute_errors_test)

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
