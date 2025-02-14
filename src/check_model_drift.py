"""
Script d'évaluation de la dérive d'un modèle de régression avec MLflow.

Ce script permet de charger un modèle entraîné depuis MLflow, d'évaluer ses performances
sur des données de référence et actuelles, et de détecter une éventuelle dérive du modèle
en comparant les métriques obtenues.

Fonctionnalités principales :
1. Chargement du dernier modèle entraîné et des données de test depuis MLflow.
2. Évaluation des performances du modèle sur deux sous-ensembles des données de test.
3. Détection de la dérive du modèle en comparant les métriques obtenues avec un seuil défini.
4. Visualisation des métriques et du niveau de dérive sous forme de graphique.
5. Enregistrement des résultats et des visualisations dans MLflow pour suivi.

Modules utilisés :
- `mlflow` pour le suivi des modèles et des métriques.
- `pandas` et `numpy` pour la manipulation des données.
- `sklearn.metrics` pour le calcul des métriques d'évaluation.
- `matplotlib.pyplot` et `seaborn` pour la visualisation des résultats.
- `os` et `logging` pour la gestion des fichiers temporaires et des logs.

Entrées :
- Le nom de l'expérience MLflow contenant les modèles entraînés.
- Un seuil de détection de dérive (par défaut 10%).

Sorties :
- Un dictionnaire contenant l'analyse de la dérive.
- Une image de visualisation des différences de métriques.
- Des logs et artifacts stockés dans MLflow.

"""


import os
from datetime import datetime
import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_data_from_mlflow(parent_run) -> Tuple[object, pd.DataFrame, pd.DataFrame]:
    """
    Charge le modèle et les données de test depuis les artifacts MLflow du run parent.
    """
    try:
        # Charger le modèle
        model_uri = f"runs:/{parent_run.info.run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Modèle chargé depuis MLflow")

        # Créer le dossier temporaire s'il n'existe pas
        local_dir = "temp_artifacts"
        os.makedirs(local_dir, exist_ok=True)
        logger.info(f"Dossier temporaire créé: {local_dir}")

        # Charger les données de test
        client = MlflowClient()

        # Télécharger les artifacts dans le dossier temporaire
        client.download_artifacts(parent_run.info.run_id, "features_test.parquet", local_dir)
        client.download_artifacts(parent_run.info.run_id, "target_test.parquet", local_dir)

        # Charger les données
        features_test = pd.read_parquet(os.path.join(local_dir, "features_test.parquet"))
        target_test = pd.read_parquet(os.path.join(local_dir, "target_test.parquet"))

        # Créer deux sous-ensembles pour la référence et le courant
        test_size = len(features_test)
        split_idx = test_size // 2

        reference_data = pd.concat([
            features_test.iloc[:split_idx],
            target_test.iloc[:split_idx]],
            axis=1
        )

        current_data = pd.concat([
            features_test.iloc[split_idx:],
            target_test.iloc[split_idx:]],
            axis=1
        )

        logger.info("Données chargées depuis MLflow")

        # Nettoyer le dossier temporaire après utilisation
        for file in os.listdir(local_dir):
            os.remove(os.path.join(local_dir, file))
        os.rmdir(local_dir)
        logger.info("Dossier temporaire nettoyé")

        return model, reference_data, current_data

    except Exception as e:
        logger.error(f"Erreur lors du chargement depuis MLflow: {str(e)}")
        raise

def evaluate_model(model, data: pd.DataFrame) -> Dict:
    """
    Évalue les performances du modèle sur un jeu de données.
    """
    try:
        # Séparer features et target
        target_col = data.columns[-1]  # Suppose que la target est la dernière colonne
        X = data.drop(target_col, axis=1)
        y = data[target_col]

        # Faire les prédictions
        y_pred = model.predict(X)

        # Calculer les métriques
        results = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }

        return results

    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation du modèle: {str(e)}")
        raise

def check_model_drift(reference_metrics: Dict, current_metrics: Dict,
                     threshold: float = 0.1) -> Dict:
    """
    Compare les performances entre les données de référence et actuelles.
    """
    try:
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'metrics_drift': {},
            'has_drift': False
        }

        # Calculer le drift pour chaque métrique
        for metric in ['mse', 'rmse', 'mae', 'r2']:
            ref_value = reference_metrics[metric]
            curr_value = current_metrics[metric]

            # Calculer la différence relative
            if ref_value != 0:
                relative_diff = abs(curr_value - ref_value) / abs(ref_value)
            else:
                relative_diff = abs(curr_value - ref_value)

            drift_results['metrics_drift'][metric] = {
                'reference_value': ref_value,
                'current_value': curr_value,
                'relative_diff': relative_diff,
                'has_drift': relative_diff > threshold
            }

            if relative_diff > threshold:
                drift_results['has_drift'] = True

        return drift_results

    except Exception as e:
        logger.error(f"Erreur lors de la vérification du drift: {str(e)}")
        raise

def create_drift_visualization(reference_metrics: Dict, current_metrics: Dict, drift_results: Dict) -> str:
    """
    Crée une visualisation comparative des métriques et du drift.
    """
    # Créer un dossier temporaire pour les visualisations
    os.makedirs("temp_viz", exist_ok=True)

    # Préparer les données pour la visualisation
    metrics_comparison = pd.DataFrame({
        'Metric': list(reference_metrics.keys()),
        'Reference': list(reference_metrics.values()),
        'Current': list(current_metrics.values()),
        'Drift (%)': [drift_results['metrics_drift'][m]['relative_diff'] * 100 for m in reference_metrics.keys()]
    })

    # Créer la visualisation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Réorganiser les données pour le graphique en barres
    metrics_melted = pd.melt(metrics_comparison,
                            id_vars=['Metric'],
                            value_vars=['Reference', 'Current'],
                            var_name='Dataset',
                            value_name='Value')

    # Graphique des métriques
    sns.barplot(data=metrics_melted,
                x='Metric',
                y='Value',
                hue='Dataset',
                ax=ax1)
    ax1.set_title('Comparison of Metrics: Reference vs Current')
    ax1.set_ylabel('Value')

    # Graphique du drift
    sns.barplot(data=metrics_comparison,
                x='Metric',
                y='Drift (%)',
                ax=ax2,
                color='steelblue')
    ax2.set_title('Drift Percentage by Metric')
    ax2.set_ylabel('Drift (%)')
    ax2.axhline(y=10, color='r', linestyle='--', label='Threshold (10%)')
    ax2.legend()

    plt.tight_layout()

    # Sauvegarder la figure
    viz_path = "temp_viz/drift_analysis.png"
    plt.savefig(viz_path)
    plt.close()

    return viz_path

def get_latest_training_run(experiment_name: str) -> str:
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

def log_results_to_mlflow(drift_results: Dict,
                         reference_metrics: Dict,
                         current_metrics: Dict,
                         parent_run,
                         viz_path: str = None):
    """
    Enregistre les résultats dans MLflow comme un child run.
    """
    try:
        # Créer un child run pour le contrôle de drift du modèle
        with mlflow.start_run(run_id=parent_run.info.run_id):
            with mlflow.start_run(run_name="model_drift_check",
                                nested=True,
                                description="Analyse du drift du modèle"):

                # Log des paramètres
                mlflow.log_params({
                    'check_type': 'model_drift',
                    'timestamp': drift_results['timestamp'],
                    'has_drift': drift_results['has_drift']
                })

                # Log des métriques de référence
                for metric, value in reference_metrics.items():
                    mlflow.log_metric(f"reference_{metric}", value)

                # Log des métriques actuelles
                for metric, value in current_metrics.items():
                    mlflow.log_metric(f"current_{metric}", value)

                # Log des drifts
                for metric, drift_info in drift_results['metrics_drift'].items():
                    mlflow.log_metric(f"drift_{metric}", drift_info['relative_diff'])

                # Log du rapport complet
                mlflow.log_dict(drift_results, "model_drift_analysis.json")

                # Log de la visualisation si disponible
                if viz_path and os.path.exists(viz_path):
                    mlflow.log_artifact(viz_path)

    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement dans MLflow: {str(e)}")
        raise
    finally:
        # Nettoyage des fichiers temporaires
        if viz_path and os.path.exists(viz_path):
            os.remove(viz_path)
            os.rmdir(os.path.dirname(viz_path))

def main():
    try:
        # Configuration de MLflow avec le bon experiment
        EXPERIMENT_NAME = "regression_model_prediction_immobilières"
        mlflow.set_tracking_uri("file:" + os.path.abspath("mlruns"))
        mlflow.set_experiment(EXPERIMENT_NAME)

        # Récupérer le dernier run d'entraînement
        parent_run = get_latest_training_run(EXPERIMENT_NAME)
        logger.info(f"Run parent trouvé: {parent_run.info.run_id}")

        # Charger le modèle et les données depuis MLflow
        model, reference_data, current_data = load_model_and_data_from_mlflow(parent_run)

        # Évaluer le modèle sur les deux jeux de données
        reference_metrics = evaluate_model(model, reference_data)
        current_metrics = evaluate_model(model, current_data)

        # Vérifier le drift
        drift_results = check_model_drift(reference_metrics, current_metrics)

        # Créer la visualisation
        viz_path = create_drift_visualization(reference_metrics, current_metrics, drift_results)

        # Logger les résultats dans MLflow
        log_results_to_mlflow(drift_results, reference_metrics, current_metrics, parent_run, viz_path)

        # Afficher les résultats
        if drift_results['has_drift']:
            logger.warning("⚠️ Drift détecté dans les performances du modèle!")
            logger.warning("Résumé des drifts détectés:")
            for metric, info in drift_results['metrics_drift'].items():
                if info['has_drift']:
                    logger.warning(f"- {metric}: {info['relative_diff']:.2%} de différence")
                    logger.warning(f"  Référence: {info['reference_value']:.4f}")
                    logger.warning(f"  Actuel: {info['current_value']:.4f}")
        else:
            logger.info("✅ Aucun drift significatif détecté dans les performances du modèle.")
            logger.info("Résumé des métriques:")
            for metric, info in drift_results['metrics_drift'].items():
                logger.info(f"- {metric}: {info['relative_diff']:.2%} de différence")

    except Exception as e:
        logger.error(f"Erreur critique: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
