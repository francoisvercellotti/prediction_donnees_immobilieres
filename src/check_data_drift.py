"""
Module d'analyse du drift des données et enregistrement des résultats dans MLflow.

Ce module permet de :
1. Charger les données de référence et actuelles.
2. Effectuer une analyse du drift des données à l'aide de la bibliothèque Evidently.
3. Générer des visualisations pour l'analyse du drift.
4. Enregistrer les résultats et visualisations dans MLflow pour le suivi.

Fonctionnalités :
- `load_reference_and_current_data()`: Charge les données actuelles et définit les données de référence si elles n'existent pas encore.
- `perform_drift_analysis()`: Analyse le drift des données en comparant l'ensemble de référence et l'ensemble actuel.
- `create_drift_visualizations()`: Génère des graphiques illustrant le drift des caractéristiques les plus affectées.
- `update_log_results_to_mlflow()`: Enregistre les résultats de l'analyse du drift ainsi que les visualisations dans MLflow.
- `log_results_to_mlflow()`: Enregistre uniquement les résultats du drift dans MLflow sans les visualisations.

Dépendances :
- pandas
- mlflow
- logging
- os
- datetime
- evidently
- seaborn
- matplotlib.pyplot
"""

import logging
import os
from typing import Tuple, Dict, List
from datetime import datetime
import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
import seaborn as sns
import matplotlib.pyplot as plt


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_reference_and_current_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les données de référence et actuelles.
    Pour le premier entraînement, crée une référence à partir des données actuelles.
    """
    try:
        reference_path = "data/reference/reference_dataset.parquet"
        current_data = pd.read_parquet("data/interim/prepared_dataset.parquet")

        if not os.path.exists(reference_path):
            logger.info("Premier entraînement détecté - Création des données de référence")

            if 'date_mutation' in current_data.columns:
                current_data = current_data.sort_values('date_mutation')
                split_idx = int(len(current_data) * 0.8)
                reference_data = current_data.iloc[:split_idx].copy()
                current_data = current_data.iloc[split_idx:].copy()
            else:
                reference_data = current_data.sample(frac=0.8, random_state=42)
                current_data = current_data.drop(reference_data.index)

            os.makedirs(os.path.dirname(reference_path), exist_ok=True)
            reference_data.to_parquet(reference_path)
            logger.info(f"Données de référence créées et sauvegardées dans {reference_path}")
        else:
            reference_data = pd.read_parquet(reference_path)

        return reference_data, current_data

    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {str(e)}")
        raise

def perform_drift_analysis(reference_data: pd.DataFrame,
                           current_data: pd.DataFrame,
                           drift_threshold: float = 0.05) -> Dict:
    """
    Effectue l'analyse de drift sur les données avec la nouvelle API Evidently.
    """
    try:
        # Créer le rapport avec les métriques de drift
        metrics = [DatasetDriftMetric()]
        column_metrics = [ColumnDriftMetric(column_name=col) for col in reference_data.columns]
        metrics.extend(column_metrics)

        report = Report(metrics=metrics)
        logger.info("Exécution du rapport Evidently pour l'analyse du drift.")
        report.run(reference_data=reference_data, current_data=current_data)
        report_dict = report.as_dict()

        # Extraire les résultats
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(reference_data.columns),
            'drifted_features': 0,
            'drift_scores': {},
        }

        # Récupérer le dataset drift depuis la première métrique
        dataset_metric = report_dict['metrics'][0]
        results['dataset_drift'] = dataset_metric['result']['dataset_drift']

        # Analyser les résultats pour chaque colonne
        for col, metric in zip(reference_data.columns, report_dict['metrics'][1:]):
            try:
                drift_score = metric['result']['drift_score']
                results['drift_scores'][col] = drift_score
                if drift_score > drift_threshold:
                    results['drifted_features'] += 1
            except KeyError as e:
                logger.warning(f"Impossible d'obtenir le drift score pour la colonne {col}: {str(e)}")
                results['drift_scores'][col] = None

        return results

    except Exception as e:
        logger.error(f"Erreur lors de l'analyse du drift: {str(e)}")
        raise


def create_drift_visualizations(reference_data: pd.DataFrame,
                              current_data: pd.DataFrame,
                              drift_results: Dict,
                              drift_threshold: float = 0.05) -> List[str]:
    """
    Crée des visualisations pour l'analyse du drift des données.
    Retourne une liste des chemins des visualisations générées.
    """
    # Créer le dossier pour les visualisations
    viz_dir = "temp_viz"
    os.makedirs(viz_dir, exist_ok=True)
    viz_paths = []

    # 1. Heatmap des scores de drift
    plt.figure(figsize=(12, 6))
    drift_scores = pd.Series(drift_results['drift_scores'])
    drift_scores = drift_scores.sort_values(ascending=False)

    sns.heatmap(drift_scores.to_frame().T,
                cmap='YlOrRd',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Drift Score'})
    plt.title('Feature Drift Scores Heatmap')
    plt.xlabel('Features')
    plt.xticks(rotation=45, ha='right')

    heatmap_path = f"{viz_dir}/drift_scores_heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    viz_paths.append(heatmap_path)

    # 2. Distribution des scores de drift
    plt.figure(figsize=(10, 6))
    sns.histplot(data=drift_scores, bins=20)
    plt.axvline(x=drift_threshold, color='r', linestyle='--',
                label=f'Drift Threshold ({drift_threshold})')
    plt.title('Distribution of Drift Scores')
    plt.xlabel('Drift Score')
    plt.ylabel('Count')
    plt.legend()

    dist_path = f"{viz_dir}/drift_scores_distribution.png"
    plt.tight_layout()
    plt.savefig(dist_path)
    plt.close()
    viz_paths.append(dist_path)

    # 3. Comparaison des distributions pour les features les plus driftées
    top_drifted = drift_scores.nlargest(min(5, len(drift_scores)))
    fig, axes = plt.subplots(len(top_drifted), 1, figsize=(12, 4*len(top_drifted)))
    if len(top_drifted) == 1:
        axes = [axes]

    for ax, (feature, score) in zip(axes, top_drifted.items()):
        # Créer un histogramme normalisé pour comparer les distributions
        sns.histplot(data=reference_data, x=feature, stat='density',
                    alpha=0.5, label='Reference', ax=ax)
        sns.histplot(data=current_data, x=feature, stat='density',
                    alpha=0.5, label='Current', ax=ax)
        ax.set_title(f'{feature} (Drift Score: {score:.3f})')
        ax.legend()

    plt.tight_layout()
    dist_comp_path = f"{viz_dir}/top_features_distribution_comparison.png"
    plt.savefig(dist_comp_path)
    plt.close()
    viz_paths.append(dist_comp_path)

    return viz_paths

def update_log_results_to_mlflow(results: Dict, reference_data: pd.DataFrame,
                                current_data: pd.DataFrame, drift_threshold: float = 0.05):
    """
    Version mise à jour de la fonction log_results_to_mlflow incluant les visualisations.
    """
    try:
        # Configuration de MLflow
        mlflow.set_tracking_uri("file:" + os.path.abspath("mlruns"))
        experiment_name = "regression_model_prediction_immobilières"
        mlflow.set_experiment(experiment_name)
        logger.info("Expérience utilisée : %s", experiment_name)

        parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
        if not parent_run_id:
            logger.error("Aucun run parent trouvé dans MLFLOW_PARENT_RUN_ID.")

        # Créer les visualisations
        viz_paths = create_drift_visualizations(reference_data, current_data,
                                              results, drift_threshold)

        # Créer un run enfant pour le contrôle de drift
        with mlflow.start_run(run_name="data_drift_check", nested=True,
                            description="Analyse du drift des données"):
            # Log des paramètres et métriques existants
            mlflow.log_params({
                'check_type': 'data_drift',
                'timestamp': results['timestamp'],
                'dataset_drift_detected': results['dataset_drift'],
                'drift_threshold': drift_threshold
            })

            mlflow.log_metrics({
                'drifted_features': results['drifted_features'],
                'total_features': results['total_features']
            })

            for feature, score in results['drift_scores'].items():
                if score is not None:
                    mlflow.log_metric(f"drift_score_{feature}", score)

            # Log du rapport complet
            mlflow.log_dict(results, "data_drift_analysis.json")

            # Log des visualisations
            for viz_path in viz_paths:
                mlflow.log_artifact(viz_path)

            logger.info("Résultats et visualisations du drift enregistrés dans MLflow")

    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement dans MLflow: {str(e)}")
        raise
    finally:
        # Nettoyage des fichiers temporaires
        for viz_path in viz_paths:
            if os.path.exists(viz_path):
                os.remove(viz_path)
        if os.path.exists('temp_viz'):
            os.rmdir('temp_viz')


def log_results_to_mlflow(results: Dict):
    """
    Enregistre les résultats dans MLflow dans un run enfant (nested) du run parent actif.
    """
    try:
        # Configuration de MLflow
        mlflow.set_tracking_uri("file:" + os.path.abspath("mlruns"))
        experiment_name = "regression_model_prediction_immobilières"
        mlflow.set_experiment(experiment_name)
        logger.info("Expérience utilisée : %s", experiment_name)

        # On suppose que le run parent a été créé et son ID transmis via la variable d'environnement
        parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
        if parent_run_id:
            logger.info("Run parent trouvé via MLFLOW_PARENT_RUN_ID : %s", parent_run_id)
        else:
            logger.error("Aucun run parent n'a été trouvé dans la variable d'environnement MLFLOW_PARENT_RUN_ID.")

        # Créer un run enfant (nested) pour le contrôle de drift
        with mlflow.start_run(run_name="data_drift_check", nested=True, description="Analyse du drift des données"):
            mlflow.log_params({
                'check_type': 'data_drift',
                'timestamp': results['timestamp'],
                'dataset_drift_detected': results['dataset_drift']
            })
            mlflow.log_metrics({
                'drifted_features': results['drifted_features'],
                'total_features': results['total_features']
            })
            for feature, score in results['drift_scores'].items():
                if score is not None:
                    mlflow.log_metric(f"drift_score_{feature}", score)
            mlflow.log_dict(results, "data_drift_analysis.json")
            logger.info("Résultats du drift enregistrés dans MLflow - Experiment: %s", experiment_name)

    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement dans MLflow: {str(e)}")
        raise

def main():
    try:
        mlflow.set_experiment("regression_model_prediction_immobilières")
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as parent_run:
            os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run.info.run_id
            logger.info("Run parent créé avec l'ID : %s", parent_run.info.run_id)

            # Charger les données et effectuer l'analyse du drift
            reference_data, current_data = load_reference_and_current_data()
            results = perform_drift_analysis(reference_data, current_data)

            # Utiliser la nouvelle version avec visualisations
            update_log_results_to_mlflow(results, reference_data, current_data)

            if results['dataset_drift']:
                logger.warning("Drift détecté dans %s features sur %s!",
                             results['drifted_features'], results['total_features'])
                if results['drifted_features'] / results['total_features'] > 0.2:
                    logger.error("Drift critique détecté! Arrêt du pipeline.")
                    exit(1)

            logger.info("Analyse du drift des données terminée avec succès.")
    except Exception as e:
        logger.error(f"Erreur critique: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()
