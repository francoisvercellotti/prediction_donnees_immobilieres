import pandas as pd
import mlflow
import tempfile
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric

# Configuration de MLflow
mlflow.set_tracking_uri('http://localhost:5000')
EXPERIMENT_NAME = "regression_model_drift_check"

# Vérifier si l'expériment existe déjà, sinon le créer
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

def prepare_data_for_drift_check():
    """ Charger et recombiner X_train, X_test, y_train, y_test pour l'analyse de drift. """
    X_train = pd.read_parquet("data/interim/preprocessed/features_train.parquet")
    X_test = pd.read_parquet("data/interim/preprocessed/features_test.parquet")
    y_train = pd.read_parquet("data/interim/preprocessed/target_train.parquet")
    y_test = pd.read_parquet("data/interim/preprocessed/target_test.parquet")

    reference_data = X_train.copy()
    reference_data['target'] = y_train

    current_data = X_test.copy()
    current_data['target'] = y_test

    return reference_data, current_data

def check_drift(run_name="drift_analysis"):
    """ Analyser le drift des données, générer un rapport en HTML et logger les résultats dans MLflow. """
    reference_data, current_data = prepare_data_for_drift_check()

    # Créer le rapport de drift
    report = Report(metrics=[DatasetDriftMetric(), ColumnDriftMetric(column_name="target")])

    # Calculer les métriques
    report.run(reference_data=reference_data, current_data=current_data)

    # Démarrer un nouveau run dans MLflow avec un nom personnalisé
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        # Enregistrer les métriques dans MLflow
        dataset_drift_result = report.metrics[0].get_result()
        target_drift_result = report.metrics[1].get_result()

        mlflow.log_metric("dataset_drift_detected", int(dataset_drift_result.dataset_drift))
        mlflow.log_metric("target_drift_score", target_drift_result.drift_score)
        mlflow.log_metric("target_drift_detected", int(target_drift_result.drift_score > 0.1))

        # Générer et sauvegarder le rapport HTML temporairement
        with tempfile.TemporaryDirectory() as tmpdirname:
            report_path = f"{tmpdirname}/drift_report.html"
            report.save_html(report_path)

            # Logger le rapport HTML dans les artefacts du run actuel
            mlflow.log_artifact(report_path, artifact_path="drift_reports")

    # Afficher les résultats dans la console
    print("\nAnalyse du drift des données:")
    print("-" * 30)
    if dataset_drift_result.dataset_drift:
        print(f"⚠️ Drift détecté dans le dataset: {dataset_drift_result.dataset_drift}")
    if target_drift_result.drift_score > 0.1:
        print(f"⚠️ Drift détecté dans la variable cible avec score: {target_drift_result.drift_score:.4f}")

    return 1 if (dataset_drift_result.dataset_drift or target_drift_result.drift_score > 0.1) else 0

if __name__ == "__main__":
    exit(check_drift(run_name="Test_Drift_01"))