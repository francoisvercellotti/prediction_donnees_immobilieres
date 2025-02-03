import sys
import mlflow.pyfunc
import pandas as pd
import argparse

def load_model(model_uri):
    """
    Charge le modèle à partir de son URI avec mlflow.pyfunc.

    :param model_uri: URI du modèle (par exemple, "runs:/<run_id>/model")
    :return: objet modèle chargé
    """
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Modèle chargé depuis {model_uri}")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        sys.exit(1)

def predict(model, input_data):
    """
    Effectue la prédiction sur un DataFrame Pandas.

    :param model: modèle chargé via MLflow
    :param input_data: DataFrame Pandas contenant les données à prédire
    :return: résultats de la prédiction
    """
    try:
        predictions = model.predict(input_data)
        return predictions
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Script de prédiction utilisant un modèle MLflow")
    parser.add_argument("--model_uri", type=str, required=True,
                        help="URI du modèle MLflow (ex: runs:/<run_id>/model)")
    parser.add_argument("--data", type=str, required=True,
                        help="Chemin vers le fichier CSV contenant les données d'entrée")

    args = parser.parse_args()

    # Charger le modèle
    model = load_model(args.model_uri)

    # Lire les données d'entrée
    try:
        data_df = pd.read_csv(args.data)
        print(f"Les données ont été chargées depuis {args.data}")
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        sys.exit(1)

    # Effectuer la prédiction
    predictions = predict(model, data_df)

    # Afficher ou sauvegarder les prédictions
    print("Prédictions:")
    print(predictions)

if __name__ == "__main__":
    main()