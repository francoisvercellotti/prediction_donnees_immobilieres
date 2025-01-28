
from src.load_data import load_data
import pandas as pd
import os
import tempfile


def create_temp_file(extension, content):
    """
    Crée un fichier temporaire avec le contenu donné et retourne son chemin.

    Args:
        extension (str): Extension du fichier (e.g., '.csv', '.json').
        content (str): Contenu du fichier.

    Returns:
        str: Chemin vers le fichier temporaire.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension, mode='w')
    temp_file.write(content)
    temp_file.close()
    return temp_file.name

def test_load_csv():
    """Test pour un fichier CSV."""
    content = "prix,id_ville,ville\n100000,1,Paris\n200000,2,Lyon"
    filepath = create_temp_file(".csv", content)
    try:
        df = load_data(filepath, delimiter=',')
        assert df.shape == (2, 3), "Le DataFrame chargé ne correspond pas aux dimensions attendues."
        assert "prix" in df.columns, "La colonne 'prix' est manquante."
        print("Test CSV passé.")
    finally:
        os.remove(filepath)

def test_load_json():
    """Test pour un fichier JSON."""
    content = '[{"prix": 100000, "id_ville": 1, "ville": "Paris"}, {"prix": 200000, "id_ville": 2, "ville": "Lyon"}]'
    filepath = create_temp_file(".json", content)
    try:
        df = load_data(filepath)
        assert df.shape == (2, 3), "Le DataFrame chargé ne correspond pas aux dimensions attendues."
        assert "ville" in df.columns, "La colonne 'ville' est manquante."
        print("Test JSON passé.")
    finally:
        os.remove(filepath)

def test_load_excel():
    """Test pour un fichier Excel."""
    data = {
        "prix": [100000, 200000],
        "id_ville": [1, 2],
        "ville": ["Paris", "Lyon"]
    }
    df = pd.DataFrame(data)
    filepath = create_temp_file(".xlsx", "")  # Créer un fichier temporaire vide
    try:
        # Sauvegarder le DataFrame en Excel
        df.to_excel(filepath, index=False)
        df_loaded = load_data(filepath)
        assert df_loaded.shape == (2, 3), "Le DataFrame chargé ne correspond pas aux dimensions attendues."
        assert "id_ville" in df_loaded.columns, "La colonne 'id_ville' est manquante."
        print("Test Excel passé.")
    finally:
        os.remove(filepath)

def test_load_txt():
    """Test pour un fichier TXT."""
    content = "prix;id_ville;ville\n100000;1;Paris\n200000;2;Lyon"
    filepath = create_temp_file(".txt", content)
    try:
        df = load_data(filepath, delimiter=';')
        assert df.shape == (2, 3), "Le DataFrame chargé ne correspond pas aux dimensions attendues."
        assert "ville" in df.columns, "La colonne 'ville' est manquante."
        print("Test TXT passé.")
    finally:
        os.remove(filepath)

def run_tests():
    """Exécute tous les tests."""
    test_load_csv()
    test_load_json()
    test_load_excel()
    test_load_txt()
    print("Tous les tests sont passés avec succès.")

if __name__ == "__main__":
    run_tests()
