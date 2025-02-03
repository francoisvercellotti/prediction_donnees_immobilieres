# Variables
PYTHON = python
DATA_DIR = data
LOADED_DIR = $(DATA_DIR)/loaded
INTERIM_DIR = $(DATA_DIR)/interim
MODELS_DIR = models
INPUT_FILE ?= $(DATA_DIR)/raw/transactions_immobilieres.parquet

# Création des répertoires nécessaires
$(LOADED_DIR) $(INTERIM_DIR) $(MODELS_DIR):
	mkdir -p $@

# Étape 1: Chargement des données
$(LOADED_DIR)/loaded_dataset.parquet: | $(LOADED_DIR)
	$(PYTHON) src/load_data.py $(INPUT_FILE)
load: $(LOADED_DIR)/loaded_dataset.parquet

# Étape 2: Feature Engineering
$(INTERIM_DIR)/prepared_dataset.parquet: $(LOADED_DIR)/loaded_dataset.parquet | $(INTERIM_DIR)
	$(PYTHON) src/feature_engineering.py
feature_engineering: $(INTERIM_DIR)/prepared_dataset.parquet

# Étape 3: Split et Encodage
split_and_encode: $(INTERIM_DIR)/prepared_dataset.parquet
	$(PYTHON) src/split_and_encode.py

# Étape 4: Preprocessing
preprocess: $(INTERIM_DIR)/prepared_dataset.parquet
	$(PYTHON) src/preprocessing_pipeline.py

# Étape 5: Entraînement du modèle
train: $(INTERIM_DIR)/prepared_dataset.parquet
	$(PYTHON) src/train_and_log_model_with_mlflow.py

# Étape 6: Évaluation du modèle
evaluate:
	$(PYTHON) src/evaluate_model.py

# Nettoyer les fichiers générés
clean:
	rm -rf $(LOADED_DIR)/*
	rm -rf $(INTERIM_DIR)/*
	rm -rf $(MODELS_DIR)/*

# Exécuter tout le pipeline
all: load feature_engineering split_and_encode preprocess train evaluate

# Aide
help:
	@echo "Pipeline ML - Commandes disponibles:"
	@echo "make load INPUT_FILE=<chemin_fichier>  : Charger les données"
	@echo "make feature_engineering              : Effectuer le feature engineering"
	@echo "make split_and_encode                 : Split et encodage des données"
	@echo "make preprocess                      : Prétraitement des données"
	@echo "make train                           : Entraîner le modèle"
	@echo "make evaluate                        : Évaluer le modèle"
	@echo "make all INPUT_FILE=<chemin_fichier>  : Exécuter tout le pipeline"
	@echo "make clean                           : Nettoyer les fichiers générés"

.PHONY: clean all help load feature_engineering split_and_encode preprocess train evaluate