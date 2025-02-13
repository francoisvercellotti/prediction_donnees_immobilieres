# Variables
PYTHON = python
DATA_DIR = data
LOADED_DIR = $(DATA_DIR)/loaded
INTERIM_DIR = $(DATA_DIR)/interim
MODELS_DIR = models
REPORTS_DIR = reports
INPUT_FILE ?= $(DATA_DIR)/raw/transactions_immobilieres.parquet


# Création des répertoires nécessaires
$(LOADED_DIR) $(INTERIM_DIR) $(MODELS_DIR) $(REPORTS_DIR):
	mkdir -p $@

# Étape 1: Chargement des données
$(LOADED_DIR)/loaded_dataset.parquet: | $(LOADED_DIR)
	$(PYTHON) src/load_data.py $(INPUT_FILE)
	touch $(LOADED_DIR)/load_complete

load: $(LOADED_DIR)/loaded_dataset.parquet

# Étape 2: Feature Engineering
$(INTERIM_DIR)/feature_engineering_complete: $(LOADED_DIR)/loaded_dataset.parquet | $(INTERIM_DIR)
	$(PYTHON) src/feature_engineering.py
	touch $(INTERIM_DIR)/feature_engineering_complete

feature_engineering: $(INTERIM_DIR)/feature_engineering_complete

# Étape 3: Split et Encodage
$(INTERIM_DIR)/split_encode_complete: $(INTERIM_DIR)/feature_engineering_complete | $(INTERIM_DIR)
	$(PYTHON) src/split_and_encode.py
	touch $(INTERIM_DIR)/split_encode_complete

split_and_encode: $(INTERIM_DIR)/split_encode_complete

# Étape 4: Preprocessing
$(INTERIM_DIR)/preprocessing_complete: $(INTERIM_DIR)/split_encode_complete | $(INTERIM_DIR)
	$(PYTHON) src/preprocessing_pipeline.py
	touch $(INTERIM_DIR)/preprocessing_complete

preprocess: $(INTERIM_DIR)/preprocessing_complete

# Étape 5: Entraînement du modèle
$(MODELS_DIR)/training_complete: $(INTERIM_DIR)/preprocessing_complete | $(MODELS_DIR)
	$(PYTHON) src/train_and_log_model_with_mlflow.py
	touch $(MODELS_DIR)/training_complete

train: $(MODELS_DIR)/training_complete

# Étape 6: Vérification du drift (après l'entraînement)
$(REPORTS_DIR)/drift_check_complete: $(MODELS_DIR)/training_complete | $(REPORTS_DIR)
	$(PYTHON) src/check_drift.py
	@if [ $$? -eq 0 ]; then \
		touch $(REPORTS_DIR)/drift_check_complete; \
	else \
		echo "Pipeline arrêtée: drift significatif détecté"; \
		exit 1; \
	fi

check_drift: $(REPORTS_DIR)/drift_check_complete

# Étape 7: Évaluation du modèle
$(REPORTS_DIR)/evaluation_complete: $(MODELS_DIR)/training_complete | $(REPORTS_DIR)
	$(PYTHON) src/evaluate_model.py
	touch $(REPORTS_DIR)/evaluation_complete

evaluate: $(REPORTS_DIR)/evaluation_complete

# Nettoyer les fichiers générés
clean:
	rm -rf $(LOADED_DIR)/*
	rm -rf $(INTERIM_DIR)/*
	rm -rf $(MODELS_DIR)/*
	rm -rf $(REPORTS_DIR)/*

# Exécuter tout le pipeline
all: load feature_engineering split_and_encode preprocess train check_drift evaluate

# Aide
help:
	@echo "Pipeline ML - Commandes disponibles:"
	@echo "make load INPUT_FILE=<chemin_fichier>  : Charger les données"
	@echo "make feature_engineering              : Effectuer le feature engineering"
	@echo "make split_and_encode                 : Split et encodage des données"
	@echo "make preprocess                      : Prétraitement des données"
	@echo "make train                           : Entraîner le modèle"
	@echo "make check_drift                     : Vérifier le drift des données"
	@echo "make evaluate                        : Évaluer le modèle"
	@echo "make all INPUT_FILE=<chemin_fichier>  : Exécuter tout le pipeline"
	@echo "make clean                           : Nettoyer les fichiers générés"

.PHONY: clean all help load feature_engineering split_and_encode preprocess train check_drift evaluate