# Variables
PYTHON = python
DATA_DIR = data
LOADED_DIR = $(DATA_DIR)/loaded
INTERIM_DIR = $(DATA_DIR)/interim
MODELS_DIR = models
MLRUNS_DIR = mlruns
INPUT_FILE ?= $(DATA_DIR)/raw/transactions_immobilieres.parquet
DRIFT_CHECK_MARKER = $(MLRUNS_DIR)/.last_model_drift_check

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

# Contrôle du drift des données (avant l'entraînement)
check_data_drift: $(INTERIM_DIR)/prepared_dataset.parquet
	$(PYTHON) src/check_data_drift.py

# Étape 3: Split et Encodage
split_and_encode: $(INTERIM_DIR)/prepared_dataset.parquet check_data_drift
	$(PYTHON) src/split_and_encode.py

# Étape 4: Preprocessing
preprocess: $(INTERIM_DIR)/prepared_dataset.parquet
	$(PYTHON) src/preprocessing_pipeline.py

# Étape 5: Entraînement du modèle
train: $(INTERIM_DIR)/prepared_dataset.parquet
	$(PYTHON) src/train_and_log_model_with_mlflow.py
	@rm -f $(DRIFT_CHECK_MARKER)  # Forcer un nouveau check après l'entraînement

# Vérifier si un nouveau check de drift est nécessaire
check_drift_needed:
	@if [ ! -f $(DRIFT_CHECK_MARKER) ] || [ $$(find $(MLRUNS_DIR) -type f -name "model" -newer $(DRIFT_CHECK_MARKER) | wc -l) -gt 0 ]; then \
		echo "Un nouveau check de drift est nécessaire."; \
		exit 0; \
	else \
		echo "Le dernier check de drift est toujours valide."; \
		exit 1; \
	fi

# Contrôle du drift du modèle (après l'entraînement)
check_model_drift:
	@if $(MAKE) -s check_drift_needed; then \
		$(PYTHON) src/check_model_drift.py; \
		touch $(DRIFT_CHECK_MARKER); \
	fi

# Étape 6: Évaluation du modèle
evaluate: check_model_drift
	$(PYTHON) src/evaluate_model.py

# Nettoyer les fichiers générés
clean:
	rm -rf $(LOADED_DIR)/*
	rm -rf $(INTERIM_DIR)/*
	rm -rf $(MODELS_DIR)/*
	rm -f $(DRIFT_CHECK_MARKER)

# Exécuter tout le pipeline
all: load feature_engineering split_and_encode preprocess train evaluate

# Aide
help:
	@echo "Pipeline ML - Commandes disponibles:"
	@echo "make load INPUT_FILE=<chemin_fichier>   : Charger les données"
	@echo "make feature_engineering                : Effectuer le feature engineering"
	@echo "make check_data_drift                  : Vérifier le drift des données"
	@echo "make split_and_encode                  : Split et encodage des données"
	@echo "make preprocess                        : Prétraitement des données"
	@echo "make train                            : Entraîner le modèle"
	@echo "make check_model_drift                : Vérifier le drift du modèle"
	@echo "make evaluate                         : Évaluer le modèle"
	@echo "make all INPUT_FILE=<chemin_fichier>   : Exécuter tout le pipeline"
	@echo "make clean                            : Nettoyer les fichiers générés"

.PHONY: clean all help load feature_engineering split_and_encode preprocess train evaluate check_data_drift check_model_drift check_drift_needed