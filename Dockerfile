# Utiliser une image Python officielle comme base
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Installer uv
RUN pip install uv

# Copier les fichiers de dépendances
COPY pyproject.toml requirements.txt* ./

# Installer les dépendances Python via uv
RUN uv venv /venv && \
    . /venv/bin/activate && \
    uv pip install -r pyproject.toml

    # Copier tous les fichiers du projet
COPY . .

# Exposer le port Streamlit
EXPOSE 8501

# Définir les variables d'environnement
ENV PATH="/venv/bin:$PATH"

# Commande pour lancer l'application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
