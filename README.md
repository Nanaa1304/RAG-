# 🔎 Assistant RAG

Un système de question-réponse basé sur vos propres documents (RAG — Retrieval-Augmented Generation), avec une interface de chat Streamlit.

Le projet lit des fichiers texte, les découpe en chunks, les transforme en embeddings stockés dans une base vectorielle ChromaDB, puis répond aux questions en récupérant les passages pertinents et en les envoyant à un LLM pour générer une réponse en langage naturel.

## Fonctionnement

1. **Ingestion** (`pipline_ingestion.py`) — charge les fichiers `.txt` du dossier `docs/`, les découpe en chunks, calcule leurs embeddings et les stocke dans ChromaDB (`db/chroma_db`).
2. **Retrieval** (`retrieval_pipline.py`) — script de test en ligne de commande : recherche les chunks les plus pertinents pour une question et génère une réponse.
3. **Interface** (`app.py`) — interface de chat Streamlit qui utilise la même logique, avec historique de conversation et affichage des sources.

## Prérequis

- Python 3.11+
- Une clé API [NVIDIA NIM](https://build.nvidia.com/) (gratuite, pour l'accès au LLM)

## Installation

```bash
# Cloner le repo
git clone https://github.com/VOTRE_USERNAME/VOTRE_REPO.git
cd VOTRE_REPO

# Créer et activer un environnement virtuel
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows PowerShell
# source venv/bin/activate       # macOS / Linux

# Installer les dépendances
pip install -r requirements.txt
```

## Configuration

Créez un fichier `.env` à la racine du projet avec votre clé API :

```
NVIDIA_API_KEY=nvapi-votre_cle_ici
```

⚠️ Ce fichier ne doit **jamais** être commité (il est déjà exclu via `.gitignore`).

## Ajouter vos documents

Placez vos fichiers `.txt` (encodage UTF-8) dans le dossier `docs/`.

## Utilisation

### 1. Lancer l'ingestion (une fois, ou à chaque ajout de nouveaux documents)

```bash
python pipline_ingestion.py
```

Cela crée/actualise la base vectorielle dans `db/chroma_db`.

### 2. Lancer l'interface

```bash
python -m streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur sur `http://localhost:8501`.

### 3. (Optionnel) Tester le retrieval en ligne de commande

```bash
python retrieval_pipline.py
```

Utile pour déboguer rapidement sans passer par l'interface.

## Structure du projet

```
rag/
├── docs/                   # Vos documents source (.txt)
├── db/chroma_db/           # Base vectorielle générée (ignorée par git)
├── venv/                   # Environnement virtuel (ignoré par git)
├── .env                    # Clé API (ignoré par git, à créer soi-même)
├── .gitignore
├── pipline_ingestion.py    # Script d'ingestion des documents
├── retrieval_pipline.py    # Script de test retrieval + génération (CLI)
├── app.py                  # Interface Streamlit
├── requirements.txt
└── README.md
```

## Technologies utilisées

- [LangChain](https://python.langchain.com/) — orchestration du pipeline RAG
- [ChromaDB](https://www.trychroma.com/) — base de données vectorielle
- [HuggingFace Sentence Transformers](https://www.sbert.net/) (`paraphrase-multilingual-MiniLM-L12-v2`) — génération des embeddings
- [NVIDIA NIM](https://build.nvidia.com/) — accès au LLM (API compatible OpenAI)
- [Streamlit](https://streamlit.io/) — interface web

## Limitations connues

- Le retrieval fonctionne bien sur des questions factuelles simples ; les questions complexes ou multi-documents peuvent nécessiter d'ajuster `chunk_size`, `chunk_overlap` ou le nombre de chunks récupérés (`k`).
- Le modèle d'embeddings est optimisé pour le multilingue mais reste plus performant sur des textes courts et factuels que sur des raisonnements complexes.
- Aucune gestion de mise à jour incrémentale de la base vectorielle : relancer l'ingestion recrée les embeddings pour tous les documents.
