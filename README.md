# 🛍️ Segmentation de Clients — Application K-Means

## Description

**Segment-CLI** est une application web interactive de segmentation de clientèle, basée sur l'analyse RFM (Récence, Fréquence, Monétaire) et le clustering non supervisé K-Means. Elle permet d'importer un fichier CSV de clients, de les regrouper automatiquement en segments homogènes, et de visualiser les résultats sous forme de graphiques 3D, 2D et radar.

L'objectif est d'aider les équipes marketing à identifier les profils clients (VIP, réguliers, occasionnels, inactifs) et à prendre des décisions ciblées grâce à des fiches de segments enrichies avec des recommandations marketing automatiques.

---

## Fonctionnalités

- 📁 **Import de données** : charger un CSV personnalisé ou utiliser les 500 clients d'exemple
- 🔢 **Paramétrage de K** : choisir le nombre de segments (2 à 10) via un slider
- 🤖 **K-Means** : algorithme principal de clustering (reproductible, `random_state=42`)
- 🌳 **Clustering hiérarchique** : algorithme alternatif avec linkage Ward *(bonus)*
- 📊 **Visualisation 3D** : nuage de points interactif (Récence × Fréquence × Dépenses)
- 📉 **Visualisation 2D** : analyse croisée de deux dimensions RFM au choix
- 🕸️ **Graphique radar** : comparaison visuelle des profils RFM par segment *(bonus)*
- 📈 **Méthode du coude** : courbe d'inertie + score de silhouette pour K = 2..10 *(bonus)*
- 🃏 **Fiches segments** : descriptions automatiques + métriques + recommandations marketing
- 📥 **Export** : téléchargement du CSV enrichi (colonne Cluster) et d'un rapport texte *(bonus)*
- 🎨 **Interface Streamlit** : 5 onglets, sidebar de configuration, design responsive

---

## Technologies utilisées

| Outil | Version | Rôle |
|-------|---------|------|
| **Python** | 3.10+ | Langage principal |
| **Streamlit** | 1.32.0 | Interface web interactive |
| **Pandas** | 2.2.1 | Manipulation et analyse des données |
| **NumPy** | 1.26.4 | Calculs numériques |
| **Scikit-learn** | 1.4.1 | K-Means, clustering hiérarchique, StandardScaler, silhouette |
| **Plotly** | 5.20.0 | Graphiques interactifs (3D, 2D, radar, coude) |
| **SciPy** | 1.12.0 | Outils scientifiques complémentaires |

---

## Source des données

Les données sont **simulées de façon réaliste** à des fins de démonstration. Elles représentent **500 clients** d'une boutique en ligne avec un comportement d'achat structuré en 4 groupes naturels :

| Groupe | Effectif | Récence | Fréquence | Dépenses |
|--------|----------|---------|-----------|----------|
| 💎 VIP | 100 | 1–30 jours | 40–100 achats | 2 000–5 000 € |
| 👤 Réguliers | 150 | 15–90 jours | 10–40 achats | 500–2 000 € |
| 🛒 Occasionnels | 150 | 30–180 jours | 1–10 achats | 10–500 € |
| 😴 Inactifs | 100 | 200–365 jours | 1–5 achats | 10–200 € |

**Structure RFM** : chaque client est décrit par trois métriques clé :
- **Recency** : nombre de jours depuis le dernier achat (plus c'est bas, mieux c'est)
- **Frequency** : nombre total d'achats sur la période
- **Monetary** : montant total dépensé en euros

Le fichier se trouve dans `data/sample_data.csv` et peut être régénéré avec le script `generate_data.py`.

---

## Installation

### Prérequis

- **Python 3.10** ou supérieur
- **pip** (gestionnaire de paquets Python)

### Étapes

1. **Cloner le repository**
   ```bash
   git clone https://github.com/du-dev/segment-cli.git
   cd segment-cli
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Lancer l'application**
   ```bash
   streamlit run app.py
   ```

4. **Ouvrir dans le navigateur**
   ```
   http://localhost:8501
   ```

---

## Utilisation

1. **Importer les données** — Dans la sidebar, cochez *"Utiliser les données exemple"* pour charger les 500 clients simulés, ou uploadez votre propre fichier CSV.

2. **Configurer le clustering** — Réglez le nombre de segments **K** avec le slider (entre 2 et 10, défaut : 4) et choisissez l'algorithme (**K-Means** ou **Hiérarchique Ward**).

3. **Lancer la segmentation** — Cliquez sur 🚀 *"Lancer la segmentation"*. L'application normalise les données RFM, applique le clustering et affiche les résultats.

4. **Explorer les onglets** :
   - **📊 Vue d'ensemble** : métriques globales, graphique 3D, tableau récapitulatif
   - **🔍 Analyse détaillée** : scatter 2D au choix + radar comparatif
   - **📋 Fiches Segments** : description, métriques et recommandation marketing par segment
   - **📈 Optimisation K** : courbe du coude et recommandation du K optimal
   - **📥 Export** : téléchargement du CSV enrichi et du rapport texte

5. **Ajuster** — Modifiez K ou l'algorithme et relancez pour comparer les résultats.

---

## Structure du projet

```
segment-cli/
├── app.py                 # Application Streamlit principale (interface, onglets, sidebar)
├── preprocessing.py       # Module de prétraitement (chargement CSV, normalisation RFM)
├── clustering.py           # Module de clustering (K-Means, hiérarchique, descriptions auto)
├── visualization.py        # Module de visualisation (graphiques 3D, 2D, coude, radar)
├── generate_data.py       # Script de génération des données simulées (reproductible)
├── requirements.txt        # Dépendances Python avec versions exactes
├── README.md               # Documentation du projet
├── .gitignore              # Fichiers et dossiers ignorés par Git
└── data/
    └── sample_data.csv     # Jeu de données simulées (500 clients RFM)
```

| Fichier | Description |
|---------|-------------|
| `app.py` | Point d'entrée de l'application. Contient l'interface Streamlit avec la sidebar, la gestion des sessions, les 5 onglets et l'export des résultats. |
| `preprocessing.py` | Chargement et validation des données CSV, normalisation StandardScaler avec inversion de Recency, statistiques descriptives RFM. |
| `clustering.py` | Algorithmes K-Means et hiérarchique Ward, recherche du K optimal (coude + silhouette), génération automatique des descriptions de segments. |
| `visualization.py` | Graphiques Plotly interactifs : scatter 3D, scatter 2D paramétrable, courbe du coude à double axe Y, radar comparatif des profils RFM. |
| `generate_data.py` | Script autonome pour régénérer `sample_data.csv` avec `np.random.seed(42)` pour la reproductibilité. |
| `requirements.txt` | Liste des 6 dépendances avec versions exactes. |
| `data/sample_data.csv` | 500 clients simulés avec les colonnes CustomerID, Recency, Frequency, Monetary, LastPurchaseDate. |

---

## Format du CSV attendu

Votre fichier CSV doit contenir au minimum les **3 colonnes RFM obligatoires**. Les colonnes `CustomerID` et `LastPurchaseDate` sont optionnelles.

```csv
CustomerID,Recency,Frequency,Monetary,LastPurchaseDate
C001,15,42,1850,2024-11-20
C002,203,2,85,2024-05-12
C003,8,67,3200,2024-12-18
C004,145,5,420,2024-08-05
C005,62,22,1100,2024-10-30
```

| Colonne | Obligatoire | Type | Description |
|---------|:-----------:|------|-------------|
| `CustomerID` | ❌ | Texte | Identifiant unique du client |
| `Recency` | ✅ | Entier | Nombre de jours depuis le dernier achat (1–365) |
| `Frequency` | ✅ | Entier | Nombre total d'achats (1–100) |
| `Monetary` | ✅ | Entier/Flottant | Montant total dépensé en euros (10–5000) |
| `LastPurchaseDate` | ❌ | Date | Date du dernier achat (format YYYY-MM-DD) |

---

## Bonus implémentés

- 🌳 **Clustering hiérarchique (Ward)** : algorithme alternatif au K-Means, accessible via un selectbox dans la sidebar
- 🕸️ **Graphique radar** : visualisation en toile d'araignée comparant les profils RFM normalisés de chaque segment
- 📈 **Analyse du K optimal** : courbe du coude (inertie) + score de silhouette sur deux axes Y, avec recommandation automatique du meilleur K
- 📥 **Export double** : téléchargement du CSV enrichi (avec colonne Cluster) et d'un rapport texte descriptif des segments
- 📋 **Fiches segments automatiques** : descriptions et recommandations marketing générées à partir des moyennes RFM comparées aux médianes globales

---

## Auteur

**Groupe 13** — Master IA — 2026

Repository : [https://github.com/du-dev/segment-cli](https://github.com/du-dev/segment-cli)
