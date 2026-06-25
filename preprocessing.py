"""
Module de prétraitement des données pour la segmentation de clients.

Ce module gère le chargement, la validation et la normalisation des données RFM
(Recency, Frequency, Monetary) nécessaires à l'algorithme K-Means.

Flux de l'application :
1. Charger et valider les données brutes (CSV)
2. Normaliser les variables RFM pour le clustering
3. Calculer les statistiques descriptives pour l'analyse
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Colonnes RFM obligatoires dans le jeu de données
COLONNES_RFM = ["Recency", "Frequency", "Monetary"]

# Colonnes optionnelles reconnues par le module
COLONNE_ID_CLIENT = "CustomerID"


# =============================================================================
# === CHARGEMENT ET VALIDATION ===
# =============================================================================

def load_and_validate_csv(file):
    """
    Charge un fichier CSV et valide la présence des colonnes RFM obligatoires.

    Accepte indifféremment un fichier uploadé via Streamlit (objet UploadedFile)
    ou un chemin d'accès vers un fichier CSV sur disque.

    Nettoie les données en supprimant les lignes avec des valeurs manquantes
    et les éventuels doublons d'identifiants clients.

    Parameters
    ----------
    file : str ou streamlit.UploadedFile
        Chemin vers le fichier CSV (ex: "data/sample_data.csv")
        ou objet fichier uploadé via le widget st.file_uploader de Streamlit.

    Returns
    -------
    pandas.DataFrame
        DataFrame nettoyé contenant uniquement les lignes valides,
        sans valeurs manquantes ni doublons sur CustomerID.

    Raises
    ------
    ValueError
        Si le fichier ne contient pas l'une des colonnes RFM obligatoires
        (Recency, Frequency, Monetary).
    """
    # --- Lecture du fichier CSV ---
    # Streamlit renvoie un objet UploadedFile avec une méthode .read()
    # Un chemin string est lu directement par pandas
    try:
        if hasattr(file, "read"):
            # Fichier uploadé via Streamlit (UploadedFile)
            df = pd.read_csv(file, comment="#")
        else:
            # Chemin de fichier classique (string)
            df = pd.read_csv(file, comment="#")
    except Exception as e:
        raise ValueError(f"Impossible de lire le fichier : {e}")

    # --- Vérification de la présence des colonnes RFM obligatoires ---
    colonnes_manquantes = [col for col in COLONNES_RFM if col not in df.columns]
    if colonnes_manquantes:
        raise ValueError(
            f"Colonnes manquantes dans le fichier CSV : {colonnes_manquantes}. "
            f"Colonnes requises : {COLONNES_RFM}"
        )

    # --- Suppression des lignes avec des valeurs manquantes ---
    # Toute ligne contenant un NaN dans n'importe quelle colonne est retirée
    lignes_avant = len(df)
    df = df.dropna()
    lignes_supprimees = lignes_avant - len(df)
    if lignes_supprimees > 0:
        print(f"[INFO] {lignes_supprimees} ligne(s) supprimée(s) (valeurs manquantes)")

    # --- Suppression des doublons sur CustomerID (si la colonne existe) ---
    # On conserve la première occurrence de chaque identifiant client
    if COLONNE_ID_CLIENT in df.columns:
        doublons_avant = len(df)
        df = df.drop_duplicates(subset=COLONNE_ID_CLIENT, keep="first")
        doublons_supprimes = doublons_avant - len(df)
        if doublons_supprimes > 0:
            print(f"[INFO] {doublons_supprimes} doublon(s) supprimé(s) sur {COLONNE_ID_CLIENT}")

    return df


# =============================================================================
# === NORMALISATION ===
# =============================================================================

def normalize_rfm(df):
    """
    Normalise les colonnes RFM avec un StandardScaler pour le clustering.

    Extrait les colonnes Recency, Frequency et Monetary, les normalise
    (moyenne = 0, écart-type = 1) puis retourne le tableau normalisé
    ainsi que le DataFrame original conservant toutes les colonnes.

    Particularité importante pour Recency :
    Une valeur basse de Recency signifie un client récent, ce qui est positif.
    Pour que cette logique soit cohérente avec le clustering (plus la valeur
    est élevée, mieux c'est), on inverse le signe de Recency normalisé.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame nettoyé contenant au minimum les colonnes RFM.

    Returns
    -------
    tuple : (numpy.ndarray, pandas.DataFrame)
        - Tableau numpy de forme (n_samples, 3) contenant les valeurs RFM
          normalisées. Recency est multipliée par -1 pour l'inversion.
        - DataFrame original inchangé (avec toutes ses colonnes).
    """
    # --- Extraction des colonnes RFM ---
    rfm_data = df[COLONNES_RFM].copy()

    # --- Normalisation StandardScaler ---
    # Centre les données (moyenne = 0) et réduit (écart-type = 1)
    # Cela évite qu'une variable à grande échelle (Monetary)
    # ne domine les autres dans le calcul des distances du K-Means
    scaler = StandardScaler()
    rfm_normalise = scaler.fit_transform(rfm_data)

    # --- Inversion de Recency ---
    # Recency = nombre de jours depuis le dernier achat
    # Plus ce nombre est petit, plus le client est récent (donc précieux)
    # On multiplie par -1 pour que "récent" = valeur élevée et positive,
    # cohérent avec Frequency et Monetary où "plus élevé = meilleur client"
    rfm_normalise[:, 0] *= -1

    return rfm_normalise, df


# =============================================================================
# === STATISTIQUES ===
# =============================================================================

def get_rfm_stats(df):
    """
    Calcule les statistiques descriptives pour chaque colonne RFM.

    Pour chaque métrique (Recency, Frequency, Monetary), calcule les
    indicateurs suivants : minimum, maximum, moyenne et médiane.
    Retourne un DataFrame formaté et lisible pour l'affichage.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame nettoyé contenant les colonnes RFM.

    Returns
    -------
    pandas.DataFrame
        DataFrame résumé avec les statistiques descriptives.
        Index : noms des indicateurs (Min, Max, Moyenne, Médiane).
        Colonnes : Recency, Frequency, Monetary.
    """
    # --- Extraction des colonnes RFM ---
    rfm_data = df[COLONNES_RFM]

    # --- Calcul des statistiques descriptives ---
    stats = pd.DataFrame({
        "Min": rfm_data.min(),
        "Max": rfm_data.max(),
        "Moyenne": rfm_data.mean().round(2),
        "Médiane": rfm_data.median(),
    })

    # Transposition : les colonnes RFM deviennent les colonnes du résumé
    # et les indicateurs statistiques deviennent les lignes (index)
    stats = stats.T

    # Renommage de l'index pour un affichage plus lisible
    stats.index.name = "Indicateur"

    return stats
