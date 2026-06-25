"""
Module de clustering pour la segmentation de clients.

Ce module implémente les algorithmes de segmentation (K-Means et hiérarchique),
la recherche du nombre optimal de clusters (méthode du coude + silhouette),
ainsi que la génération automatique de descriptions textuelles par segment.

Flux de l'application :
1. Trouver le nombre optimal de clusters K
2. Appliquer l'algorithme de clustering choisi
3. Analyser et décrire automatiquement chaque segment de clients
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Colonnes RFM utilisées pour l'analyse des segments
COLONNES_RFM = ["Recency", "Frequency", "Monetary"]

# Graine aléatoire pour la reproductibilité des résultats
RANDOM_STATE = 42


# =============================================================================
# === ALGORITHMES DE CLUSTERING ===
# =============================================================================

def apply_kmeans(data_normalized, k):
    """
    Applique l'algorithme K-Means sur les données normalisées.

    Le K-Means partitionne les clients en k clusters en minimisant
    l'inertie intra-cluster (somme des distances au centroïde).

    Parameters
    ----------
    data_normalized : numpy.ndarray
        Tableau de forme (n_samples, n_features) contenant les données RFM
        normalisées (moyenne 0, écart-type 1, Recency inversée).
    k : int
        Nombre de clusters à créer.

    Returns
    -------
    tuple : (numpy.ndarray, float)
        - labels : tableau de forme (n_samples,) avec le numéro de cluster
          assigné à chaque client (de 0 à k-1).
        - silhouette : score de silhouette moyen (entre -1 et 1, plus élevé
          = clusters mieux séparés).
    """
    # --- Initialisation et entraînement du K-Means ---
    # n_init=10 : lance l'algorithme 10 fois avec différents centroïdes initiaux
    #              et conserve le meilleur résultat (meilleure inertie)
    # max_iter=300 : nombre maximum d'itérations par exécution
    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    labels = kmeans.fit_predict(data_normalized)

    # --- Calcul du score de silhouette ---
    # Mesure la qualité du clustering : compactness intra-cluster
    # vs séparation inter-cluster (proche de 1 = bon)
    silhouette = silhouette_score(data_normalized, labels)

    return labels, silhouette


def apply_hierarchical(data_normalized, k):
    """
    Applique le clustering hiérarchique agglomératif sur les données normalisées.

    Contrairement au K-Means, cet algorithme construit une hiérarchie de
    clusters en fusionnant progressivement les plus proches (approche bottom-up).
    Le linkage 'ward' minimise la variance intra-cluster à chaque fusion.

    Parameters
    ----------
    data_normalized : numpy.ndarray
        Tableau de forme (n_samples, n_features) contenant les données RFM
        normalisées (moyenne 0, écart-type 1, Recency inversée).
    k : int
        Nombre de clusters à obtenir (coupe de l'arbre hiérarchique).

    Returns
    -------
    tuple : (numpy.ndarray, float)
        - labels : tableau de forme (n_samples,) avec le numéro de cluster
          assigné à chaque client (de 0 à k-1).
        - silhouette : score de silhouette moyen (entre -1 et 1).
    """
    # --- Initialisation et entraînement du clustering hiérarchique ---
    # linkage='ward' : minimise la variance totale intra-cluster
    #                   (recommandé pour des données normalisées)
    hierarchical = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
    )
    labels = hierarchical.fit_predict(data_normalized)

    # --- Calcul du score de silhouette ---
    silhouette = silhouette_score(data_normalized, labels)

    return labels, silhouette


# =============================================================================
# === OPTIMISATION DU K ===
# =============================================================================

def find_optimal_k(data_normalized, k_min=2, k_max=10):
    """
    Recherche le nombre optimal de clusters K en testant plusieurs valeurs.

    Pour chaque valeur de K entre k_min et k_max (inclus), calcule :
    - L'inertie (méthode du coude) : somme des distances au centroïde le plus
      proche. On cherche le "coude" où le gain d'ajouter un cluster diminue.
    - Le score de silhouette : qualité de la séparation entre clusters.
      Le K avec le score le plus élevé est généralement un bon choix.

    Parameters
    ----------
    data_normalized : numpy.ndarray
        Tableau de forme (n_samples, n_features) contenant les données RFM
        normalisées.
    k_min : int, optionnel
        Nombre minimum de clusters à tester (défaut : 2).
    k_max : int, optionnel
        Nombre maximum de clusters à tester (défaut : 10).

    Returns
    -------
    pandas.DataFrame
        DataFrame avec une ligne par valeur de K testée et les colonnes :
        - k : nombre de clusters
        - inertia : inertie (somme des distances intra-cluster)
        - silhouette_score : score de silhouette moyen
    """
    # --- Initialisation des listes de résultats ---
    resultats_k = []
    resultats_silhouette = []

    # --- Itération sur chaque valeur de K ---
    for k in range(k_min, k_max + 1):
        # Entraînement du K-Means pour cette valeur de K
        kmeans = KMeans(
            n_clusters=k,
            n_init=10,
            max_iter=300,
            random_state=RANDOM_STATE,
        )
        kmeans.fit(data_normalized)

        # Inertie : somme des carrés des distances au centroïde assigné
        resultats_k.append(kmeans.inertia_)

        # Score de silhouette (nécessite au moins 2 clusters et
        # que chaque cluster contienne au moins 2 échantillons)
        if k < data_normalized.shape[0]:
            score = silhouette_score(data_normalized, kmeans.labels_)
        else:
            score = np.nan  # Impossible de calculer si k >= n_samples
        resultats_silhouette.append(score)

    # --- Construction du DataFrame de résultats ---
    df_resultats = pd.DataFrame({
        "k": list(range(k_min, k_max + 1)),
        "inertia": resultats_k,
        "silhouette_score": np.round(resultats_silhouette, 4),
    })

    return df_resultats


# =============================================================================
# === GÉNÉRATION DES DESCRIPTIONS ===
# =============================================================================

def generate_segment_description(df_with_clusters, k):
    """
    Génère automatiquement une description textuelle pour chaque segment.

    Pour chaque cluster, calcule les moyennes des métriques RFM et les compare
    aux médianes globales pour déterminer le profil du segment :

    - Recency inversée (après normalisation, déjà gérée dans preprocessing)
    - Frequency : nombre d'achats
    - Monetary : montant total dépensé

    Profils reconnus :
    - VIP : Monetary élevé ET Frequency élevée
    - Inactifs : Recency élevée ET Frequency basse
    - Occasionnels : Monetary faible ET Frequency basse
    - Réguliers : toutes les autres combinaisons (valeurs moyennes)

    Parameters
    ----------
    df_with_clusters : pandas.DataFrame
        DataFrame contenant les colonnes RFM plus une colonne 'Cluster'
        avec le numéro de cluster assigné à chaque client.
    k : int
        Nombre de clusters (doit correspondre aux valeurs uniques dans
        la colonne 'Cluster').

    Returns
    -------
    dict
        Dictionnaire {numero_cluster: description_textuelle} avec une
        description par segment de clientèle.
    """
    # --- Calcul des médianes globales de référence ---
    # La médiane est plus robuste que la moyenne face aux valeurs extrêmes
    mediane_recency = df_with_clusters["Recency"].median()
    mediane_frequency = df_with_clusters["Frequency"].median()
    mediane_monetary = df_with_clusters["Monetary"].median()

    # --- Calcul des moyennes par cluster ---
    # Agrégation des métriques RFM pour chaque segment
    moyennes_par_cluster = df_with_clusters.groupby("Cluster")[COLONNES_RFM].mean()

    # --- Génération des descriptions par cluster ---
    descriptions = {}

    for cluster_num in range(k):
        # Extraction des moyennes RFM de ce cluster
        moy_r = moyennes_par_cluster.loc[cluster_num, "Recency"]
        moy_f = moyennes_par_cluster.loc[cluster_num, "Frequency"]
        moy_m = moyennes_par_cluster.loc[cluster_num, "Monetary"]

        # Comparaison aux médianes globales pour déterminer le profil
        recency_elevee = moy_r > mediane_recency
        recency_basse = moy_r <= mediane_recency
        frequency_elevee = moy_f > mediane_frequency
        frequency_basse = moy_f <= mediane_frequency
        monetary_eleve = moy_m > mediane_monetary
        monetary_bas = moy_m <= mediane_monetary

        # --- Attribution du profil selon les combinaisons de RFM ---

        if monetary_eleve and frequency_elevee and recency_basse:
            # Fort pouvoir d'achat, très fidèles, achats récents
            description = (
                "Clients VIP - Fort pouvoir d'achat et très fidèles. "
                "Achats fréquents et récents avec des montants élevés. "
                "Segment prioritaire à conserver."
            )

        elif recency_elevee and frequency_basse:
            # Aucun achat récent et peu d'achats au total
            description = (
                "Clients inactifs - À risque de départ, nécessitent une "
                "campagne de réactivation. Aucun achat récent et historique "
                "d'achats limité."
            )

        elif monetary_bas and frequency_basse:
            # Peu d'achats et faibles dépenses
            description = (
                "Clients occasionnels - Sensibles aux promotions. "
                "Achats rares et montants faibles. Potentiel de conversion "
                "via offres incitatives."
            )

        else:
            # Profil intermédiaire : valeurs moyennes ou mixtes
            description = (
                "Clients réguliers - Potentiel de fidélisation intéressant. "
                "Comportement d'achat modéré avec une marge de progression "
                "vers le segment VIP."
            )

        descriptions[cluster_num] = description

    return descriptions
