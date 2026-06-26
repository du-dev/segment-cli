"""
Application Streamlit — Segmentation de clients via K-Means.

Interface web interactive permettant de :
1. Importer un jeu de données clients (CSV)
2. Configurer et lancer un clustering (K-Means ou hiérarchique)
3. Visualiser les segments en 3D, 2D et radar
4. Analyser les fiches détaillées de chaque segment
5. Trouver le nombre optimal de clusters K
6. Exporter les résultats

Modules utilisés : preprocessing, clustering, visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

# =============================================================================
# Imports des modules du projet
# =============================================================================
from preprocessing import load_and_validate_csv, normalize_rfm, get_rfm_stats
from clustering import (
    apply_kmeans,
    apply_hierarchical,
    find_optimal_k,
    generate_segment_description,
)
from visualization import (
    plot_clusters_3d,
    plot_clusters_2d,
    plot_elbow_curve,
    plot_segment_radar,
    CLUSTER_COLORS,
)

# =============================================================================
# Constantes de l'application
# =============================================================================
COLONNES_RFM = ["Recency", "Frequency", "Monetary"]
CHEMIN_DONNEES_EXEMPLE = "data/sample_data.csv"

# =============================================================================
# === CONFIGURATION PAGE ===
# =============================================================================
# Titre, icône et layout large pour une meilleure expérience visuelle
st.set_page_config(
    page_title="Segmentation de Clients — K-Means",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Suppression du padding par défaut pour un rendu plus immersif ---
st.markdown(
    """
    <style>
        /* Réduction des marges autour du contenu principal */
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }

        /* Cartes de segments avec ombre légère et texte lisible */
        .segment-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border-left: 5px solid;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: transform 0.2s;
            color: #212529 !important;
        }
        .segment-card p,
        .segment-card strong,
        .segment-card span {
            color: #212529 !important;
        }
        .segment-card h3 {
            color: #212529 !important;
            margin-top: 0 !important;
        }
        .segment-card:hover { transform: translateY(-2px); }

        /* Métriques stylisées en haut de page */
        [data-testid="stMetricValue"] { font-size: 1.6rem; }

        /* Sidebar avec fond subtil */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3a5f 0%, #0f1c2e 100%);
        }
        section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

        /* Séparateurs plus fins */
        hr { height: 1px; border-color: rgba(128,128,128,0.2); }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# === SIDEBAR ===
# =============================================================================
with st.sidebar:

    # --- Logo et description du projet ---
    st.markdown(
        """
        # 🛍️ Segment-CLI
        **Segmentation de clients** par analyse RFM
        et clustering K-Means
        """
    )
    st.markdown("---")

    # --- Titre des paramètres ---
    st.markdown("## ⚙️ Paramètres")

    # --- Section import des données ---
    st.markdown("### 📁 Import des données")

    # Option : utiliser les données d'exemple ou uploader un fichier
    utiliser_exemple = st.checkbox(
        "Utiliser les données exemple (500 clients)",
        value=True,
        key="use_sample",
    )

    fichier_upload = None
    if not utiliser_exemple:
        fichier_upload = st.file_uploader(
            "Charger un fichier CSV",
            type=["csv"],
            help="Format requis : colonnes Recency, Frequency, Monetary",
        )

    st.markdown("---")

    # --- Section nombre de clusters ---
    st.markdown("### 🔢 Nombre de clusters")

    k = st.slider(
        "Sélectionner K",
        min_value=2,
        max_value=10,
        value=4,
        step=1,
        help="Nombre de segments à créer (2 à 10)",
    )

    # --- Section algorithme (BONUS) ---
    st.markdown("### 🤖 Algorithme")

    choix_algo = st.selectbox(
        "Méthode de clustering",
        options=["K-Means (recommandé)", "Clustering Hiérarchique (Ward)"],
        index=0,
        help="K-Means : rapide et efficace. Hiérarchique : structure arborescente.",
    )

    st.markdown("---")

    # --- Bouton principal de segmentation ---
    lancer = st.button("🚀 Lancer la segmentation", type="primary", use_container_width=True)

    st.markdown("---")

    # --- Section optimisation du K ---
    st.markdown("### 📊 Trouver le K optimal")
    lancer_coude = st.button("🔍 Analyser K = 2..10", use_container_width=True)



# =============================================================================
# === INITIALISATION DU SESSION STATE ===
# =============================================================================
# Le session_state conserve les résultats entre les interactions
# sans relancer le clustering à chaque re-rendu de la page

if "df" not in st.session_state:
    st.session_state.df = None
if "rfm_norm" not in st.session_state:
    st.session_state.rfm_norm = None
if "labels" not in st.session_state:
    st.session_state.labels = None
if "silhouette" not in st.session_state:
    st.session_state.silhouette = None
if "algo_nom" not in st.session_state:
    st.session_state.algo_nom = None
if "df_coude" not in st.session_state:
    st.session_state.df_coude = None

# =============================================================================
# === CHARGEMENT DES DONNÉES ===
# =============================================================================

def charger_donnees():
    """
    Charge les données depuis le fichier source (exemple ou upload).

    Met à jour le session_state avec le DataFrame nettoyé et les données
    RFM normalisées. Affiche un message d'erreur en cas de problème.

    Returns
    -------
    bool
        True si le chargement a réussi, False sinon.
    """
    try:
        # Détermination de la source : fichier exemple ou upload utilisateur
        if st.session_state.get("use_sample", True):
            source = CHEMIN_DONNEES_EXEMPLE
        else:
            source = fichier_upload
            if source is None:
                st.warning("⚠️ Veuillez charger un fichier CSV ou cocher l'option des données d'exemple.")
                return False

        # Chargement et validation via le module preprocessing
        df = load_and_validate_csv(source)

        # Vérification : K ne doit pas dépasser le nombre de clients
        if len(df) <= k:
            st.error(
                f"❌ Impossible de créer {k} clusters avec seulement "
                f"{len(df)} clients. Réduisez K ou utilisez plus de données."
            )
            return False

        # Normalisation RFM pour le clustering
        rfm_norm, _ = normalize_rfm(df)

        # Stockage dans le session_state
        st.session_state.df = df
        st.session_state.rfm_norm = rfm_norm
        st.session_state.labels = None  # Réinitialisation du clustering
        st.session_state.silhouette = None

        st.success(f"✅ {len(df)} clients chargés avec succès.")
        return True

    except ValueError as e:
        st.error(f"❌ Erreur de chargement : {e}")
        return False
    except Exception as e:
        st.error(f"❌ Erreur inattendue : {e}")
        return False


# =============================================================================
# === LANCEMENT DU CLUSTERING ===
# =============================================================================

def lancer_clustering():
    """
    Applique l'algorithme de clustering sélectionné et stocke les résultats.

    Met à jour le session_state avec les labels de cluster et le score de
    silhouette.
    """
    if st.session_state.df is None:
        st.error("❌ Aucune donnée chargée. Importez un fichier d'abord.")
        return

    rfm_norm = st.session_state.rfm_norm

    # Sélection de l'algorithme
    if choix_algo == "K-Means (recommandé)":
        labels, silhouette = apply_kmeans(rfm_norm, k)
        st.session_state.algo_nom = "K-Means"
    else:
        labels, silhouette = apply_hierarchical(rfm_norm, k)
        st.session_state.algo_nom = "Hiérarchique (Ward)"

    # Ajout des labels au DataFrame
    st.session_state.df["Cluster"] = labels
    st.session_state.labels = labels
    st.session_state.silhouette = silhouette


def lancer_analyse_coude():
    """
    Lance l'analyse du coude pour toutes les valeurs de K de 2 à 10
    et stocke les résultats dans le session_state.
    """
    if st.session_state.rfm_norm is None:
        st.error("❌ Aucune donnée chargée. Importez un fichier d'abord.")
        return

    df_coude = find_optimal_k(st.session_state.rfm_norm, k_min=2, k_max=10)
    st.session_state.df_coude = df_coude


# =============================================================================
# === GESTION DES ÉVÉNEMENTS (BOUTONS) ===
# =============================================================================
# Streamlit détecte les clics sur les boutons à chaque re-rendu.
# On vérifie l'état des boutons pour déclencher les actions.

if lancer:
    if charger_donnees():
        lancer_clustering()

if lancer_coude:
    if st.session_state.rfm_norm is None:
        charger_donnees()
    if st.session_state.rfm_norm is not None:
        lancer_analyse_coude()

# =============================================================================
# === TITRE DE LA PAGE ===
# =============================================================================
st.markdown("# 🛍️ Segmentation de Clients — K-Means")
st.markdown(
    "Analysez votre base clients en segments homogènes grâce au clustering non supervisé."
)
st.markdown("---")

# =============================================================================
# === PAGE D'ACCUEIL (si aucune donnée) ===
# =============================================================================
if st.session_state.df is None:

    # Message d'accueil avec instructions
    st.info("👋 **Bienvenue !** Chargez vos données et lancez une segmentation.")

    st.markdown(
        """
        ### 📖 Comment utiliser cette application ?

        1. **Sidebar → Import** : Cochez *« Utiliser les données exemple »*
           ou uploadez votre propre fichier CSV
        2. **Sidebar → Paramètres** : Choisissez le nombre de clusters **K**
           et l'algorithme de clustering
        3. **Sidebar → Lancer** : Cliquez sur *« 🚀 Lancer la segmentation »*

        ### 📋 Format CSV attendu

        Votre fichier doit contenir au minimum ces colonnes :

        | Colonne | Description | Exemple |
        |---------|-------------|---------|
        | `Recency` | Jours depuis le dernier achat | 15 |
        | `Frequency` | Nombre total d'achats | 42 |
        | `Monetary` | Montant total dépensé (€) | 1850 |

        > Les colonnes `CustomerID` et `LastPurchaseDate` sont optionnelles.
        """
    )

    # Arrêt du rendu : pas la peine d'afficher les onglets sans données
    st.stop()

# =============================================================================
# === AFFICHAGE DES ONGLETS ===
# =============================================================================
# Si les données sont chargées, on affiche l'interface complète

tab_vue, tab_analyse, tab_fiches, tab_optim, tab_export = st.tabs([
    "📊 Vue d'ensemble",
    "🔍 Analyse détaillée",
    "📋 Fiches Segments",
    "📈 Optimisation K",
    "📥 Export",
])

# -----------------------------------------------------------------------------
# ONGLET 1 — Vue d'ensemble
# -----------------------------------------------------------------------------
with tab_vue:

    # Vérifie si un clustering a été lancé
    if st.session_state.labels is None:
        st.warning(
            "⏳ Aucun clustering effectué. "
            "Utilisez la sidebar pour lancer la segmentation."
        )
        st.stop()

    # --- Métriques en haut (4 colonnes) ---
    nb_clients = len(st.session_state.df)
    nb_clusters = st.session_state.df["Cluster"].nunique()
    score_sil = st.session_state.silhouette
    algo = st.session_state.algo_nom

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.metric(label="👥 Clients", value=f"{nb_clients}")
    with col_m2:
        st.metric(label="🎯 Segments", value=f"{nb_clusters}")
    with col_m3:
        st.metric(
            label="📐 Silhouette",
            value=f"{score_sil:.4f}",
            delta="Bon" if score_sil > 0.5 else "Moyen" if score_sil > 0.3 else "Faible",
        )
    with col_m4:
        st.metric(label="🤖 Algorithme", value=algo)

    st.markdown("---")

    # --- Graphique 3D interactif ---
    st.subheader("🌐 Visualisation 3D des segments")
    fig_3d = plot_clusters_3d(st.session_state.df, nb_clusters)
    st.plotly_chart(fig_3d, use_container_width=True)

    # --- Tableau récapitulatif des segments ---
    st.subheader("📋 Récapitulatif des segments")

    # Calcul de la taille et des moyennes par cluster
    recap = st.session_state.df.groupby("Cluster").agg(
        Taille=("CustomerID" if "CustomerID" in st.session_state.df.columns else "Recency", "count"),
        Récence_moy=("Recency", "mean"),
        Fréquence_moy=("Frequency", "mean"),
        Dépenses_moy=("Monetary", "mean"),
    ).round(1)

    # Ajout du pourcentage de chaque segment
    recap["Pourcentage (%)"] = (recap["Taille"] / recap["Taille"].sum() * 100).round(1)

    # Coloration des colonnes pour un rendu visuel
    st.dataframe(
        recap.style.format("{:.1f}").background_gradient(
            subset=["Taille", "Pourcentage (%)"],
            cmap="Blues",
        ),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------------------------------
# ONGLET 2 — Analyse détaillée
# -----------------------------------------------------------------------------
with tab_analyse:

    if st.session_state.labels is None:
        st.warning("⏳ Lancez une segmentation d'abord.")
        st.stop()

    # --- Choix des axes pour le graphique 2D ---
    st.subheader("📉 Analyse croisée des dimensions RFM")

    col_x, col_y = st.columns(2)
    with col_x:
        axe_x = st.selectbox(
            "Axe X",
            options=COLONNES_RFM,
            index=1,  # Frequency par défaut
            format_func=lambda v: v,
        )
    with col_y:
        axe_y = st.selectbox(
            "Axe Y",
            options=COLONNES_RFM,
            index=2,  # Monetary par défaut
            format_func=lambda v: v,
        )

    # --- Graphique 2D ---
    fig_2d = plot_clusters_2d(st.session_state.df, axe_x, axe_y)
    st.plotly_chart(fig_2d, use_container_width=True)

    st.markdown("---")

    # --- Graphique radar comparatif ---
    st.subheader("🕸️ Profil radar des segments")
    nb_clusters = st.session_state.df["Cluster"].nunique()
    fig_radar = plot_segment_radar(st.session_state.df, nb_clusters)
    st.plotly_chart(fig_radar, use_container_width=True)


# -----------------------------------------------------------------------------
# ONGLET 3 — Fiches Segments
# -----------------------------------------------------------------------------
with tab_fiches:

    if st.session_state.labels is None:
        st.warning("⏳ Lancez une segmentation d'abord.")
        st.stop()

    st.subheader("🃏 Fiches détaillées de chaque segment")
    st.markdown(
        "Chaque segment est décrit automatiquement en fonction de son profil RFM."
    )

    # --- Génération des descriptions automatiques ---
    nb_clusters = st.session_state.df["Cluster"].nunique()
    descriptions = generate_segment_description(st.session_state.df, nb_clusters)

    # --- Affichage des fiches (2 par ligne via st.columns) ---
    # Calcul des moyennes par cluster pour les métriques des fiches
    moyennes = st.session_state.df.groupby("Cluster")[COLONNES_RFM].mean().round(1)
    tailles = st.session_state.df["Cluster"].value_counts().sort_index()

    # Parcours de chaque cluster pour créer une fiche
    for cluster_num in range(nb_clusters):

        # Détermination de la couleur pour la bordure gauche de la carte
        couleur = CLUSTER_COLORS[cluster_num % len(CLUSTER_COLORS)]

        # Extraction des métriques de ce segment
        taille = tailles.get(cluster_num, 0)
        moy_r = moyennes.loc[cluster_num, "Recency"]
        moy_f = moyennes.loc[cluster_num, "Frequency"]
        moy_m = moyennes.loc[cluster_num, "Monetary"]
        description = descriptions.get(cluster_num, "Segment non décrit.")

        # --- Extraction d'un nom court à partir de la description ---
        # Le nom court est la première partie de la description
        # avant le premier tiret
        nom_court = description.split("—")[0].split("-")[0].split("–")[0].strip()

        # --- Génération de la recommandation marketing ---
        if "VIP" in description or "VIP" in nom_court:
            reco = (
                "🎁 **Recommandation** : Programme de fidélité exclusif, "
                "offres VIP, accès anticipé aux nouveautés."
            )
        elif "inactif" in description.lower():
            reco = (
                "📩 **Recommandation** : Campagne de réactivation par email, "
                "code promo de bienvenue, enquête de satisfaction."
            )
        elif "occasionnel" in description.lower():
            reco = (
                "🏷️ **Recommandation** : Promotions flash, ventes privées, "
                "parrainage avec récompense."
            )
        else:
            reco = (
                "📈 **Recommandation** : Personnalisation des suggestions, "
                "fréquence de contact adaptée, montée en gamme progressive."
            )

        # --- Affichage de la fiche dans une carte stylisée ---
        st.markdown(
            f"""
            <div class="segment-card" style="border-left-color: {couleur};">
                <h3 style="color: {couleur}; margin-top: 0;">
                    Segment {cluster_num} — {nom_court}
                </h3>
                <p style="color: #212529;">
                    <strong style="color: #212529;">Taille :</strong> {taille} clients
                </p>
                <p style="color: #212529;">
                    <strong style="color: #212529;">Récence moy. :</strong> {moy_r:.0f} jours
                    &nbsp;|&nbsp;
                    <strong style="color: #212529;">Fréquence moy. :</strong> {moy_f:.0f} achats
                    &nbsp;|&nbsp;
                    <strong style="color: #212529;">Dépenses moy. :</strong> {moy_m:.0f} €
                </p>
                <p style="color: #343a40;">{description}</p>
                <p style="color: #343a40;">{reco}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# ONGLET 4 — Optimisation K
# -----------------------------------------------------------------------------
with tab_optim:

    st.subheader("📈 Recherche du nombre optimal de clusters")

    # --- Vérifie si l'analyse du coude a été lancée ---
    if st.session_state.df_coude is None:
        st.info(
            "💡 Cliquez sur **« 🔍 Analyser K = 2..10 »** dans la sidebar "
            "pour lancer l'analyse."
        )
        st.stop()

    # --- Affichage de la courbe du coude ---
    fig_coude = plot_elbow_curve(st.session_state.df_coude)
    st.plotly_chart(fig_coude, use_container_width=True)

    # --- Tableau détaillé des scores ---
    st.subheader("📋 Scores pour chaque valeur de K")
    st.dataframe(
        st.session_state.df_coude.style.format({"inertia": "{:.1f}"}).background_gradient(
            subset=["silhouette_score"],
            cmap="RdYlGn",
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --- Recommandation automatique du meilleur K ---
    df_coude = st.session_state.df_coude

    # Méthode 1 : K avec le meilleur score de silhouette
    idx_meilleur_sil = df_coude["silhouette_score"].idxmax()
    k_meilleur_sil = int(df_coude.loc[idx_meilleur_sil, "k"])
    score_meilleur_sil = df_coude.loc[idx_meilleur_sil, "silhouette_score"]

    # Méthode 2 : K du coude (plus grande baisse relative d'inertie)
    df_coude_copy = df_coude.copy()
    df_coude_copy["delta_inertia"] = df_coude_copy["inertia"].diff().abs()
    df_coude_copy["delta_pct"] = df_coude_copy["delta_inertia"] / df_coude_copy["inertia"].shift(1)
    idx_coude = df_coude_copy["delta_pct"].idxmax()
    k_coude = int(df_coude.loc[idx_coude, "k"])

    st.markdown("---")
    st.markdown(
        f"""
        ### 🎯 Recommandation

        | Méthode | K recommandé | Score |
        |---------|:------------:|-------|
        | Score de silhouette | **{k_meilleur_sil}** | {score_meilleur_sil:.4f} |
        | Méthode du coude | **{k_coude}** | inertie ≈ {df_coude.loc[idx_coude, 'inertia']:.1f} |

        > 💡 **Conseil** : Un K entre {k_coude} et {k_meilleur_sil} est un bon compromis
        > entre qualité des segments et interprétabilité.
        """
    )


# -----------------------------------------------------------------------------
# ONGLET 5 — Export
# -----------------------------------------------------------------------------
with tab_export:

    if st.session_state.labels is None:
        st.warning("⏳ Lancez une segmentation d'abord pour pouvoir exporter.")
        st.stop()

    st.subheader("📥 Télécharger les résultats")

    col_exp1, col_exp2 = st.columns(2)

    # --- Export CSV enrichi avec la colonne Cluster ---
    with col_exp1:
        st.markdown("#### 📄 Données enrichies (CSV)")

        # Préparation du CSV en mémoire
        csv_buffer = io.StringIO()
        st.session_state.df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode("utf-8")

        st.download_button(
            label="⬇️ Télécharger le CSV",
            data=csv_data,
            file_name="clients_segmentes.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(
            "Contient toutes les colonnes originales + la colonne `Cluster`."
        )

    # --- Export rapport texte des descriptions de segments ---
    with col_exp2:
        st.markdown("#### 📝 Rapport descriptif (TXT)")

        # Génération du rapport texte
        nb_clusters = st.session_state.df["Cluster"].nunique()
        descriptions = generate_segment_description(st.session_state.df, nb_clusters)
        moyennes = st.session_state.df.groupby("Cluster")[COLONNES_RFM].mean().round(1)
        tailles = st.session_state.df["Cluster"].value_counts().sort_index()

        lignes_rapport = []
        lignes_rapport.append("=" * 60)
        lignes_rapport.append("RAPPORT DE SEGMENTATION DES CLIENTS")
        lignes_rapport.append(f"Algorithme : {st.session_state.algo_nom}")
        lignes_rapport.append(f"Nombre de clusters : {nb_clusters}")
        lignes_rapport.append(f"Score de silhouette : {st.session_state.silhouette:.4f}")
        lignes_rapport.append(f"Nombre de clients : {len(st.session_state.df)}")
        lignes_rapport.append("=" * 60)
        lignes_rapport.append("")

        for cluster_num in range(nb_clusters):
            taille = tailles.get(cluster_num, 0)
            desc = descriptions.get(cluster_num, "")
            moy_r = moyennes.loc[cluster_num, "Recency"]
            moy_f = moyennes.loc[cluster_num, "Frequency"]
            moy_m = moyennes.loc[cluster_num, "Monetary"]

            lignes_rapport.append(f"--- SEGMENT {cluster_num} ({taille} clients) ---")
            lignes_rapport.append(f"  Récence moyenne : {moy_r} jours")
            lignes_rapport.append(f"  Fréquence moyenne : {moy_f} achats")
            lignes_rapport.append(f"  Dépenses moyennes : {moy_m} €")
            lignes_rapport.append(f"  Description : {desc}")
            lignes_rapport.append("")

        rapport_texte = "\n".join(lignes_rapport).encode("utf-8")

        st.download_button(
            label="⬇️ Télécharger le rapport",
            data=rapport_texte,
            file_name="rapport_segmentation.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.caption(
            "Résumé textuel avec les métriques et descriptions de chaque segment."
        )

    # --- Aperçu du DataFrame enrichi ---
    st.markdown("---")
    st.subheader("👁️ Aperçu des données enrichies")
    st.dataframe(
        st.session_state.df.head(10),
        use_container_width=True,
        hide_index=True,
    )
