"""
Module de visualisation interactive pour la segmentation de clients.

Ce module fournit les graphiques Plotly utilisés dans l'interface Streamlit :
scatter plots 3D/2D, courbe du coude et radar comparatif des segments.

Flux de l'application :
1. Visualisation 3D des clusters (vue d'ensemble RFM)
2. Visualisation 2D (analyse détaillée de deux dimensions)
3. Courbe du coude (aide au choix du nombre de clusters K)
4. Graphique radar (comparaison des profils RFM par segment)
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# =============================================================================
# Palette de couleurs pour les clusters
# =============================================================================
# 10 couleurs distinctes et accessibles (bon contraste, adaptées au daltonisme)
# Utilisées pour tous les graphiques du module
CLUSTER_COLORS = [
    "#E63946",  # Rouge
    "#2196F3",  # Bleu
    "#4CAF50",  # Vert
    "#FF9800",  # Orange
    "#9C27B0",  # Violet
    "#00BCD4",  # Cyan
    "#FF5722",  # Rouge profond
    "#607D8B",  # Gris bleu
    "#8BC34A",  # Vert clair
    "#FFC107",  # Jaune ambre
]

# Colonnes RFM utilisées dans les graphiques
COLONNES_RFM = ["Recency", "Frequency", "Monetary"]

# Noms lisibles pour les axes et légendes
NOMS_AXIS = {
    "Recency": "Récence (jours)",
    "Frequency": "Fréquence (achats)",
    "Monetary": "Dépenses (€)",
}

# Nom de la colonne identifiant le cluster dans le DataFrame
COLONNE_CLUSTER = "Cluster"


# =============================================================================
# === VISUALISATION 3D ===
# =============================================================================

def plot_clusters_3d(df_with_clusters, k):
    """
    Crée un nuage de points 3D interactif montrant les segments de clients.

    Chaque point représente un client positionné dans l'espace RFM
    (Récence, Fréquence, Dépenses). La couleur indique le segment
    assigné par l'algorithme de clustering.

    Parameters
    ----------
    df_with_clusters : pandas.DataFrame
        DataFrame contenant au minimum les colonnes RFM et 'Cluster'.
        Si la colonne 'CustomerID' est présente, elle est affichée au survol.
    k : int
        Nombre de clusters (utilisé dans le titre et pour la palette).

    Returns
    -------
    plotly.graph_objects.Figure
        Figure Plotly 3D interactive (rotation, zoom, survol).
    """
    # --- Sélection des colonnes nécessaires ---
    df_plot = df_with_clusters.copy()

    # --- Construction du texte affiché au survol de chaque point ---
    # On inclut l'identifiant client si disponible, plus les valeurs RFM
    if "CustomerID" in df_plot.columns:
        df_plot["hover_text"] = (
            "<b>" + df_plot["CustomerID"].astype(str) + "</b><br>"
            + "Récence : " + df_plot["Recency"].astype(str) + " jours<br>"
            + "Fréquence : " + df_plot["Frequency"].astype(str) + " achats<br>"
            + "Dépenses : " + df_plot["Monetary"].astype(str) + " €"
        )
        hover_data = {"hover_text": True}
    else:
        hover_data = {
            "Recency": ":.0f",
            "Frequency": ":.0f",
            "Monetary": ":.0f",
        }

    # --- Création du scatter plot 3D ---
    fig = px.scatter_3d(
        data_frame=df_plot,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color=COLONNE_CLUSTER,
        title=f"Segmentation K-Means — {k} segments",
        labels={
            "Recency": NOMS_AXIS["Recency"],
            "Frequency": NOMS_AXIS["Frequency"],
            "Monetary": NOMS_AXIS["Monetary"],
            COLONNE_CLUSTER: "Segment",
        },
        color_discrete_sequence=CLUSTER_COLORS[:k],
        opacity=0.7,
        hover_data=hover_data,
    )

    # --- Personnalisation du survol ---
    if "CustomerID" in df_plot.columns:
        fig.update_traces(
            hovertemplate="%{customdata[0]}<extra></extra>",
            customdata=df_plot[["hover_text"]],
        )

    # --- Ajustement de la mise en page ---
    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        legend_title_text="Segment",
        height=700,
        margin=dict(l=0, r=0, t=60, b=0),
    )

    # Personnalisation des axes pour un affichage plus lisible
    fig.update_scenes(
        xaxis_title_text=NOMS_AXIS["Recency"],
        yaxis_title_text=NOMS_AXIS["Frequency"],
        zaxis_title_text=NOMS_AXIS["Monetary"],
    )

    return fig


# =============================================================================
# === VISUALISATION 2D ===
# =============================================================================

def plot_clusters_2d(df_with_clusters, x_col, y_col):
    """
    Crée un nuage de points 2D interactif pour deux dimensions RFM au choix.

    La taille de chaque point est proportionnelle au montant dépensé (Monetary),
    ce qui ajoute une troisième dimension d'information sur un graphique 2D.

    Parameters
    ----------
    df_with_clusters : pandas.DataFrame
        DataFrame contenant au minimum les colonnes RFM et 'Cluster'.
    x_col : str
        Nom de la colonne pour l'axe X (ex: "Recency").
    y_col : str
        Nom de la colonne pour l'axe Y (ex: "Frequency").

    Returns
    -------
    plotly.graph_objects.Figure
        Figure Plotly 2D interactive avec points colorés par cluster
        et dimensionnés par Monetary.
    """
    # --- Validation des colonnes demandées ---
    colonnes_disponibles = [x_col, y_col, COLONNE_CLUSTER, "Monetary"]
    for col in colonnes_disponibles:
        if col not in df_with_clusters.columns:
            raise ValueError(
                f"La colonne '{col}' n'existe pas dans le DataFrame. "
                f"Colonnes disponibles : {list(df_with_clusters.columns)}"
            )

    # --- Copie et préparation des données ---
    df_plot = df_with_clusters.copy()

    # --- Construction du texte au survol ---
    if "CustomerID" in df_plot.columns:
        df_plot["hover_text"] = (
            "<b>" + df_plot["CustomerID"].astype(str) + "</b><br>"
            + f"{x_col} : " + df_plot[x_col].astype(str) + "<br>"
            + f"{y_col} : " + df_plot[y_col].astype(str) + "<br>"
            + "Dépenses : " + df_plot["Monetary"].astype(str) + " €"
        )
    else:
        df_plot["hover_text"] = (
            f"{x_col} : " + df_plot[x_col].astype(str) + "<br>"
            + f"{y_col} : " + df_plot[y_col].astype(str) + "<br>"
            + "Dépenses : " + df_plot["Monetary"].astype(str) + " €"
        )

    # --- Détermination du nombre de clusters pour la palette ---
    nb_clusters = df_plot[COLONNE_CLUSTER].nunique()

    # --- Création du scatter plot 2D ---
    fig = px.scatter(
        data_frame=df_plot,
        x=x_col,
        y=y_col,
        color=COLONNE_CLUSTER,
        size="Monetary",
        size_max=40,
        title=f"Analyse {x_col} vs {y_col} (taille = Dépenses)",
        labels={
            x_col: NOMS_AXIS.get(x_col, x_col),
            y_col: NOMS_AXIS.get(y_col, y_col),
            COLONNE_CLUSTER: "Segment",
        },
        color_discrete_sequence=CLUSTER_COLORS[:nb_clusters],
        opacity=0.75,
        hover_data={"hover_text": True},
    )

    # --- Personnalisation du survol ---
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        customdata=df_plot[["hover_text"]],
    )

    # --- Ajustement de la mise en page ---
    fig.update_layout(
        title_font_size=16,
        title_x=0.5,
        legend_title_text="Segment",
        height=550,
        margin=dict(l=60, r=40, t=60, b=60),
    )

    return fig


# =============================================================================
# === COURBE DU COUDE ===
# =============================================================================

def plot_elbow_curve(df_optimal_k):
    """
    Crée un graphique combinant la courbe du coude (inertie) et le score de silhouette.

    Ce graphique à double axe Y aide l'utilisateur à choisir le nombre optimal
    de clusters K :
    - Axe Y gauche : inertie (diminue avec K, on cherche le "coude")
    - Axe Y droit : score de silhouette (on cherche le maximum)

    Parameters
    ----------
    df_optimal_k : pandas.DataFrame
        DataFrame retourné par find_optimal_k() contenant les colonnes :
        'k', 'inertia' et 'silhouette_score'.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure Plotly avec deux courbes et deux axes Y.
    """
    # --- Extraction des données ---
    valeurs_k = df_optimal_k["k"].values
    inertie = df_optimal_k["inertia"].values
    silhouette = df_optimal_k["silhouette_score"].values

    # --- Création de la figure avec deux axes Y ---
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
    )

    # --- Courbe de l'inertie (axe Y principal, gauche) ---
    # L'inertie diminue quand K augmente : on cherche le "coude"
    # où le gain marginal devient faible
    fig.add_trace(
        go.Scatter(
            x=valeurs_k,
            y=inertie,
            mode="lines+markers",
            name="Inertie",
            line=dict(color="#2196F3", width=3),
            marker=dict(size=8, color="#2196F3"),
        ),
        secondary_y=False,  # Axe Y gauche
    )

    # --- Courbe du score de silhouette (axe Y secondaire, droit) ---
    # Le score de silhouette mesure la qualité de la séparation
    # entre clusters (proche de 1 = bon)
    fig.add_trace(
        go.Scatter(
            x=valeurs_k,
            y=silhouette,
            mode="lines+markers",
            name="Score de silhouette",
            line=dict(color="#E63946", width=3),
            marker=dict(size=8, color="#E63946"),
        ),
        secondary_y=True,  # Axe Y droit
    )

    # --- Configuration des axes ---
    fig.update_xaxes(
        title_text="Nombre de clusters (K)",
        dtick=1,
        tick0=df_optimal_k["k"].min(),
    )

    # Axe Y gauche : inertie
    fig.update_yaxes(
        title_text="Inertie (somme des distances²)",
        secondary_y=False,
        gridcolor="#f0f0f0",
    )

    # Axe Y droit : silhouette
    fig.update_yaxes(
        title_text="Score de silhouette",
        secondary_y=True,
        gridcolor="rgba(0,0,0,0)",
        range=[0, 1],
    )

    # --- Mise en page ---
    fig.update_layout(
        title="Méthode du coude — Choix optimal de K",
        title_font_size=18,
        title_x=0.5,
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        height=550,
        margin=dict(l=80, r=80, t=80, b=60),
    )

    return fig


# =============================================================================
# === GRAPHIQUE RADAR ===
# =============================================================================

def plot_segment_radar(df_with_clusters, k):
    """
    Crée un graphique radar (toile d'araignée) comparant les profils RFM des segments.

    Chaque segment est représenté par une ligne de couleur différente sur le radar.
    Les axes correspondent aux trois dimensions RFM normalisées en pourcentage
    par rapport au maximum de chaque dimension, permettant une comparaison
    visuelle directe des profils.

    Parameters
    ----------
    df_with_clusters : pandas.DataFrame
        DataFrame contenant au minimum les colonnes RFM et 'Cluster'.
    k : int
        Nombre de clusters (utilisé pour itérer sur les segments).

    Returns
    -------
    plotly.graph_objects.Figure
        Figure Plotly de type radar avec une ligne par segment.
    """
    # --- Calcul des moyennes RFM par cluster ---
    moyennes = df_with_clusters.groupby(COLONNE_CLUSTER)[COLONNES_RFM].mean()

    # --- Normalisation en pourcentage du maximum global ---
    # Chaque valeur est exprimée en % du max de sa colonne
    # pour rendre les 3 axes comparables sur le radar
    max_par_colonne = df_with_clusters[COLONNES_RFM].max()

    # Construction du DataFrame normalisé (0-100%)
    radar_data = pd.DataFrame(index=moyennes.index, columns=COLONNES_RFM)
    for col in COLONNES_RFM:
        radar_data[col] = (moyennes[col] / max_par_colonne[col]) * 100

    # --- Création de la figure radar ---
    fig = go.Figure()

    # --- Ajout d'une ligne par segment ---
    for cluster_num in range(k):
        if cluster_num in radar_data.index:
            valeurs = radar_data.loc[cluster_num].values.tolist()
            # Fermeture du radar : on ajoute le premier point à la fin
            valeurs_fermees = valeurs + [valeurs[0]]

            fig.add_trace(go.Scatterpolar(
                r=valeurs_fermees,
                theta=COLONNES_RFM + [COLONNES_RFM[0]],
                fill="toself",
                name=f"Segment {cluster_num}",
                line_color=CLUSTER_COLORS[cluster_num % len(CLUSTER_COLORS)],
                opacity=0.5,
            ))

    # --- Configuration du radar ---
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                # Affichage des pourcentages sur l'axe radial
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0%", "25%", "50%", "75%", "100%"],
            ),
        ),
        title="Profil RFM des segments — Comparaison radar",
        title_font_size=18,
        title_x=0.5,
        legend=dict(
            orientation="h",
            y=-0.15,
            x=0.5,
            xanchor="center",
        ),
        height=550,
        margin=dict(l=80, r=80, t=80, b=100),
        showlegend=True,
    )

    return fig
