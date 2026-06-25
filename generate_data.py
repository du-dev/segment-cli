"""
Script de génération de données simulées réalistes pour la segmentation de clients.

Ce script crée un jeu de données de 500 clients avec des groupes naturels
(VIP, réguliers, occasionnels, inactifs) pour démontrer l'efficacité du K-Means.

Le fichier généré est : data/sample_data.csv
"""

import numpy as np
import csv
from datetime import datetime, timedelta

# =============================================================================
# Paramètres de génération
# =============================================================================

np.random.seed(42)  # Graine aléatoire pour la reproductibilité

OUTPUT_FILE = "data/sample_data.csv"

# Dates de référence pour les achats
MIN_DATE = datetime(2023, 1, 1)
MAX_DATE = datetime(2024, 12, 31)
DAYS_RANGE = (MAX_DATE - MIN_DATE).days  # Nombre total de jours dans l'intervalle

# =============================================================================
# Définition des profils de clients (groupes naturels pour le clustering)
# Chaque profil est défini par : (taille, recency, frequency, monetary)
# Les plages sont données sous forme (min, max) pour chaque métrique RFM
# =============================================================================

PROFILS = {
    "vip": {
        "taille": 100,
        # Clients VIP : achats récents, fréquents, dépenses élevées
        "recency": (1, 30),       # Dernier achat très récent (1 à 30 jours)
        "frequency": (40, 100),   # Très nombreux achats
        "monetary": (2000, 5000), # Montants élevés
    },
    "reguliers": {
        "taille": 150,
        # Clients réguliers : valeurs moyennes, achats modérés
        "recency": (15, 90),      # Achat récent à modérément récent
        "frequency": (10, 40),     # Nombre d'achats moyen
        "monetary": (500, 2000),   # Dépenses moyennes
    },
    "occasionnels": {
        "taille": 150,
        # Clients occasionnels : achats peu fréquents, faibles dépenses
        "recency": (30, 180),     # Achat il y a quelques semaines/mois
        "frequency": (1, 10),      # Très peu d'achats
        "monetary": (10, 500),    # Faibles dépenses
    },
    "inactifs": {
        "taille": 100,
        # Clients inactifs : pas acheté depuis longtemps, tout le reste faible
        "recency": (200, 365),    # Dernier achat très ancien
        "frequency": (1, 5),      # Très peu d'achats
        "monetary": (10, 200),    # Dépenses minimales
    },
}


def generer_donnees():
    """
    Génère les données simulées pour les 500 clients répartis en 4 groupes.

    Returns:
        list: Liste de dictionnaires, un par client, avec les clés
              CustomerID, Recency, Frequency, Monetary, LastPurchaseDate.
    """
    clients = []
    customer_id = 1  # Compteur pour les identifiants uniques

    # --- Parcours de chaque profil de client ---
    for profil, params in PROFILS.items():
        for _ in range(params["taille"]):
            # Génération de valeurs aléatoires uniformes dans les plages définies
            recency = np.random.randint(
                params["recency"][0], params["recency"][1] + 1
            )
            frequency = np.random.randint(
                params["frequency"][0], params["frequency"][1] + 1
            )
            monetary = np.random.randint(
                params["monetary"][0], params["monetary"][1] + 1
            )

            # Calcul de la date du dernier achat à partir de la recency
            # On soustrait 'recency' jours à la date maximale (2024-12-31)
            ecart_jours = np.random.randint(0, recency + 1)
            last_purchase = MAX_DATE - timedelta(days=ecart_jours)

            # On s'assure que la date reste dans l'intervalle valide
            if last_purchase < MIN_DATE:
                last_purchase = MIN_DATE
            if last_purchase > MAX_DATE:
                last_purchase = MAX_DATE

            # Formatage de l'identifiant client (C001 à C500)
            id_str = f"C{customer_id:03d}"

            clients.append({
                "CustomerID": id_str,
                "Recency": recency,
                "Frequency": frequency,
                "Monetary": monetary,
                "LastPurchaseDate": last_purchase.strftime("%Y-%m-%d"),
            })

            customer_id += 1

    # --- Mélange aléatoire des clients pour ne pas avoir les groupes triés ---
    np.random.shuffle(clients)

    return clients


def ecrire_csv(clients, chemin):
    """
    Écrit les données des clients dans un fichier CSV avec les commentaires
    d'en-tête requis.

    Args:
        clients (list): Liste de dictionnaires client.
        chemin (str): Chemin du fichier CSV à créer.
    """
    # En-têtes du fichier CSV (commentaires métier)
    commentaires = [
        "# Source : données simulées de façon réaliste pour démonstration",
        "# Projet : Segmentation de clients - Application K-Means",
        "# Date de génération : 2024",
    ]

    # Noms des colonnes
    colonnes = ["CustomerID", "Recency", "Frequency", "Monetary", "LastPurchaseDate"]

    with open(chemin, "w", newline="", encoding="utf-8") as f:
        # Écriture des lignes de commentaire en haut du fichier
        for ligne in commentaires:
            f.write(ligne + "\n")

        # Écriture des données au format CSV
        writer = csv.DictWriter(f, fieldnames=colonnes)
        writer.writeheader()
        for client in clients:
            writer.writerow(client)

    print(f"Fichier CSV généré avec succès : {chemin}")
    print(f"Nombre total de clients : {len(clients)}")


# =============================================================================
# Point d'entrée du script
# =============================================================================

if __name__ == "__main__":
    # Génération des données simulées
    donnees = generer_donnees()

    # Écriture dans le fichier CSV
    ecrire_csv(donnees, OUTPUT_FILE)

    # Résumé des groupes générés
    print("\nRépartition par groupe :")
    for profil, params in PROFILS.items():
        print(f"  - {profil.capitalize()} : {params['taille']} clients")
