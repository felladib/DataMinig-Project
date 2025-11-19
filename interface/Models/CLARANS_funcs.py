import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
import random

#----------------------------------------CLUSTERING------------------------------------------------------------------------
def euclidean_distance(point1, point2):
    """
    Calcule la distance euclidienne entre deux points.
    Args:
        point1 (np.ndarray): Point de données 1 (1-D).
        point2 (np.ndarray): Point de données 2 (1-D).
    Returns:
        float: Distance euclidienne.
    """
    return euclidean(np.array(point1), np.array(point2))


def compute_cost(data_points, medoids, distance_metric):
    """
    Calcule le coût total pour un ensemble donné de médoïdes.
    Args:
        data_points (np.ndarray): Les points de données (2-D).
        medoids (set): Indices des médoïdes dans les données.
        distance_metric (callable): Fonction pour calculer la distance entre deux points.
    Returns:
        float: Le coût total.
    """
    total_cost = 0
    
    # Convertir les indices des médoïdes en vecteurs
    medoid_vectors = [data_points[medoid] for medoid in medoids]
    
    for point in data_points:
        # Calculer la distance entre ce point et tous les médoïdes
        distances = [distance_metric(point, medoid) for medoid in medoid_vectors]
        # Ajouter la distance minimale au coût total
        total_cost += min(distances)
    
    return total_cost


def replace_one_medoid(data, current_medoids, distance_metric):
    """
    Génère un nouvel ensemble de médoïdes en remplaçant un médoïde existant par un point non-médoïde.
    
    Args:
        data (list of list or np.ndarray): Les points de données.
        current_medoids (set): Ensemble des indices des médoïdes actuels.
        distance_metric (callable): Fonction pour calculer la distance entre deux points.
        
    Returns:
        set: Un nouvel ensemble de médoïdes avec un remplacement.
    """
    new_medoids = current_medoids.copy()
    non_medoids = set(range(len(data))) - current_medoids

    # Choisir un médoïde à remplacer et un point non-médoïde pour le remplacer
    medoid_to_replace = random.choice(list(current_medoids))
    new_medoid = random.choice(list(non_medoids))

    # Effectuer le remplacement
    new_medoids.remove(medoid_to_replace)
    new_medoids.add(new_medoid)

    return new_medoids


def assign_clusters(data, medoids):
    """
    Assigne chaque point de données au médoïde le plus proche.
    
    Args:
        data (list of list or np.ndarray): Les points de données.
        medoids (set): Ensemble des indices des médoïdes.
        distance_metric (callable): Fonction pour calculer la distance entre deux points.
        
    Returns:
        dict: Un dictionnaire où les clés sont les indices des médoïdes et les valeurs sont des listes
              des indices des points de données assignés à chaque médoïde.
    """
    clusters = {medoid: [] for medoid in medoids}

    for i, point in enumerate(data):
        # Trouver le médoïde le plus proche
        closest_medoid = min(medoids, key=lambda medoid: euclidean_distance(point, data[medoid]))
        # Ajouter le point au cluster correspondant
        clusters[closest_medoid].append(i)

    return clusters


def clarans_custom(data, k, numlocal, maxneighbor, stop_threshold , distance_metric=euclidean_distance):
    '''
    entrees:
        k : Dépend du problème, généralement entre 3 et 10.
        numlocal : Une valeur entre 10 et 20 est souvent utilisée.
        maxneighbor : Entre 5 et 10 dans les premières itérations.
        distance_metric : Par défaut, utiliser la distance euclidienne.
        stop_threshold: Seuil pour stopper la recherche si la différence de coût entre les itérations est inférieure à ce seuil.

    sorties:
        Best_Medoids, 
        Cluster_Assignment
    '''
    # Initialisation
    Best_Cost = float('inf')  # L'infini positif pour représenter un coût minimal initialement inconnu
    Best_Medoids = set()      # Un ensemble vide pour les meilleurs medoids

    for i in range(maxneighbor):
        Current_Medoids = set(random.sample(range(len(data)), k))
        Current_Cost = compute_cost(data, Current_Medoids, distance_metric)

        Neighbor_Count = 0
        while Neighbor_Count < numlocal:
            Neighbor = replace_one_medoid(data, Current_Medoids, distance_metric)
            Neighbor_Cost = compute_cost(data, Neighbor, distance_metric)

            if Neighbor_Cost < Current_Cost:
                Current_Medoids = Neighbor
                Current_Cost = Neighbor_Cost
                Neighbor_Count = 0
            else:
                Neighbor_Count += 1

        if Current_Cost < Best_Cost:
            Best_Medoids = Current_Medoids
            Best_Cost = Current_Cost

        if abs(Current_Cost - Best_Cost) < stop_threshold:
            break

    Cluster_Assignment = assign_clusters(data, Best_Medoids)

    return Best_Medoids, Cluster_Assignment
