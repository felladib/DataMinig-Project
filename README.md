# Projet de Data Mining : Analyse et Prétraitement des Données

### Introduction

Ce projet a pour objectif d'effectuer une analyse exploratoire et un prétraitement des données à partir d'un ensemble de données climatiques couvrant l'algerie. L'analyse des données réelles est essentielle pour extraire des informations significatives et garantir la qualité des données avant l'application d'algorithmes de Data Mining.

### Objectifs du Projet

Le projet est divisé en deux parties principales :

# Projet de Data Mining - Partie 2
1. **Analyse Exploratoire des Données (AED)** : 
   - Examen des attributs, des valeurs et des distributions.
   - Identification des problèmes potentiels tels que les valeurs aberrantes et les données manquantes.

2. **Prétraitement des Données** :
   - Nettoyage des données, intégration de sources multiples, réduction de la taille des données, et transformation par normalisation.

### Fonctionnalités de l'Application

L'application développée doit permettre les manipulations de données suivantes :

#### A. Manipulation des Données
- Importation, visualisation et sauvegarde du contenu d'un dataset.
- Description globale du dataset.
- Mise à jour et suppression d'instances ou de valeurs du dataset.

#### B. Analyse des Caractéristiques des Attributs
- Calcul des mesures de tendance centrale et déduction des symétries.
- Calcul des mesures de dispersion et identification des valeurs aberrantes.
- Calcul des valeurs manquantes et des valeurs uniques.
- Construction de boxplots pour afficher les valeurs aberrantes.
- Construction d'histogrammes pour visualiser la distribution des données.
- Construction et affichage de scatter plots pour déduire des corrélations.

#### C. Prétraitement des Données
- Réduction des données par agrégation saisonnière.
- Intégration des données : fusion de données provenant de plusieurs sources.
- Options multiples pour traiter les valeurs aberrantes et les valeurs manquantes.
- Normalisation des données : méthodes Min-Max et z-score.
- Réduction des données via la discrétisation des données continues : égalité de fréquence/amplitude.
- Réduction des données par élimination des redondances (horizontal/vertical).

### Technologies Utilisées

- **Langage de programmation** : Python
- **Bibliothèques** :
  - Pandas : pour la manipulation des données.
  - NumPy : pour les calculs numériques.
  - Matplotlib et Seaborn : pour la visualisation des données.
  - Django : pour l'interface
 

### Description des jeux de données
#### Dataset `country` (Informations géographiques et administratives)

| Attribut      | Type       | Description |
|---------------|------------|-------------|
| **AREA**      | float      | La superficie totale de la région (en km², par exemple). |
| **PERIMETER** | float      | Le périmètre de la frontière du pays ou de la région. |
| **CNT1M_1_**  | int        | Un identifiant unique (numérique) pour chaque pays ou région. |
| **CNT1M_1_ID**| int        | Un autre identifiant unique, utilisé pour indexer les entités géographiques. |
| **FAO_NAME**  | string     | Nom du pays selon la FAO (Organisation des Nations unies pour l'alimentation et l'agriculture). |
| **FAO_CODE**  | int        | Code FAO du pays (numérique). |
| **UN_CODE**   | int        | Code numérique unique attribué par l'ONU pour chaque pays. |
| **ISO_CODE**  | string     | Code ISO du pays (2 lettres). |
| **CNTRY_NAME**| string     | Nom complet du pays. |
| **ISO3_CODE** | string     | Code ISO du pays (3 lettres). |
| **geometry**  | Polygon    | Géométrie du pays (données spatiales pour le contour du pays). |

#### Dataset `soil` (Données de sol)

| Attribut             | Type   | Description |
|----------------------|--------|-------------|
| **CNT_FULLNAME**     | string | Nom complet du pays ou de la région. |
| **sand % topsoil**   | float  | Pourcentage de sable dans le sol de surface. |
| **sand % subsoil**   | float  | Pourcentage de sable dans le sous-sol. |
| **silt % topsoil**   | float  | Pourcentage de limon dans le sol de surface. |
| **silt % subsoil**   | float  | Pourcentage de limon dans le sous-sol. |
| **clay % topsoil**   | float  | Pourcentage d'argile dans le sol de surface. |
| **clay % subsoil**   | float  | Pourcentage d'argile dans le sous-sol. |
| **pH water topsoil** | float  | pH du sol de surface mesuré avec de l'eau (indicateur d'acidité ou de basicité). |
| **pH water subsoil** | float  | pH du sous-sol mesuré avec de l'eau. |
| **OC % topsoil**     | float  | Pourcentage de carbone organique dans le sol de surface. |
| **OC % subsoil**     | float  | Pourcentage de carbone organique dans le sous-sol. |
| **N % topsoil**      | float  | Pourcentage d'azote dans le sol de surface. |
| **N % subsoil**      | float  | Pourcentage d'azote dans le sous-sol. |
| **BS % topsoil**     | float  | Saturation basique en surface (indicateur de fertilité). |
| **BS % subsoil**     | float  | Saturation basique dans le sous-sol. |
| **CEC topsoil**      | float  | Capacité d'échange cationique en surface. |
| **CEC subsoil**      | float  | Capacité d'échange cationique dans le sous-sol. |
| **CEC clay topsoil** | float  | Capacité d'échange cationique de l'argile en surface. |
| **CEC clay subsoil** | float  | Capacité d'échange cationique de l'argile dans le sous-sol. |
| **CaCO3 % topsoil**  | float  | Pourcentage de carbonate de calcium dans le sol de surface. |
| **CaCO3 % subsoil**  | float  | Pourcentage de carbonate de calcium dans le sous-sol. |
| **BD topsoil**       | float  | Densité apparente du sol en surface. |
| **BD subsoil**       | float  | Densité apparente dans le sous-sol. |
| **C/N topsoil**      | float  | Ratio carbone/azote en surface (indicateur de fertilité). |
| **C/N subsoil**      | float  | Ratio carbone/azote dans le sous-sol. |
| **geometry**         | Polygon | Géométrie des zones de sol (pour visualisation spatiale). |


#### Dataset `climate` (Données climatiques de pluie)

| Attribut       | Type      | Description |
|----------------|-----------|-------------|
| **lat**        | float     | Latitude du point de mesure. |
| **lon**        | float     | Longitude du point de mesure. |
| **time**       | datetime  | Timestamp indiquant la date et l'heure de la mesure. |
| **spatial_ref**| string    | Référence spatiale utilisée pour le système de coordonnées. |
| **Rainf**      | float     | Précipitations mesurées en millimètres (mm) pour le point donné. |


# Projet de Data Mining - Partie 2

## Description
La deuxième partie de ce projet se concentre sur les tâches de **régression** et de **clustering** appliquées à un dataset prétraité. Cette étape vise à implémenter, tester et comparer des algorithmes spécifiques pour chaque tâche, tout en analysant leurs performances.

---

## Tâches Réalisées

### 1. Régression
L'objectif est de prédire la variable **"Near-surface specific humidity"** à partir des autres variables du dataset.

#### Algorithmes Implémentés :
- **Decision Trees (Arbres de Décision)**
- **Random Forest (Forêt d'Arbres Aléatoires)**

#### Étapes Réalisées :
1. Division du dataset en données d'entraînement et de test.
2. Programmation des algorithmes de régression ci-dessus.
3. Application des algorithmes sur le dataset avec différents paramètres.
4. Évaluation des modèles en calculant les métriques adéquates et le temps d'exécution moyen.
5. Comparaison avec les versions préimplémentées des bibliothèques courantes (ex. : Scikit-learn).

---

### 2. Clustering
L'objectif est de regrouper les données en clusters significatifs pour révéler des structures sous-jacentes.

#### Algorithmes Implémentés :
- **CLARANS (Clustering Large Applications based upon RANdomized Search)**
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

#### Étapes Réalisées :
1. Programmation des algorithmes CLARANS et DBSCAN.
2. Évaluation des algorithmes avec divers paramètres.
3. Comparaison des résultats obtenus à l'aide de métriques pertinentes.
4. Illustration des résultats avec des graphiques et interprétation.

---

## Fonctionnalités Avancées de l'Interface
- **Sélection de la méthode de data mining à exécuter.**
- **Prédiction :** Entrer une nouvelle instance de données et prédire la valeur via les modèles de régression.
- **Visualisation :** Afficher le résultat de l'ACP (Analyse en Composantes Principales) des clusters.

---

## Résultats
- Les résultats obtenus pour chaque algorithme ont été analysés en termes de précision, robustesse et performance.
- Les graphiques permettent une meilleure interprétation des résultats et des comparaisons entre les algorithmes.

---
## Contributeurs
- [Fella DIB](github.com/felladib)
- [Lina IGHILAZA](github.com/liAghZn)


## Instructions
### Pré-requis
- Python 3.x
- Bibliothèques : `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`


### Instructions

1. Clonez ce repository :
   ```bash
   git clone https://github.com/votre_nom_dutilisateur/Projet_Data_Mining.git
   ```



