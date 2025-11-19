
"""
# **Random Forest algorithmes**
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


"""
## **preprocessing**
"""

# Étape 1 : Charger le dataset
# Remplacez 'your_dataset.csv' par le chemin de votre fichier CSV
data_path = "data/Training_Test_data/data_updated.csv"
df = pd.read_csv(data_path , sep=',')


# Vérifiez les premières lignes du dataset
print("Aperçu des données:")
print(df.head(5))

print(df.columns)


# Supprimer les colonnes qui se terminent par 'min' ou 'max'
df = df.drop(df.filter(regex='(min|max)$').columns, axis=1)

# print(df.columns)

# Étape 2 : Vérification des données manquantes
# print("\nStatistiques des valeurs manquantes:")
# print(df.isnull().sum())

# Remplissage ou suppression des valeurs manquantes (Exemple: suppression)
df = df.dropna()

# Sélectionner les colonnes numériques à normaliser
columns_to_normalize = ['Fall_PSurf_mean', 'Spring_PSurf_mean', 'Summer_PSurf_mean', 'Winter_PSurf_mean',
                        'Fall_Qair_mean', 'Spring_Qair_mean', 'Summer_Qair_mean', 'Winter_Qair_mean',
                        'Fall_Rainf_mean', 'Spring_Rainf_mean', 'Summer_Rainf_mean', 'Winter_Rainf_mean',
                        'Fall_Snowf_mean', 'Spring_Snowf_mean', 'Summer_Snowf_mean', 'Winter_Snowf_mean',
                        'Fall_Tair_mean', 'Spring_Tair_mean', 'Summer_Tair_mean', 'Winter_Tair_mean',
                        'Fall_Wind_mean', 'Spring_Wind_mean', 'Summer_Wind_mean', 'Winter_Wind_mean',
                        'sand % topsoil', 'sand % subsoil', 'silt % topsoil', 'silt% subsoil', 'clay % topsoil',
                        'clay % subsoil', 'pH water topsoil', 'pH water subsoil', 'OC % topsoil', 'OC % subsoil',
                        'N % topsoil', 'N % subsoil', 'BS % topsoil', 'BS % subsoil', 'CEC topsoil', 'CEC subsoil',
                        'CEC clay topsoil', 'CEC Clay subsoil', 'CaCO3 % topsoil', 'CaCO3 % subsoil', 'BD topsoil',
                        'BD subsoil', 'C/N topsoil', 'C/N subsoil']

# Initialiser le MinMaxScaler
scaler = MinMaxScaler()

# Appliquer le scaler aux colonnes sélectionnées
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])


# df.to_csv('data_for_Qair_pred.csv')

# %%
# Dataset pour l'automne (Fall)
# fall_columns = ['lat', 'lon', 'Fall_PSurf_mean', 'Fall_Qair_mean', 'Fall_Rainf_mean', 'Fall_Snowf_mean',
#                 'Fall_Tair_mean', 'Fall_Wind_mean' , 'sand % topsoil', 'sand % subsoil', 'silt % topsoil', 'silt% subsoil', 'clay % topsoil',
#                         'clay % subsoil', 'pH water topsoil', 'pH water subsoil', 'OC % topsoil', 'OC % subsoil',
#                         'N % topsoil', 'N % subsoil', 'BS % topsoil', 'BS % subsoil', 'CEC topsoil', 'CEC subsoil',
#                         'CEC clay topsoil', 'CEC Clay subsoil', 'CaCO3 % topsoil', 'CaCO3 % subsoil', 'BD topsoil',
#                         'BD subsoil', 'C/N topsoil', 'C/N subsoil','geometry']
# df_fall = df[fall_columns]

# # Dataset pour le printemps (Spring)
# spring_columns = ['lat', 'lon', 'Spring_PSurf_mean', 'Spring_Qair_mean', 'Spring_Rainf_mean', 'Spring_Snowf_mean',
#                   'Spring_Tair_mean', 'Spring_Wind_mean' ,'sand % topsoil', 'sand % subsoil', 'silt % topsoil', 'silt% subsoil', 'clay % topsoil',
#                         'clay % subsoil', 'pH water topsoil', 'pH water subsoil', 'OC % topsoil', 'OC % subsoil',
#                         'N % topsoil', 'N % subsoil', 'BS % topsoil', 'BS % subsoil', 'CEC topsoil', 'CEC subsoil',
#                         'CEC clay topsoil', 'CEC Clay subsoil', 'CaCO3 % topsoil', 'CaCO3 % subsoil', 'BD topsoil',
#                         'BD subsoil', 'C/N topsoil', 'C/N subsoil', 'geometry']
# df_spring = df[spring_columns]

# # Dataset pour l'été (Summer)
# summer_columns = ['lat', 'lon', 'Summer_PSurf_mean', 'Summer_Qair_mean', 'Summer_Rainf_mean', 'Summer_Snowf_mean',
#                   'Summer_Tair_mean', 'Summer_Wind_mean' ,'sand % topsoil', 'sand % subsoil', 'silt % topsoil', 'silt% subsoil', 'clay % topsoil',
#                         'clay % subsoil', 'pH water topsoil', 'pH water subsoil', 'OC % topsoil', 'OC % subsoil',
#                         'N % topsoil', 'N % subsoil', 'BS % topsoil', 'BS % subsoil', 'CEC topsoil', 'CEC subsoil',
#                         'CEC clay topsoil', 'CEC Clay subsoil', 'CaCO3 % topsoil', 'CaCO3 % subsoil', 'BD topsoil',
#                         'BD subsoil', 'C/N topsoil', 'C/N subsoil', 'geometry']
# df_summer = df[summer_columns]

# # Dataset pour l'hiver (Winter)
# winter_columns = ['lat', 'lon', 'Winter_PSurf_mean', 'Winter_Qair_mean', 'Winter_Rainf_mean', 'Winter_Snowf_mean',
#                   'Winter_Tair_mean', 'Winter_Wind_mean','sand % topsoil', 'sand % subsoil', 'silt % topsoil', 'silt% subsoil', 'clay % topsoil',
#                         'clay % subsoil', 'pH water topsoil', 'pH water subsoil', 'OC % topsoil', 'OC % subsoil',
#                         'N % topsoil', 'N % subsoil', 'BS % topsoil', 'BS % subsoil', 'CEC topsoil', 'CEC subsoil',
#                         'CEC clay topsoil', 'CEC Clay subsoil', 'CaCO3 % topsoil', 'CaCO3 % subsoil', 'BD topsoil',
#                         'BD subsoil', 'C/N topsoil', 'C/N subsoil', 'geometry']
# df_winter = df[winter_columns]

# # Affichage des 5 premières lignes de chaque dataset
# df_fall.to_csv('data_for_Qair_fall_pred.csv')
# print("Dataset Fall:")
# print(df_fall.head())
# df_spring.to_csv('data_for_Qair_spring_pred.csv')
# print("\nDataset Spring:")
# print(df_spring.head())
# df_summer.to_csv('data_for_Qair_summer_pred.csv')
# print("\nDataset Summer:")
# print(df_summer.head())
# df_winter.to_csv('data_for_Qair_winter_pred.csv')
# print("\nDataset Winter:")
# print(df_winter.head())

# %%
target_column = ['geometry']
df = df.drop(columns=target_column)

# %%
# df.columns

"""
## **data Split**
"""

# Définir les colonnes cibles et les colonnes explicatives
qair_columns = ['Fall_Qair_mean', 'Spring_Qair_mean', 'Summer_Qair_mean', 'Winter_Qair_mean']
features_columns = [col for col in df.columns if col not in qair_columns]

# Diviser les données en ensembles d'entraînement et de test
X = df[features_columns].values
Y = df[qair_columns].values

def train_test_split_custom(X, Y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split = int((1 - test_size) * X.shape[0])
    return X[indices[:split]], X[indices[split:]], Y[indices[:split]], Y[indices[split:]]

X_train, X_test, Y_train, Y_test = train_test_split_custom(X, Y)

# print(len(X_train))
# print(len(X_test))
# print(len(Y_train))
# print(len(Y_test))


def bootstrap_sampling(data, target):
    """
    Effectue un échantillonnage bootstrap sur les données et les cibles.
    - data : tableau 2D des données (caractéristiques).
    - target : tableau 2D des cibles (valeurs à prédire).
    
    Retourne un échantillon bootstrap de data et target.
    """
    n_samples = len(data)# Nombre total d'échantillons dans les données
    indices = np.random.randint(0, n_samples, size=n_samples)# Indices aléatoires avec remplacement
    # print(np.unique(indices))
    return data[indices], target[indices]# Renvoie les échantillons sélectionnés


# Calcul de réduction de variance pour multi-cibles
"""
En minimisant la variance dans les sous-groupes, on garantit que les prédictions sont basées sur des données homogènes.
Les feuilles représentent des moyennes locales avec peu de dispersion, ce qui donne des prédictions plus précises.
"""
def calculate_variance_reduction(S, S_left, S_right):
    """
    Calcule la réduction de variance pour une division donnée.
    - S : Ensemble complet des cibles (avant la division).
    - S_left : Cibles de la sous-division à gauche.
    - S_right : Cibles de la sous-division à droite.
    
    Retourne la réduction de variance totale résultant de la division.
    """
    var_total = np.sum(np.var(S, axis=0)) # Variance totale avant division
    var_left = np.sum(np.var(S_left, axis=0))# Variance de la sous-division gauche
    var_right = np.sum(np.var(S_right, axis=0))# Variance de la sous-division droite
    weight_left = len(S_left) / len(S)# Poids proportionnel des échantillons à gauche
    weight_right = len(S_right) / len(S) # Poids proportionnel des échantillons à droite
    return var_total - (weight_left * var_left + weight_right * var_right)# Réduction de variance

# ------------------------------Decision Tree Class ------------------------------
class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=10):
        """
        Initialise un arbre de régression.
        - max_depth : Profondeur maximale de l'arbre.
        - min_samples_split : Nombre minimum d'échantillons pour diviser un nœud.
        """
        self.max_depth = max_depth #La profondeur maximale de l'arbre. Cela limite la taille de l'arbre pour éviter le surapprentissage (overfitting)
        self.min_samples_split = min_samples_split #Le nombre minimum d'échantillons requis pour effectuer un split. Si le nombre d'échantillons restants est inférieur, le noeud devient une feuille.

    def fit(self, X, y):
        """
        Entraîne l'arbre de régression sur les données et les cibles fournies.
        - X : Données d'entraînement (caractéristiques).
        - y : Cibles d'entraînement (valeurs à prédire).
        """
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """
        Construit un sous-arbre récursivement.
        - X : Données actuelles.
        - y : Cibles actuelles.
        - depth : Profondeur actuelle dans l'arbre.
        
        Retourne un nœud terminal (moyenne des cibles) ou un sous-arbre.
        """
        
        # Cas de base (condition d'arrêt)
        if len(X) <= self.min_samples_split or depth >= self.max_depth : #Si le nombre d'échantillons est inférieur ou égal à min_samples_split, ou si la profondeur actuelle atteint max_depth, le nœud devient une feuille.
            return np.mean(y, axis=0)  # La valeur de la feuille est la moyenne des valeurs cibles y, car il s'agit d'une tâche de régression.

        best_split = None #Stockera les informations du meilleur split trouvé.
        max_variance_reduction = -float('inf')#Initialisé à une valeur très basse pour comparer les réductions de variance.

        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):#Pour chaque feature (colonne dans X), on teste tous les seuils uniques possibles.
                left_indices = X[:, feature] <= threshold #left_indices: Échantillons où la valeur de la caractéristique est inférieure ou égale au seuil.
                right_indices = X[:, feature] > threshold #right_indices: Échantillons où la valeur de la caractéristique est strictement supérieure au seuil.
                S_left, S_right = y[left_indices], y[right_indices]
                variance_reduction = calculate_variance_reduction(y, S_left, S_right) #Utilisation de la fonction calculate_variance_reduction pour mesurer l'efficacité du split.
                if variance_reduction > max_variance_reduction: #Si le split courant réduit davantage la variance que les splits précédents, on met à jour :
                    max_variance_reduction = variance_reduction #Nouvelle meilleure réduction.
                    best_split = (feature, threshold, left_indices, right_indices) #Informations sur ce split (caractéristique, seuil, indices gauche et droit).

        if best_split is None: #Si aucun split valide n'est trouvé (rare mais possible, par exemple si toutes les valeurs de la feature sont identiques), on retourne la moyenne des cibles y.
            return np.mean(y, axis=0)
        
        # Construction récursive des sous-arbres
        feature, threshold, left_indices, right_indices = best_split # On divise les données en sous-ensembles gauche et droit selon le meilleur split.
        # devlopper les sous-arbre gauche et droit
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return (feature, threshold, left_subtree, right_subtree)

    def predict(self, X):
        print(self.tree)
        return np.array([self._predict_one(x, self.tree) for x in X])#Pour chaque échantillon xdans X, elle appelle _predict_onepour descendre l'arbre et obtenir une prédiction.

    def _predict_one(self, x, tree):
        # Si treen'est pas un tuple, il s'agit d'une valeur moyenne (feuille). On la retourne directement.
        if not isinstance(tree, tuple):
            return tree
        feature, threshold, left_subtree, right_subtree = tree
        
        if x[feature] <= threshold:
            return self._predict_one(x, left_subtree) #Si x[feature] <= threshold, on descend dans le sous-arbre gauche.
        else:
            return self._predict_one(x, right_subtree) #Sinon, on descend dans le sous-arbre droit.


# --------------------------------Random Forest classe --------------------------------

class RandomForestRegressorr:
    #Cette classe s'appuie sur la classe DecisionTreeRegressor et utilise le principe de l'agrégation (bagging) pour améliorer la robustesse et la précision.
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=10):
        self.n_estimators = n_estimators #Nombre d'arbres à construire dans la forêt.
        self.max_depth = max_depth # Profondeur maximale de chaque arbre, contrôlant leur complexité , et eviter le overfitting
        self.min_samples_split = min_samples_split # Nombre minimum d'échantillons requis pour diviser un nœud dans chaque arbre.
        self.trees = []#self.trees: Liste contenant les instances de DecisionTreeRegressorreprésentant chaque arbre de la forêt.

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = bootstrap_sampling(X, y)# bootstrap_sampling est utilisée pour créer un échantillon aléatoire avec remplacement à partir des données Xet y.Cela garantit que chaque arbre est entraîné sur une version légèrement différente des données, favorisant la diversité des arbres.
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)#Pour chaque itération (selon n_estimators), un nouvel arbre est instancié avec les mêmes hyperparamètres ( max_depth, min_samples_split).
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Chaque arbre de la forêt effectue une prédiction sur les données X. Cela produit un tableau 2D, où chaque ligne correspond aux prédictions d'un arbre.
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # La moyenne des prédictions sur tous les arbres est calculée pour chaque échantillon.
        return predictions.mean(axis=0) #Cela donne un tableau 1D contenant les prédictions finales pour tous les échantillons de X


# ---------------------------------Training costum Random Forest RF------------------------------

import time
# Initialisation et entraînement du modèle
start_time = time.time()
model = RandomForestRegressorr(n_estimators=10, max_depth=20, min_samples_split=5) # 10 30 8
model.fit(X_train, Y_train)
rf_exe_time = time.time()-start_time


# ----------------------------------Test Costum random Forest------------------------------ 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# Prédictions
X_test = X_test
y_test = Y_test
predictions = model.predict(X_test)

# Calcul des métriques
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Calcul du R² Score
r2 = r2_score(y_test, predictions)
print("R² Score :", r2)
print("MAE :", mae)
print("RMSE :", rmse)
print("execution time : ",rf_exe_time)


#  -------------------------------Visualisation de costum Random Forest------------------------

target = ['Fall_Qair_mean', 'Spring_Qair_mean', 'Summer_Qair_mean', 'Winter_Qair_mean']

# Visualisation
plt.figure(figsize=(10, 6))
for i, col in enumerate(target):
    plt.subplot(2, 2, i + 1)
    plt.plot(y_test[:, i], label="Valeurs Réelles")
    plt.plot(predictions[:, i], label="Prédictions")
    plt.title(f"{col}")
    plt.legend()
    plt.xlabel("Échantillons")
    plt.ylabel("Valeurs")

plt.tight_layout()
plt.show()

#--------------------------------random Forest from sklearn----------------------------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialisation et entraînement du modèle
start_time =  time.time()
model = RandomForestRegressor(n_estimators=4, max_depth=10, random_state=42)
model.fit(X_train, Y_train)
rf_sklearn_exe_time = time.time() - start_time

# Prédictions
predictions_sklearn = model.predict(X_test)

# Calcul des métriques
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("MAE :", mae)
print("RMSE :", rmse)
print("R² Score :", r2)
print("execution time : ",rf_sklearn_exe_time)



# Visualisation des résultats
plt.figure(figsize=(10, 6))
for i, col in enumerate(target):
    plt.subplot(2, 2, i + 1)
    plt.plot(y_test[:, i], label="Valeurs Réelles")
    plt.plot(predictions[:, i], label="Prédictions")
    plt.title(f"{col}")
    plt.legend()
    plt.xlabel("Échantillons")
    plt.ylabel("Valeurs")

plt.tight_layout()
plt.show()

# --------------------------------Comparaison des erreurs entre les modèles (Custom RF vs Sklearn RF)--------------------------------
plt.figure(figsize=(10, 6))
for i, col in enumerate(target):
    plt.subplot(2, 2, i + 1)
    plt.grid()
    sns.kdeplot(y_test[:, i] - predictions[:, i], color='red', label=f'Custom RF - {target[i]}')
    sns.kdeplot(y_test[:, i] - predictions_sklearn[:, i], color='red', linestyle='--', label=f'Sklearn RF - {target[i]}')
    plt.title(f"{col}")
    plt.legend()
    plt.xlabel("Erreur")
    plt.ylabel("frequence")


plt.tight_layout()
# plt.title("Comparaison des erreurs entre les modèles (Custom RF vs Sklearn RF)")
plt.show()