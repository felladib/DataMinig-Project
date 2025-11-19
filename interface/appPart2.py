from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import adjusted_rand_score, silhouette_score, homogeneity_score, completeness_score,  v_measure_score, davies_bouldin_score
                            
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.model_selection import train_test_split
import sys
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
# from pyclustering.cluster.clarans import clarans
# from pyclustering.utils import read_sample
# from pyclustering.cluster.center_initializer import random_center_initializer
import inspect
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Print the source code
# print(inspect.getsource(clarans))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Models.CLARANS_funcs import clarans_custom

# import graphviz

def bootstrap_sampling(data, target):
    """
    Effectue un échantillonnage bootstrap sur les données et les cibles.
    - data : tableau 2D des données (caractéristiques).
    - target : tableau 2D des cibles (valeurs à prédire).
    
    Retourne un échantillon bootstrap de data et target.
    """
    # target = target.to_numpy()
    # data = data.to_numpy()
    n_samples = len(data)# Nombre total d'échantillons dans les données
    indices = np.random.randint(0, n_samples, size=n_samples)# Indices aléatoires avec remplacement
    # print(np.unique(indices))
    return data[indices], target[indices]# Renvoie les échantillons sélectionnés


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

# ------------------------------ Decision Tree for random forest Class ------------------------------
class DecisionTreeRegressorr:
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

# -------------------------------- Random Forest classe --------------------------------

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
            tree = DecisionTreeRegressorr(max_depth=self.max_depth, min_samples_split=self.min_samples_split)#Pour chaque itération (selon n_estimators), un nouvel arbre est instancié avec les mêmes hyperparamètres ( max_depth, min_samples_split).
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Chaque arbre de la forêt effectue une prédiction sur les données X. Cela produit un tableau 2D, où chaque ligne correspond aux prédictions d'un arbre.
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # La moyenne des prédictions sur tous les arbres est calculée pour chaque échantillon.
        return predictions.mean(axis=0) #Cela donne un tableau 1D contenant les prédictions finales pour tous les échantillons de X

# -------------------------------- Decision Tree classe --------------------------------
class DecisionTree:
    # Initialization of the DecisionTree class
    def __init__(self, max_depth=None, min_samples_split=2):
        # Stopping criteria : (because having too deep trees can cause overfitting)
        # max_depth: Maximum depth of the tree, None means no limit on depth
        # min_samples_split: Minimum number of samples required to split a node
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # Initialize an empty list to store trees for each target variable in multi-output regression
        self.trees = []

    # Fit method to train the decision tree model on the data
    def fit(self, X, Y):
        # Loop through each target column in the Y dataset (multi-output regression)
        for target in range(Y.shape[1]):  # Y.shape[1] gives the number of target variables
            # Combine X and the current target column from Y to create a new dataset
            # This allows us to build a tree for each target
            tree = self._build_tree(np.c_[X, Y[:, target]], depth=0)
            # Append the generated tree for this target to the trees list
            self.trees.append(tree)

    # Recursive method to build a decision tree
    def _build_tree(self, data, depth):
        # Base case: stop building the tree if there are not enough samples to split or if max depth is reached
        if len(data) < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            # Return the mean of the target variable as the prediction (leaf node)
            return np.mean(data[:, -1])
        
        # Find the best split based on the current dataset
        best_split = self._find_best_split(data)
        # If no valid split is found, return the mean of the target variable (leaf node)
        if not best_split:
            return np.mean(data[:, -1])
        
        # Recursively build the left and right subtrees
        left = self._build_tree(best_split['left'], depth + 1)
        right = self._build_tree(best_split['right'], depth + 1)
        # Return a dictionary representing the tree node (split criteria and child nodes)
        return {'feature': best_split['feature'], 'threshold': best_split['threshold'], 'left': left, 'right': right}

    # Method to find the best feature and threshold to split the data
    def _find_best_split(self, data):
        best_split = None  # Variable to store the best split found
        min_error = float('inf')  # Start with an infinitely large error value
        
        # Loop through each feature in the dataset (excluding the target column)
        for feature in range(data.shape[1] - 1):
            # Get the unique values of the feature to try as possible thresholds
            thresholds = np.unique(data[:, feature])
            # Try each threshold value for the feature
            for threshold in thresholds:
                # Split the data into left and right subsets based on the threshold
                left = data[data[:, feature] <= threshold]
                right = data[data[:, feature] > threshold]
                # Skip splits where either side is empty
                if len(left) > 0 and len(right) > 0:
                    # Calculate the error of the current split
                    error = self._split_error(left, right)
                    # If this split has less error than previous ones, update the best split
                    if error < min_error:
                        min_error = error
                        best_split = {'feature': feature, 'threshold': threshold, 'left': left, 'right': right}
        # Return the best split found
        return best_split

    # Method to calculate the error of a split (weighted variance of target values in left and right subsets)
    def _split_error(self, left, right):
        left_error = np.mean((left[:, -1] - np.mean(left[:, -1])) ** 2) * len(left)
        right_error = np.mean((right[:, -1] - np.mean(right[:, -1])) ** 2) * len(right)
        return left_error + right_error


    # Predict method to generate predictions for input data X
    def predict(self, X):
        predictions = []  # List to store predictions for each target variable
        # Loop through each tree (one for each target variable)
        for tree in self.trees:
            # For each tree, make predictions for each row in X
            predictions.append(np.array([self._predict_row(row, tree) for row in X]))
        # Return the predictions for all trees as a column stack (multi-output predictions)
        return np.column_stack(predictions)

    # Recursive method to make a prediction for a single row of data
    def _predict_row(self, row, tree):
        # If the tree is a leaf node (not a dictionary), return the predicted value (mean of target values)
        if not isinstance(tree, dict):
            return tree
        # Otherwise, traverse the tree based on the feature value and threshold
        if row[tree['feature']] <= tree['threshold']:
            return self._predict_row(row, tree['left'])  # Go left
        else:
            return self._predict_row(row, tree['right'])  # Go right

    # To visualize the tree
    # def visualize_tree(self, tree, feature_names):
    #     dot = graphviz.Digraph()
    #     self._add_node(dot, tree, feature_names, 0)
    #     return dot
    
    def _add_node(self, dot, tree, feature_names, node_id):
        if isinstance(tree, dict):
            feature_name = feature_names[tree['feature']]
            threshold = tree['threshold']
            dot.node(str(node_id), f'{feature_name} <= {threshold}')
            left_id = node_id * 2 + 1
            right_id = node_id * 2 + 2
            dot.edge(str(node_id), str(left_id), label="True")
            dot.edge(str(node_id), str(right_id), label="False")
            self._add_node(dot, tree['left'], feature_names, left_id)
            self._add_node(dot, tree['right'], feature_names, right_id)
        else:
            dot.node(str(node_id), f'Value: {tree:.2f}')
        return dot

# -------------------------------- DB Scan ---------------------------------------
def dbscan(X, eps, min_samples):
    n_points = X.shape[0]
    labels = -np.ones(n_points, dtype=int)  # Initialize all labels to -1 (unvisited) as integers
    cluster_id = 0

    # Helper function: calculate neighbors
    def region_query(point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= eps)[0]

    # Helper function: expand cluster
    def expand_cluster(point_idx, neighbors):
        nonlocal cluster_id
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1:  # Unvisited point
                labels[neighbor_idx] = cluster_id
                new_neighbors = region_query(neighbor_idx)
                if len(new_neighbors) >= min_samples:
                    neighbors = np.append(neighbors, new_neighbors)
            elif labels[neighbor_idx] == -1:  # Noise becomes part of cluster
                labels[neighbor_idx] = cluster_id
            i += 1

    # Main loop
    for point_idx in range(n_points):
        if labels[point_idx] != -1:
            continue  # Already processed
        neighbors = region_query(point_idx)
        if len(neighbors) < min_samples:
            labels[point_idx] = -1  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster(point_idx, neighbors)

    # Adjust cluster IDs to start from 0 instead of 1
    labels[labels != -1] -= 1

    return labels




# Initialisation de l'état dans st.session_state
if "X_train" not in st.session_state:
    # st.session_state["X_train"] = pd.read_csv("data/Training_data/X_train.csv")
    # st.session_state['features'] = st.session_state["X_train"].columns
    # st.session_state["X_test"] = pd.read_csv("data/Training_data/X_test.csv")
    # st.session_state["y_train"] = pd.read_csv("data/Training_data/y_train.csv")
    # st.session_state["y_test"] = pd.read_csv("data/Training_data/y_test.csv")
    
    st.session_state["X_train"] = pd.read_csv("data/rd_training_data/X_train.csv")
    st.session_state['features'] = st.session_state["X_train"].columns
    st.session_state["X_test"] = pd.read_csv("data/rd_training_data/X_test.csv")
    st.session_state["y_train"] = pd.read_csv("data/rd_training_data/y_train.csv")
    st.session_state["y_test"] = pd.read_csv("data/rd_training_data/y_test.csv")
    st.session_state["target"] = st.session_state["y_train"].columns

    st.session_state["X_test"]= st.session_state["X_test"].to_numpy()
    st.session_state["y_test"]= st.session_state["y_test"].to_numpy()
    st.session_state["X_train"]= st.session_state["X_train"].to_numpy()
    st.session_state["y_train"]= st.session_state["y_train"].to_numpy()
    st.session_state["selected_target_random_forest"] = None
    st.session_state["selected_target_decision_tree"] = None
    
    
# Random Forest
# if 'custom_random_forest_model' not in st.session_state:
#     # Modèle personnalisé Random Forest
#     start_time = time.time()
#     st.session_state['custom_random_forest_model'] = RandomForestRegressorr(n_estimators=10, max_depth=20, min_samples_split=10)
#     st.session_state['custom_random_forest_model'].fit(st.session_state["X_train"], st.session_state["y_train"])
#     st.session_state['custom_random_forest_model_exe_time'] = time.time() - start_time

#     # Modèle scikit-learn Random Forest
#     start_time = time.time()
#     st.session_state['sklearn_random_forest_model'] = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=42)
#     st.session_state['sklearn_random_forest_model'].fit(st.session_state["X_train"], st.session_state["y_train"])
#     st.session_state['sklearn_random_forest_model_exe_time'] = time.time() - start_time


# if 'custom_decision_tree_model' and 'sklearn_decision_tree_model' not in st.session_state:
#     # Modèle personnalisé Random Forest
#     start_time = time.time()
#     st.session_state['custom_decision_tree_model'] = DecisionTree(max_depth=10, min_samples_split=10)
#     st.session_state['custom_decision_tree_model'].fit(st.session_state["X_train"], st.session_state["y_train"])
#     st.session_state['custom_decision_tree_model_exe_time'] = time.time() - start_time

#     # Modèle scikit-learn Random Forest
#     start_time = time.time()
#     st.session_state['sklearn_decision_tree_model'] = DecisionTreeRegressor(max_depth=10, min_samples_split=10, random_state=42)
#     st.session_state['sklearn_decision_tree_model'].fit(st.session_state["X_train"], st.session_state["y_train"])
#     st.session_state['sklearn_decision_tree_model_exe_time'] = time.time() - start_time




class App:
    def __init__(self):
        self.custom_model =  None
        self.sklearn_model = None
        self.make_prediction = False
        

        

    def run_regression(self):
        # st.title("Model Testing Application")

        # Selection du modèle
        model_choice = st.selectbox("Choisissez un modèle à tester", [
            "Random Forest",
            "Decision Tree",
        ])
        
        if model_choice == "Random Forest":
            n_estimators = st.slider("Nombre d'arbres (n_estimators)", 10, 200, 100, step=10)
            max_depth = st.slider("Profondeur maximale (max_depth)", 1, 50, 10, step=1)
            min_samples_split = st.slider("Minimum d'échantillons pour une division (min_samples_split)", 2, 10, 2, step=1)
            
            # Sélection de la ou des cibles
            target_options = ['Fall_Qair_mean', 'Spring_Qair_mean', 'Summer_Qair_mean', 'Winter_Qair_mean', "All"]
            st.session_state["selected_target"]  = st.selectbox("Choisissez un ou plusieurs targets", target_options)
            st.session_state["selected_target_random_forest"]  = st.session_state["selected_target"] 
            print(f" je suis dans random forest et je veux afficher le seletcted target : {st.session_state["selected_target_random_forest"]}")
            
            if st.button("Entraîner Random Forest"):
                
                # print(st.session_state["selected_target"])
                # Gestion des cibles
                if st.session_state["selected_target"]  == "All":
                    y_train_filtered = st.session_state["y_train"]  # Utilisez toutes les colonnes
                else:
                    if st.session_state["selected_target"]  in st.session_state["target"]:
                        y_train_filtered = st.session_state["y_train"][:, st.session_state["target"] == st.session_state["selected_target"]]
                    else:
                        st.error("La cible sélectionnée n'existe pas dans les données.")
                        y_train_filtered = None
                print(y_train_filtered)
                if y_train_filtered is not None:
                    
                    print("part 0: {} {} {} {}".format(n_estimators, max_depth, min_samples_split, y_train_filtered.shape))

                    start_time = time.time()
                    st.session_state['custom_random_forest_model'] = RandomForestRegressorr(
                                                                                            n_estimators=n_estimators,
                                                                                            max_depth=max_depth,
                                                                                            min_samples_split=min_samples_split
                                                                                            )
                    print("part 1")
                    st.session_state['custom_random_forest_model'].fit(st.session_state["X_train"], y_train_filtered)
                    st.session_state['custom_random_forest_model_exe_time'] = time.time() - start_time

                    
                    start_time = time.time()
                    st.session_state["sklearn_random_forest_model"]  = RandomForestRegressor(
                                                                                            n_estimators=n_estimators,
                                                                                            max_depth=max_depth,
                                                                                            min_samples_split=min_samples_split,
                                                                                            random_state=42,
                                                                                        )
                    st.session_state["sklearn_random_forest_model"] .fit(st.session_state["X_train"], y_train_filtered)
                    st.session_state["sklearn_random_forest_model_exe_time"] = time.time() - start_time
                    

                    st.success(f"Random Forest entraîné en {st.session_state['custom_random_forest_model_exe_time']:.2f} secondes.")
            
        elif model_choice == "Decision Tree":
            max_depth = st.slider("Profondeur maximale (max_depth)", 1, 50, 10, step=1)
            min_samples_split = st.slider("Minimum d'échantillons pour une division (min_samples_split)", 2, 10, 2, step=1)
            # min_samples_leaf = st.slider("Minimum d'échantillons par feuille (min_samples_leaf)", 1, 10, 1, step=1)
            # Sélection de la ou des cibles
            target_options = ['Fall_Qair_mean', 'Spring_Qair_mean', 'Summer_Qair_mean', 'Winter_Qair_mean', "All"]
            st.session_state["selected_target"]  = st.selectbox("Choisissez un ou plusieurs targets", target_options)
            st.session_state["selected_target_decision_tree"]  = st.session_state["selected_target"] 
            if st.button("Entraîner Decision Tree"):
                
                if st.session_state["selected_target"]  == "All":
                    y_train_filtered = st.session_state["y_train"]  # Utilisez toutes les colonnes
                else:
                    if st.session_state["selected_target"]  in st.session_state["target"]:
                        y_train_filtered = st.session_state["y_train"][:, st.session_state["target"] == st.session_state["selected_target"]]
                    else:
                        st.error("La cible sélectionnée n'existe pas dans les données.")
                        y_train_filtered = None
                
                print(y_train_filtered)
                
                if y_train_filtered is not None:
                    start_time = time.time()
                    st.session_state['custom_decision_tree_model'] = DecisionTree(
                                                                                max_depth=max_depth,
                                                                                min_samples_split=min_samples_split
                                                                                )
                    st.session_state['custom_decision_tree_model'].fit(st.session_state["X_train"], y_train_filtered)
                    st.session_state['custom_decision_tree_model_exe_time'] = time.time() - start_time
                    
                    # entrainemenent de sklearn decision tree
                    start_time = time.time()
                    st.session_state['sklearn_decision_tree_model'] = DecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=42,
                    )
                    st.session_state['sklearn_decision_tree_model'].fit(st.session_state["X_train"], y_train_filtered)
                    st.session_state["sklearn_decision_tree_model_exe_time"] = time.time() - start_time

                    

                    st.success(f"Decision Tree entraîné en {st.session_state['custom_decision_tree_model_exe_time']:.2f} secondes.")
            
        
        # Mode de test : instance personnalisée ou donnée de test
        test_mode = st.checkbox("Tester une instance personnalisée")

        if test_mode:
            st.write("Veuillez entrer les valeurs de l'instance personnalisée :")
            instance_input = st.text_area("Entrez les valeurs de l'instance (séparées par des virgules):")
            # st.markdown("<span style='color:#FF4B4B;'>Entrez les valeurs de l'instance (séparées par des virgules):</span>", unsafe_allow_html=True)
            # instance_input = st.text_area("")
            
            if st.button("Prédire"):
                self.make_prediction = False
                
                try:
                    instance = np.array([float(x) for x in instance_input.split(",")]).reshape(1, -1)
                    self.custom_model = st.session_state.get(f"custom_{model_choice.lower().replace(' ', '_')}_model" , None)
                    self.sklearn_model = st.session_state.get(f"sklearn_{model_choice.lower().replace(' ', '_')}_model", None)

                    st.write("### Résultats des prédictions")
                    if self.custom_model:
                        st.write("Custom Model Prediction:", self.custom_model.predict(instance))

                    if self.sklearn_model:
                        st.write("Sklearn Model Prediction:", self.sklearn_model.predict(instance))
                    
                    self.make_prediction = True
                except Exception as e:
                    st.error(f"Erreur: {e}")

        else:
            st.write("Veuillez sélectionner une instance du jeu de test :")
            index = st.number_input("Indice de l'instance", min_value=0, max_value=len(st.session_state["X_test"])-1, step=1, value=0)
            
            
            if st.button("Prédire"):
                if model_choice == "Random Forest":
                    st.session_state["selected_target"] = st.session_state["selected_target_random_forest"]
                    print(f" au cas ou random forest {st.session_state['selected_target']}" )
                elif model_choice == "Decision Tree":
                    st.session_state["selected_target"] = st.session_state["selected_target_decision_tree"]
                    print(f" au cas ou random decision tree {st.session_state['selected_target']}" )
                    
                    
                self.make_prediction = False
                # Extraction de l'instance de test
                X_test_instance = st.session_state["X_test"][index:index + 1]
                print(f" la valeur de X_test : {X_test_instance}")
                if st.session_state["selected_target"] == 'All':
                    y_test_instance = st.session_state["y_test"][index]  # Valeurs réelles pour les targets
                    target_names = list(st.session_state["target"])
                else:
                    y_test_instance = st.session_state["y_test"][index, st.session_state["target"] == st.session_state["selected_target"] ]
                    print(f" la valeur de y_test_instance : {y_test_instance}")
                    target_names = [st.session_state["selected_target"]]
                    
                print(f"selected target : {st.session_state["selected_target"]}")
                
                # Récupération des modèles
                self.custom_model = st.session_state.get(f"custom_{model_choice.lower().replace(' ', '_')}_model", None)
                self.sklearn_model = st.session_state.get(f"sklearn_{model_choice.lower().replace(' ', '_')}_model", None)

                # Récupération des noms des features et des targets
                feature_names = list(st.session_state['features'])
                # target_names = list(st.session_state["selected_target"] if st.session_state["selected_target"]!= 'All' else st.session_state["target"])


                # Prédictions Custom
                custom_predictions = (
                    self.custom_model.predict(X_test_instance).flatten() if self.custom_model else [None] * len(target_names)
                )

                # Prédictions Sklearn
                sklearn_predictions = (
                    self.sklearn_model.predict(X_test_instance).flatten() if self.sklearn_model else [None] * len(target_names)
                )

                # Création d'un dictionnaire pour les features
                feature_data = {feature_names[i] : X_test_instance[0, i] for i in range(X_test_instance.shape[1])}

                # Ajout des valeurs réelles et des prédictions
                data = {
                    "Type": ["Valeurs réelles", "Sklearn Predictions", "Custom Predictions"],
                }
                

                # Ajout des features aux lignes
                for key, value in feature_data.items():
                    data[key] = [value, value, value]

                print(f"-------------------Xtest instance {y_test_instance}")
                # Ajout des targets aux lignes
                for i, target in enumerate(target_names):
                    data[target] = [
                        y_test_instance[i],  # Valeurs réelles
                        sklearn_predictions[i],  # Sklearn Predictions
                        custom_predictions[i],  # Custom Predictions
                    ]

                # Conversion en DataFrame
                results_df = pd.DataFrame(data)

                # Affichage
                st.write("### Résultats des prédictions")
                st.dataframe(results_df)

                self.make_prediction = True  # Marquer les prédictions comme effectuées  
                
            
        if  self.make_prediction == True:
            st.write("### Métriques") 
            
            if self.custom_model:
                self.custom_predictions = self.custom_model.predict(st.session_state["X_test"])
                
            if self.sklearn_model:
                
                self.sklearn_predictions = self.sklearn_model.predict(st.session_state["X_test"]) if self.sklearn_model else None
                
                st.write("#### Custom Model")
                if st.session_state["selected_target"] == 'All':
                    self.display_metrics(st.session_state["y_test"], self.custom_predictions, st.session_state[f"custom_{model_choice.lower().replace(' ', '_')}_model_exe_time"])
                    st.write("#### Sklearn Model")
                    self.display_metrics(st.session_state["y_test"], self.sklearn_predictions, st.session_state[f"sklearn_{model_choice.lower().replace(' ', '_')}_model_exe_time"])
                    st.write("### Comparaison Sklearn Model et Custom Model")
                    self.visualis_distribution(st.session_state["y_test"],self.custom_predictions,self.sklearn_predictions)
                else :
                    print(f" l'indice de target est : {st.session_state["target"] == st.session_state["selected_target"]}")
                    self.display_metrics(
                                        st.session_state["y_test"][:, st.session_state["target"] == st.session_state["selected_target"]], 
                                        self.custom_predictions, 
                                        st.session_state[f"custom_{model_choice.lower().replace(' ', '_')}_model_exe_time"],
                                        target = st.session_state["selected_target"]      
                    )
                    
                    st.write("#### Sklearn Model")
                    self.display_metrics(
                                        st.session_state["y_test"][:, st.session_state["target"] == st.session_state["selected_target"]], 
                                        self.sklearn_predictions, st.session_state[f"sklearn_{model_choice.lower().replace(' ', '_')}_model_exe_time"] ,
                                        sklearn = True,
                                        target = st.session_state["selected_target"]
                    )
                    st.write("### Comparaison Sklearn Model et Custom Model")
                    if model_choice ==  "Random Forest":
                        self.visualis_distribution(
                                            y_test = st.session_state["y_test"][:, st.session_state["target"] == st.session_state["selected_target"]],
                                            custom_model_predictions = self.custom_predictions,
                                            sklearn_model_predictions=self.sklearn_predictions ,
                                            name_algo = 'RF',
                                            target = st.session_state["selected_target"]
                        )
                    elif model_choice == "Decision Tree":
                        self.visualis_distribution(
                                            y_test = st.session_state["y_test"][:, st.session_state["target"] == st.session_state["selected_target"]],
                                            custom_model_predictions = self.custom_predictions,
                                            sklearn_model_predictions=self.sklearn_predictions ,
                                            name_algo = 'DT',
                                            target = st.session_state["selected_target"]
                        )
            
                    
     
     
     
     
            
            
        
    def run_clustering(self):
        # st.title("Clustering Application")

        # Sélection de l'algorithme
        algo_choice = st.selectbox("Choisissez un algorithme de clustering", ["DBSCAN", "CLARANS"])

        if algo_choice == "DBSCAN":
            st.subheader("Paramètres DBSCAN")
            eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=10.0, step=0.1, value=0.5)
            min_samples = st.slider("Nombre minimum d'échantillons (min_samples)", min_value=1, max_value=20, value=5)

            if st.button("Lancer DBSCAN"):
                

                # Application des modèles
                starttime = time.time()
                labels_custom = dbscan(st.session_state["X_train"], eps, min_samples)
                st.session_state['dbscan_exe_time'] = time.time() - starttime 
                st.success(f"Clustering avec C terminé en {st.session_state['dbscan_exe_time']:.2f} secondes. Nombre de clusters : .")

                print(f"\n les cluster de dbscan {labels_custom} ")
                sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_samples)
                labels_sklearn = sklearn_dbscan.fit_predict(st.session_state["X_train"])

                # Stockage dans le session_state
                dbscan_results = pd.DataFrame(st.session_state["X_train"], columns=[st.session_state["features"][i] for i in range(st.session_state["X_train"].shape[1])])
                dbscan_results["Custom_Cluster"] = labels_custom
                dbscan_results["Sklearn_Cluster"] = labels_sklearn

                # Stockage dans le session_state
                st.session_state['dbscan_results'] = dbscan_results
                # print(f"db scan result {st.session_state['dbscan_results']}")
                
                # st.success("Clustering avec DBSCAN terminé.")
                self.display_clustering_results("DBSCAN")
                
                self.visualize_clusters(dbscan_results.iloc[:, :25], dbscan_results["Custom_Cluster"], dbscan_results["Sklearn_Cluster"])
                silhouette, dbi, calinski_harabasz = self.evaluate_clustering(dbscan_results.iloc[:, :25], dbscan_results["Custom_Cluster"], 'DBSCAN')
                st.subheader("Métriques des Algorithmes (DBSCAN)")
                
                st.write("* silhouette : " ,silhouette)
                st.write("* dbi (Davies-Bouldin Index) :" , dbi)
                st.write("* calinski_harabasz :" , calinski_harabasz)
                
                
                
                if "clarans_results" not in st.session_state or st.session_state["clarans_results"] is None:
                    st.warning("Les résultats de l'algorithme CLARANS ne sont pas disponibles.")
                elif "dbscan_results" not in st.session_state or st.session_state["dbscan_results"] is None:
                    st.warning("Les résultats de l'algorithme DBSCAN ne sont pas disponibles.")
                else:
                    
                    # metrics_df = self.compare_algorithms(
                    #         dbscan_labels=st.session_state["dbscan_results"]["Custom_Cluster"],
                    #         clarans_labels= st.session_state['clarans_results'].iloc[:, 25:26]
                    # )
                    
                    clarans_silhouette, clarans_dbi, clarans_calinski = self.evaluate_clustering(
                                                            st.session_state["X_train"], 
                                                            st.session_state['clarans_results'].iloc[:, 25:26],
                                                            'CLARANS'
                                                            )
                    
                        

                    # Calcul des métriques pour DBSCAN
                    dbscan_silhouette, dbscan_dbi, dbscan_calinski = self.evaluate_clustering(st.session_state["X_train"], 
                                                                                           st.session_state["dbscan_results"]["Custom_Cluster"],
                                                                                           'DBSCAN'
                                                                                           )

                    # Organisation des résultats dans un DataFrame Pandas pour Streamlit
                    metrics_data = {
                        "Algorithme": ["CLARANS", "DBSCAN"],
                        "Silhouette Score": [clarans_silhouette, dbscan_silhouette],
                        "Davies-Bouldin Index": [clarans_dbi, dbscan_dbi],
                        "Calinski-Harabasz Index": [clarans_calinski, dbscan_calinski],
                    }

                    df_metrics = pd.DataFrame(metrics_data)
                    # Affichage dans un tableau structuré
                    st.write("### Comparaison entre CLARANS et DBSCAN")
                    st.write("""
                            - **Silhouette Score** : Évalue la séparation et la cohésion des clusters (proche de 1 = meilleur clustering).
                            - **Davies-Bouldin Index** : Mesure la qualité des clusters en termes de compacité et séparation (plus faible est meilleur).
                            - **Calinski-Harabasz Index** : Évalue le rapport entre la dispersion intra-cluster et inter-cluster (plus élevé est meilleur).
                        """)
                    st.table(df_metrics)
                    # st.table(metrics_df)
                    
        elif algo_choice == "CLARANS":
            st.subheader("Paramètres CLARANS")
            k = st.slider("Nombre de clusters (k)", min_value=2, max_value=10, step=1, value=3)
            numlocal = st.slider("Nombre de recherches locales (numlocal)", min_value=10, max_value=20, step=1, value=13)
            maxneighbor = st.slider("Nombre maximal de voisins (maxneighbor)", min_value=5, max_value=10, step=1, value=6)
            stop_threshold = st.slider("Seuil d'arrêt (stop_threshold)", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
            # print(k , numlocal , maxneighbor , stop_threshold)
            if st.button("Lancer CLARANS"):
                # Chargement des données
                # X_scaled = st.session_state["X_train"]

                # Application de CLARANS
                tempstart = time.time()
                Best_Medoids, Cluster_Assignment = clarans_custom(
                    data=st.session_state["X_train"], k=k, numlocal=numlocal,
                    maxneighbor=maxneighbor,
                    stop_threshold=stop_threshold
                )
                st.session_state["clarans_exe_time"] = time.time() - tempstart
                print(Cluster_Assignment)
                # Fusion avec X_train
              
                clarans_results = self.convert_clarans_to_dataframe(st.session_state["X_train"], Cluster_Assignment)

                # Stockage dans le session_state
                st.session_state['clarans_results'] = clarans_results

                st.success(f"Clustering avec CLARANS terminé en {st.session_state['clarans_exe_time']:.2f} secondes. Nombre de clusters : .")

                # Affichage des résultats
                self.display_clustering_results("CLARANS")
                print(clarans_results.iloc[:, :25].columns) #features value
                print(clarans_results.iloc[:, 25:27].columns) # cluster et medoid
                self.visualize_clusters( clarans_results.iloc[:, :25], clarans_results.iloc[:, 25:27], sklearn_clusters=None)
                silhouette, dbi, calinski_harabasz = self.evaluate_clustering(clarans_results.iloc[:, :25], clarans_results.iloc[:, 25:26], 'CLARANS')
                st.subheader("Métriques des Algorithmes (CLARANS)")
                # st.write("""
                #         - **Silhouette Score** : Évalue la séparation et la cohésion des clusters (proche de 1 = meilleur clustering).
                #         - **Davies-Bouldin Index** : Mesure la qualité des clusters en termes de compacité et séparation (plus faible est meilleur).
                #         - **Calinski-Harabasz Index** : Évalue le rapport entre la dispersion intra-cluster et inter-cluster (plus élevé est meilleur).
                #         """)
                
                st.write("* silhouette : " ,silhouette)
                st.write("* dbi :" , dbi)
                st.write("* calinski_harabasz :" , calinski_harabasz)
                
                # Vérification des résultats dans la session
                if "clarans_results" not in st.session_state or st.session_state["clarans_results"] is None:
                    st.warning("Les résultats de l'algorithme CLARANS ne sont pas disponibles.")
                elif "dbscan_results" not in st.session_state or st.session_state["dbscan_results"] is None:
                    st.warning("Les résultats de l'algorithme DBSCAN ne sont pas disponibles.")
                else:
                    # Comparaison des algorithmes
                    # metrics_df = self.compare_algorithms(
                    #         dbscan_labels=st.session_state["dbscan_results"]["Custom_Cluster"],
                    #         clarans_labels= st.session_state['clarans_results'].iloc[:, 25:26]
                    # )
                    clarans_silhouette, clarans_dbi, clarans_calinski = self.evaluate_clustering(
                                                            st.session_state["X_train"], 
                                                            st.session_state['clarans_results'].iloc[:, 25:26],
                                                            'CLARANS'
                                                            )
                    
                        

                    # Calcul des métriques pour DBSCAN
                    dbscan_silhouette, dbscan_dbi, dbscan_calinski = self.evaluate_clustering(st.session_state["X_train"], 
                                                                                           st.session_state["dbscan_results"]["Custom_Cluster"],
                                                                                           'DBSCAN'
                                                                                           )

                    # Organisation des résultats dans un DataFrame Pandas pour Streamlit
                    metrics_data = {
                        "Algorithme": ["CLARANS", "DBSCAN"],
                        "Silhouette Score": [clarans_silhouette, dbscan_silhouette],
                        "Davies-Bouldin Index": [clarans_dbi, dbscan_dbi],
                        "Calinski-Harabasz Index": [clarans_calinski, dbscan_calinski],
                    }

                    df_metrics = pd.DataFrame(metrics_data)

                    # Affichage dans un tableau structuré
                    st.write("### Comparaison entre CLARANS et DBSCAN")
                    st.write("""
                            - **Silhouette Score** : Évalue la séparation et la cohésion des clusters (proche de 1 = meilleur clustering).
                            - **Davies-Bouldin Index** : Mesure la qualité des clusters en termes de compacité et séparation (plus faible est meilleur).
                            - **Calinski-Harabasz Index** : Évalue le rapport entre la dispersion intra-cluster et inter-cluster (plus élevé est meilleur).
                        """)
                    st.table(df_metrics)
                    # st.table(metrics_df)
            
            
          

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    import pandas as pd


    
    
    def evaluate_clustering(self, data, labels, algorithm_name):
    # Vérifiez que les labels contiennent au moins 2 clusters pour éviter des erreurs
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(data, labels)
            dbi = davies_bouldin_score(data, labels)
            calinski_harabasz = calinski_harabasz_score(data, labels)

            print(f"--- {algorithm_name} ---")
            print(f"Silhouette Score: {silhouette:.4f}")
            print(f"Davies-Bouldin Index: {dbi:.4f}")
            print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
            return silhouette, dbi, calinski_harabasz   
        else:
            print(f"--- {algorithm_name} ---")
            print("Clustering échoué : moins de 2 clusters détectés.")
            return None, None, None  
        
        
    def compare_algorithms(self, dbscan_labels, clarans_labels):
        """
        Compare les résultats de DBSCAN et CLARANS en termes de similarité des clusters.
        
        Args:
            dbscan_labels (array-like): Indices de clusters générés par DBSCAN.
            clarans_labels (array-like): Indices de clusters générés par CLARANS.
        
        Returns:
            pd.DataFrame: Tableau comparatif des métriques.
        """
        import numpy as np
        import pandas as pd
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        # Assurez-vous que dbscan_labels est un tableau 1D
        if isinstance(dbscan_labels, pd.Series):
            dbscan_labels = dbscan_labels.to_numpy()
        elif isinstance(dbscan_labels, np.ndarray) and dbscan_labels.ndim == 2:
            dbscan_labels = dbscan_labels.ravel()  # Utilisez ravel pour aplatir en 1D
        
        # Assurez-vous que clarans_labels est également un tableau 1D
        if isinstance(clarans_labels, pd.Series):
            clarans_labels = clarans_labels.to_numpy()
        elif isinstance(clarans_labels, np.ndarray) and clarans_labels.ndim == 2:
            clarans_labels = clarans_labels.ravel()
        
        # Debugging : Afficher les types et formes des labels
        print('----------------------------- DBSCAN Labels -----------------------------')
        dbscan_labels = np.array(dbscan_labels).flatten()
        print("Shape:", dbscan_labels.shape)
        print("Type:", type(dbscan_labels))
        
        print('----------------------------- CLARANS Labels -----------------------------')
        clarans_labels = np.array(clarans_labels).flatten()
        print("Shape:", clarans_labels.shape)
        print("Type:", type(clarans_labels))
        
        # Calcul des métriques de comparaison
        ari = adjusted_rand_score(clarans_labels, dbscan_labels)
        nmi = normalized_mutual_info_score(clarans_labels, dbscan_labels)
        
        # Construction du DataFrame
        comparison_df = pd.DataFrame({
            "Metric": ["Adjusted Rand Index (ARI)", "Normalized Mutual Info (NMI)"],
            "Value": [ari, nmi]
        })
        
        return comparison_df


    def visualize_clusters(self, data, custom_clusters, sklearn_clusters=None):
        print('je suis dans visualisation')

        st.write("### Visualisation des clusters")

        # Réduction dimensionnelle à 2D
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        print(data_2d)

        # Assurez-vous que custom_clusters est un tableau plat et que sa taille correspond à data_2d
        # if isinstance(custom_clusters, pd.DataFrame):
        #     custom_clusters = custom_clusters.to_numpy()
        # elif isinstance(custom_clusters, pd.Series):
        custom_clusters = custom_clusters.to_numpy()

        if len(custom_clusters) != len(data_2d):
            raise ValueError("'custom_clusters' doit avoir le même nombre d'éléments que les lignes de 'data_2d'.")

        medoid_indices = None
        if custom_clusters.ndim == 2:
            medoid_indices = custom_clusters[:, 1]  # Assigner les indices des medoids
            custom_clusters = custom_clusters[:, 0]  # Assigner les clusters

        # Création des subplots
        if sklearn_clusters is not None:
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            ax[0].grid(True)
            ax[1].grid(True)
            
        else:
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax = [ax]  # Convertir en liste pour garder une interface cohérente
            ax[0].grid(True)

        # Visualisation Custom DBSCAN/CLARANS
        unique_clusters = np.unique(custom_clusters)
        colors = plt.cm.tab10.colors
        legend_labels = []

        for i, cluster in enumerate(unique_clusters):
            cluster_points = data_2d[custom_clusters == cluster]
            ax[0].scatter(cluster_points[:, 0], cluster_points[:, 1],
                          c=[colors[i % len(colors)]], s=50, label=f"Cluster {cluster}")
            legend_labels.append(f"Cluster {cluster}")

        if medoid_indices is not None:
            ax[0].scatter(data_2d[medoid_indices.astype(int), 0],
                          data_2d[medoid_indices.astype(int), 1],
                          c='red', marker='o', s=100, label='Medoids')
            legend_labels.append("Medoids")

        ax[0].set_title("Custom Clustering")
        ax[0].set_xlabel(f"{data.columns[0]} (PCA Component 1)")
        ax[0].set_ylabel(f"{data.columns[1]} (PCA Component 2)")
        ax[0].legend(legend_labels)

        # Visualisation Sklearn DBSCAN (si disponible)
        if sklearn_clusters is not None:
            if isinstance(sklearn_clusters, pd.DataFrame):
                sklearn_clusters = sklearn_clusters.to_numpy()
            elif isinstance(sklearn_clusters, pd.Series):
                sklearn_clusters = sklearn_clusters.to_numpy()

            if len(sklearn_clusters) != len(data_2d):
                raise ValueError("'sklearn_clusters' doit avoir le même nombre d'éléments que les lignes de 'data_2d'.")

            unique_sklearn_clusters = np.unique(sklearn_clusters)
            for i, cluster in enumerate(unique_sklearn_clusters):
                cluster_points = data_2d[sklearn_clusters == cluster]
                ax[1].scatter(cluster_points[:, 0], cluster_points[:, 1],
                              c=[colors[i % len(colors)]], s=50, label=f"Cluster {cluster}")

            ax[1].set_title("Sklearn Clustering")
            ax[1].set_xlabel(f"{data.columns[0]} (PCA Component 1)")
            ax[1].set_ylabel(f"{data.columns[1]} (PCA Component 2)")
            ax[1].legend()

        fig.suptitle("Visualisation des Clusters", fontsize=16)
        st.pyplot(fig)

    def visualis_distribution(self, y_test, custom_model_predictions, sklearn_model_predictions ,name_algo, target=st.session_state["target"]):
        import seaborn as sns  # Import de seaborn si ce n'est pas déjà fait
        # print(f"----------------------------------------\n costume_model{costume_model_predictions.shape}")
        # print(f"----------------------------------------\n sklearn model{sklearn_model_predictions.shape}")
        # print(f"----------------------------------------\n y_test{y_test.shape}")
        
        # target = ['Fall_Qair_mean', 'Spring_Qair_mean', 'Summer_Qair_mean', 'Winter_Qair_mean']
        # Vérification de la forme des prédictions
        if sklearn_model_predictions.ndim == 1:
            sklearn_model_predictions = sklearn_model_predictions.reshape(-1, 1)  # Convertir en tableau 2D (171, 1)

        num_targets = y_test.shape[1]
        num_cols = 2
        num_rows = (num_targets + 1) // num_cols

        plt.figure(figsize=(10, 4 * num_rows))
        for i, col in enumerate(target[:num_targets]):  # Limiter à `num_targets` si `target` est plus grand
            plt.subplot(num_rows, num_cols, i + 1)
            plt.grid()
            sns.kdeplot(y_test[:, i] - custom_model_predictions[:, i], color='blue', label=f'Custom {name_algo} - {col}')
            sns.kdeplot(y_test[:, i] - sklearn_model_predictions[:, i], color='green', linestyle='--', label=f'Sklearn {name_algo} - {col}')
            plt.title(f"{col}")
            plt.legend()
            plt.xlabel("Erreur")
            plt.ylabel("Fréquence")

        plt.tight_layout()
        plt.suptitle(f"Comparaison des erreurs entre les modèles (Custom {name_algo} vs Sklearn {name_algo})", fontsize=16, y=1.02)
        st.pyplot(plt)  # Affichage dans Streamlit
        

    # def compute_clustering_metrics(self, X_train, custom_labels, sklearn_labels=None):
    #     st.write("### Évaluation des Algorithmes de Clustering")

    #     # Adjusted Rand Index (ARI)
    #     if sklearn_labels is not None:
    #         ari = adjusted_rand_score(custom_labels, sklearn_labels)
    #         st.write(f"Adjusted Rand Index (ARI) entre Custom et Sklearn: {ari:.4f}")

    #     # Silhouette Score
    #     silhouette_custom = silhouette_score(X_train, custom_labels) if len(set(custom_labels)) > 1 else "Non applicable"
    #     st.write(f"Silhouette Score - Custom: {silhouette_custom}")

    #     if sklearn_labels is not None:
    #         silhouette_sklearn = silhouette_score(X_train, sklearn_labels) if len(set(sklearn_labels)) > 1 else "Non applicable"
    #         st.write(f"Silhouette Score - Sklearn: {silhouette_sklearn}")

    #     # Homogeneity, Completeness et V-measure
    #     if sklearn_labels is not None:
    #         homogeneity = homogeneity_score(custom_labels, sklearn_labels)
    #         completeness = completeness_score(custom_labels, sklearn_labels)
    #         v_measure = v_measure_score(custom_labels, sklearn_labels)
    #         st.write(f"Homogeneity: {homogeneity:.4f}")
    #         st.write(f"Completeness: {completeness:.4f}")
    #         st.write(f"V-measure: {v_measure:.4f}")

    #     # Davies-Bouldin Index
    #     db_custom = davies_bouldin_score(X_train, custom_labels) if len(set(custom_labels)) > 1 else "Non applicable"
    #     st.write(f"Davies-Bouldin Index - Custom: {db_custom}")

    #     if sklearn_labels is not None:
    #         db_sklearn = davies_bouldin_score(X_train, sklearn_labels) if len(set(sklearn_labels)) > 1 else "Non applicable"
    #         st.write(f"Davies-Bouldin Index - Sklearn: {db_sklearn}")

    #     # Nombre de clusters
    #     n_clusters_custom = len(set(custom_labels)) - (1 if -1 in custom_labels else 0)
    #     st.write(f"Nombre de clusters détectés - Custom: {n_clusters_custom}")

    #     if sklearn_labels is not None:
    #         n_clusters_sklearn = len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0)
    #         st.write(f"Nombre de clusters détectés - Sklearn: {n_clusters_sklearn}")

    #     # Proportion de bruit
    #     noise_custom = np.sum(np.array(custom_labels) == -1) / len(custom_labels)
    #     st.write(f"Proportion de bruit - Custom: {noise_custom:.2%}")

    #     if sklearn_labels is not None:
    #         noise_sklearn = np.sum(np.array(sklearn_labels) == -1) / len(sklearn_labels)
    #         st.write(f"Proportion de bruit - Sklearn: {noise_sklearn:.2%}")


    # def compute_clustering_metrics1(self, X_train, custom_labels, sklearn_labels=None):
    #     st.write("### Évaluation des Algorithmes de Clustering")

    #     # Adjusted Rand Index (ARI)
    #     if sklearn_labels is not None:
    #         ari = adjusted_rand_score(custom_labels, sklearn_labels)
    #         st.write(f"Adjusted Rand Index (ARI) entre Custom et Sklearn: {ari:.4f}")

    #     # Silhouette Score
    #     silhouette_custom = silhouette_score(X_train, custom_labels) if len(set(custom_labels)) > 1 else "Non applicable"
    #     st.write(f"Silhouette Score - Custom: {silhouette_custom}")

    #     if sklearn_labels is not None:
    #         silhouette_sklearn = silhouette_score(X_train, sklearn_labels) if len(set(sklearn_labels)) > 1 else "Non applicable"
    #         st.write(f"Silhouette Score - Sklearn: {silhouette_sklearn}")

    #     # Homogeneity, Completeness et V-measure
    #     if sklearn_labels is not None:
    #         homogeneity = homogeneity_score(custom_labels, sklearn_labels)
    #         completeness = completeness_score(custom_labels, sklearn_labels)
    #         v_measure = v_measure_score(custom_labels, sklearn_labels)
    #         st.write(f"Homogeneity: {homogeneity:.4f}")
    #         st.write(f"Completeness: {completeness:.4f}")
    #         st.write(f"V-measure: {v_measure:.4f}")

    #     # Davies-Bouldin Index
    #     db_custom = davies_bouldin_score(X_train, custom_labels) if len(set(custom_labels)) > 1 else "Non applicable"
    #     st.write(f"Davies-Bouldin Index - Custom: {db_custom}")

    #     if sklearn_labels is not None:
    #         db_sklearn = davies_bouldin_score(X_train, sklearn_labels) if len(set(sklearn_labels)) > 1 else "Non applicable"
    #         st.write(f"Davies-Bouldin Index - Sklearn: {db_sklearn}")

    #     # Nombre de clusters
    #     n_clusters_custom = len(set(custom_labels)) - (1 if -1 in custom_labels else 0)
    #     st.write(f"Nombre de clusters détectés - Custom: {n_clusters_custom}")

    #     if sklearn_labels is not None:
    #         n_clusters_sklearn = len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0)
    #         st.write(f"Nombre de clusters détectés - Sklearn: {n_clusters_sklearn}")

    #     # Proportion de bruit
    #     noise_custom = np.sum(np.array(custom_labels) == -1) / len(custom_labels)
    #     st.write(f"Proportion de bruit - Custom: {noise_custom:.2%}")

    #     if sklearn_labels is not None:
    #         noise_sklearn = np.sum(np.array(sklearn_labels) == -1) / len(sklearn_labels)
    #         st.write(f"Proportion de bruit - Sklearn: {noise_sklearn:.2%}")





    def display_clustering_results(self, algo_name):
        st.subheader(f"Résultats du clustering : {algo_name}")

        if algo_name == "DBSCAN" and "dbscan_results" in st.session_state:
            st.dataframe(st.session_state['dbscan_results'])

        elif algo_name == "CLARANS" and "clarans_results" in st.session_state:
            st.dataframe(st.session_state['clarans_results'])

        else:
            st.warning("Aucun résultat disponible pour cet algorithme.")


    # Fonction pour convertir les résultats CLARANS en DataFrame
    def convert_clarans_to_dataframe(self, X_train, clarans_clusters):
        """
        Convertir la sortie de CLARANS en DataFrame avec les clusters associés.
        
        Args:
        - X_train (numpy.ndarray): Les données originales.
        - clarans_clusters (dict): Les résultats de CLARANS (médoïdes et indices des clusters).
        
        Returns:
        - pd.DataFrame: DataFrame contenant les caractéristiques et les clusters associés.
        """
        # Créer une liste pour stocker les informations sur les clusters
        cluster_assignment = [-1] * len(X_train)
        medoid_assignment = [-1] * len(X_train)
        
        for cluster_idx, (medoid, indices) in enumerate(clarans_clusters.items()):
            for idx in indices:
                cluster_assignment[idx] = cluster_idx
                medoid_assignment[idx] = medoid

        # Convertir X_train en DataFrame
        df = pd.DataFrame(X_train, columns=[st.session_state["features"][i] for i in range(X_train.shape[1])])
        df["Cluster"] = cluster_assignment
        df["Medoid"] = medoid_assignment

        return df

    @staticmethod
    def display_metrics(y_true, y_pred, training_time , sklearn=None , target = st.session_state["target"]):
        mae = mean_absolute_error(y_true, y_pred ,multioutput='uniform_average')
        rmse = np.sqrt(mean_squared_error(y_true, y_pred , multioutput='uniform_average'))
        r2 = r2_score(y_true, y_pred , multioutput='uniform_average')

        st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        st.write(f"R-squared (R2): {r2:.4f}")
        st.write(f"Training Time: {training_time:.4f} seconds")

        # Visualisation des prédictions
        # fig, ax = plt.subplots()
        # comparaison valeur réel et valeur predit 
        # Vérification de la forme des prédictions
        if sklearn and y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)  # Convertir en tableau 2D (171, 1)

        
        num_targets = y_true.shape[1]
        fig, axes = plt.subplots(num_targets, 1, figsize=(10, 5 * num_targets))
        if num_targets == 1:
            axes = [axes]  # Pour s'assurer que `axes` soit iterable

        for i in range(num_targets):
            axes[i].plot(y_true[:, i], label="Valeurs Réelles")
            axes[i].plot(y_pred[:, i], label="Prédictions")
            axes[i].set_title(target[i])
            axes[i].legend()
            axes[i].set_xlabel("Échantillons")
            axes[i].set_ylabel("Valeurs")
        
        st.pyplot(fig)
        
    def run(self):
        
        st.title("Application d'Analyse de Modèles")

        tab1, tab2 = st.tabs(["Régression", "Clustering"])

        # Page Régression
        with tab1:
            st.header("Analyse pour les Modèles de Régression")
            # Logique et visualisations spécifiques à la régression
            self.run_regression()

        # Page Clustering
        with tab2:
            st.header("Analyse pour les modèles de Clustering")
            # Logique et visualisations spécifiques au clustering
            self.run_clustering()
                
            

if __name__ == "__main__":
    app = App()
    app.run()

   
   
"""_summary_
    DBSCAN
    eps :
        Le rayon maximal d'un voisinage autour d'un point (appelé epsilon-neighborhood).
        Les points dans ce rayon sont considérés comme voisins. Une bonne valeur dépend de la distribution des données et peut être trouvée en analysant la courbe des distances (méthode k-distance plot).
        
    min_samples :
        Le nombre minimum de points requis dans un voisinage pour qu'un point soit considéré comme un core point (point central d'un cluster).
        Une valeur typique est entre 5 et 10. Plus elle est petite, plus il y a de chances de former de petits clusters.

    CLARANS
    numlocal :
        Le nombre d'itérations pour la recherche aléatoire de médoïdes locaux.
        Une valeur courante est entre 10 et 20. Plus elle est élevée, plus le résultat peut être optimal, mais cela augmente aussi le temps de calcul.
    maxneighbor :
        Le nombre maximal de voisins à examiner pour déplacer un médoïde.
        Une valeur typique est entre 5 et 10 pour équilibrer précision et vitesse.
    stop_threshold :
        Le seuil pour stopper la recherche si la différence de coût (fonction d'évaluation ou perte) entre deux itérations consécutives est inférieure à ce seuil.
        Il permet de contrôler la convergence de l'algorithme.



"""