import os
import glob
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from shapely import wkt
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from dbfread import DBF
from pyproj import CRS
import geopandas as gpd
from pathlib import Path


class DataProcessor:
    def __init__(self):
        pass
    
    def normalize_data(self, data, norm_attributes, normalisation):
        if normalisation == "Z-score":
            scaler = MinMaxScaler()
            data[norm_attributes] = scaler.fit_transform(data[norm_attributes])
        elif normalisation == "Min-Max":  
            scaler = MinMaxScaler()     
            data[norm_attributes] = scaler.fit_transform(data[norm_attributes])
        return data
    
    
    ''' 
        Function to discretize data with one of 2 methodes (equal-amplitude or equal-frequency)
        Input : - A Dataframe
                - The attribute name
                - the discretization method name
        Output : The dataframe with discretized data
    '''
    def discretize_data(self,data, column, method='equal-frequency'):
        
        n = len(data[column])
        K = int(1 + (10 / 3) * np.log10(n))
        print(f"The number of intervals = {K}\n")

        if method == 'equal-amplitude':
            # Equal-amplitude (equal-width) binning
            min_value = data[column].min()
            max_value = data[column].max()
            width = (max_value - min_value) / K
            bins = [min_value + i * width for i in range(K + 1)]
            print(f"The interval's bins (equal-amplitude) are: {bins}\n")

            # Calculate labels as mean values for each bin
            labels = []
            for i in range(K):
                bin_data = data[(data[column] >= bins[i]) & (data[column] < bins[i + 1])]
                mean_value = bin_data[column].mean() if not bin_data.empty else (bins[i] + bins[i + 1]) / 2
                labels.append(mean_value)
            print(f"The interval's labels (equal-amplitude) are: {labels}\n")

            d = pd.cut(data[column], bins=bins, labels=labels)
            data[column] = d
            
        elif method == 'equal-frequency':
            # Equal-frequency binning
            d = pd.qcut(data[column], q=K, labels=False, duplicates='drop')
            
            # Calculate mean for each equal-frequency bin
            labels = []
            for bin_number in range(K):
                bin_data = data[d == bin_number]
                mean_value = bin_data[column].mean()
                labels.append(mean_value)
            print(f"The interval's labels (equal-frequency) are: {labels}\n")
            
            # Reassign the discretized column with mean labels
            d = d.map(lambda x: labels[int(x)] if pd.notna(x) else None)
            data[column] = d
            

        # data[column] = d

        return data
    
    
    def eliminate_redundancies(self, data, method='Vertical', columns=None):
        """
        Élimine les redondances dans les données.

        Parameters:
            data (DataFrame): Le DataFrame contenant les données.
            method (str): Méthode pour éliminer les redondances ('horizontal' ou 'vertical').
            columns (list): Liste des colonnes à vérifier pour redondance (utilisé pour 'vertical').

        Returns:
            DataFrame: Le DataFrame après suppression des redondances.
        """
        if method == 'Horizontal':
            # Suppression des lignes dupliquées
            data = data.drop_duplicates()
        elif method == 'Vertical':
            data = data.drop(columns=columns)
        else:
            raise ValueError("Méthode non reconnue : choisissez 'horizontal' ou 'vertical'")
        return data


    def handle_missing_data(self, data, columns, method='Delete'):
        """
        Handle missing values in a dataset for specified columns.

        Parameters:
            data (DataFrame): Le DataFrame contenant les données.
            columns (list or str): Une ou plusieurs colonnes à traiter.
            method (str): La méthode de traitement des valeurs manquantes ('Delete', 'mean', 'median', 'mode').

        Returns:
            DataFrame: Le DataFrame avec les valeurs manquantes gérées.
        """
        # if isinstance(columns, str):
        #     columns = [columns]  # Convertir en liste si une seule colonne est donnée
        data = data.replace({None: np.nan})
        if method == 'Delete':
            # Supprimer les lignes où les colonnes spécifiées contiennent des NaN
            data = data.dropna(subset=columns)
        elif method == 'Mean':
            for col in columns:
                mean = data[col].mean()
                data[col] = data[col].fillna(mean)
        elif method == 'Median':
            for col in columns:
                median = data[col].median()
                data[col] = data[col].fillna(median)
        elif method == 'Mode':
            for col in columns:
                mode = data[col].mode()[0]
                data[col] = data[col].fillna(mode)  # Utilise le mode le plus fréquent
        else:
            raise ValueError(f"Méthode '{method}' non reconnue. Utilisez 'Delete', 'mean', 'median' ou 'mode'.")

        return data

    
    def calcule_outliers(self,data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        iqr = Q3 - Q1
        lower_bound = Q1 - 1.5 * iqr
        upper_bound = Q3 + 1.5 * iqr
        
        return lower_bound, upper_bound
    
    def handle_outliers(self,data, columns, outlier_method='Delete'):
        """
        Handle outliers in the dataset using the specified method.

        Parameters:
            data (DataFrame): Le DataFrame contenant les données.
            columns (list): Liste des colonnes à traiter.
            outlier_method (str): Méthode pour gérer les valeurs aberrantes ('IQR', 'mean', 'median', 'mode').

        Returns:
            DataFrame: Le DataFrame avec les valeurs aberrantes traitées.
        """
        for column in columns:
            if outlier_method == 'Delete':
                # Détecter les bornes des valeurs aberrantes via l'IQR
                
                # Q1 = data[column].quantile(0.25)
                # Q3 = data[column].quantile(0.75)
                # iqr = Q3 - Q1
                # lower_bound = Q1 - 1.5 * iqr
                # upper_bound = Q3 + 1.5 * iqr
                lower_bound, upper_bound = self.calcule_outliers(data[column])
                # Filtrer les lignes dans les bornes
                data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
            elif outlier_method == 'Mean':
                # Remplacer les valeurs aberrantes par la moyenne
                # Q1 = data[column].quantile(0.25)
                # Q3 = data[column].quantile(0.75)
                # iqr = Q3 - Q1
                # lower_bound = Q1 - 1.5 * iqr
                # upper_bound = Q3 + 1.5 * iqr
                lower_bound, upper_bound = self.calcule_outliers(data[column])
                
                mean_value = data[column].mean()
                data[column] = data[column].apply(
                    lambda x: mean_value if (x < lower_bound or x > upper_bound) else x
                )
            elif outlier_method == 'Median':
                # Remplacer les valeurs aberrantes par la médiane
                # Q1 = data[column].quantile(0.25)
                # Q3 = data[column].quantile(0.75)
                # iqr = Q3 - Q1
                # lower_bound = Q1 - 1.5 * iqr
                # upper_bound = Q3 + 1.5 * iqr
                lower_bound, upper_bound = self.calcule_outliers(data[column])
                
                median_value = data[column].median()
                data[column] = data[column].apply(
                    lambda x: median_value if (x < lower_bound or x > upper_bound) else x
                )
            elif outlier_method == 'Mode':
                # Remplacer les valeurs aberrantes par la mode
                # Q1 = data[column].quantile(0.25)
                # Q3 = data[column].quantile(0.75)
                # iqr = Q3 - Q1
                # lower_bound = Q1 - 1.5 * iqr
                # upper_bound = Q3 + 1.5 * iqr
                lower_bound, upper_bound = self.calcule_outliers(data[column])
                
                mode_value = data[column].mode()[0]
                data[column] = data[column].apply(
                    lambda x: mode_value if (x < lower_bound or x > upper_bound) else x
                )
            else:
                raise ValueError(f"Méthode {outlier_method} non reconnue.")
        return data