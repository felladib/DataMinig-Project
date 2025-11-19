import os
import glob

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from shapely import wkt
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler

from dbfread import DBF
from pyproj import CRS
import geopandas as gpd
from pathlib import Path



class DataAnalyzer:
    def calculate_central_tendency(self, data, column):
        """
        Calcule les statistiques de tendance centrale et la symétrie pour une colonne donnée.
        """
        mean = data[column].mean()
        median = data[column].median()
        mode = data[column].mode()

        if len(mode) > 1:  # Vérifie s'il y a plusieurs modes
            symmetry = "Asymétrie (multimodal)"
        else:
            mode_value = mode.iloc[0]  # Prend la première valeur du mode
            symmetry = "Symétrie" if np.isclose(mean, median) and np.isclose(mean, mode_value) else "Skewed"

        skewness = skew(data[column].dropna())  # Calcul de l'asymétrie

        return {
            "mean": mean,
            "median": median,
            # "mode": mode.to_list(),
            "mode":mode.iloc[0],
            "symmetry": symmetry,
            "skewness": skewness
        }
        
    def calculate_dispersion(self,data, column , inhandling_columns):

        std_dev = data[column].std() 
        variance = data[column].var()
        
        Q0 = data[column].quantile(0)
        Q1 = data[column].quantile(0.25)
        Q2 = data[column].quantile(0.50)
        Q3 = data[column].quantile(0.75)
        Q4 = data[column].quantile(1)

        iqr = Q3 - Q1
        lower_bound = Q1 - 1.5 * iqr
        upper_bound = Q3 + 1.5 * iqr
        if column not in inhandling_columns:
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        else : 
            outliers = []
        return {
            "std_dev": std_dev,
            "variance": variance,
            "min": Q0,
            "Q1": Q1,
            "median": Q2,
            "Q3": Q3,
            "max": Q4,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outliers": outliers
        }
        
    def list_data(self , data_path='data'): #recuperer la liste des dataset disponible
        # Extensions exclues
        excluded_extensions = ['.prj', '.dbf', '.shx','json']
        
        # Chemin vers le dossier data
        datasets = []

        # Parcourir le dossier et ses sous-dossiers
        for root, dirs, files in os.walk(data_path):
            for file in files:
                # Vérifier si le fichier n'a pas une des extensions exclues
                if not any(file.endswith(ext) for ext in excluded_extensions):
                    datasets.append(os.path.join(root, file))  # Chemin complet
            # self.list_dataset = datasets  # Mise à jour de l'attribut
        datasets.append("Combined_Climate_Soil_data")
        return datasets

    def load_data(self, data_path):
        data = None
        extension = Path(data_path).suffix
        try:
            if extension == ".csv":
                return pd.read_csv(data_path, sep=',')
            elif extension == ".shp":
                return gpd.read_file(data_path, encoding='utf-8')
            elif extension == ".shx":
                return gpd.read_file(data_path, encoding='utf-8')
            elif extension == ".dbf":
                table = DBF(data_path, encoding='utf-8')
                return pd.DataFrame(iter(table))
            elif extension == ".prj":
                with open(data_path, 'r') as f:
                    prj_text = f.read()
                return CRS.from_wkt(prj_text)
        except Exception as e:
            print(f"Erreur lors du chargement des données : {e}")
        return None
    
    def save_data(self, data, data_path):
        if not os.path.exists(data_path): 
            os.makedirs(data_path)
            
        extension = Path(data_path).suffix
        try:
            if extension == ".csv":
                data.to_csv(data_path, index=False)
                print("-----------------------------saved data----------")
            elif extension == ".shp":
                data.to_file(data_path, driver='ESRI Shapefile', encoding='utf-8')
            elif extension == ".json":
                data.to_json(data_path, orient='records', force_ascii=False)
            else:
                raise ValueError("Format de fichier non pris en charge pour la sauvegarde.")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des données : {e}")
            
