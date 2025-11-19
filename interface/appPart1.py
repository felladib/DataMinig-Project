import streamlit as st
import pandas as pd
import os
import base64
import json
import time 
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from scipy.stats import gaussian_kde
import numpy as np
from sklearn.neighbors import KernelDensity
import plotly.graph_objects as go
import plotly.graph_objects as go
from sklearn.neighbors import KernelDensity
from DataAnalyzer import DataAnalyzer
from DataVisualizer import DataVisualizer
import plotly.figure_factory as ff
from DataProcessor import DataProcessor
from DataManipulation import DataManipulation
from shapely.geometry import Polygon, MultiPolygon
from interface.Models.RF import RandomForestRegressorr , DecisionTreeRegressor 

# Configurer la page
st.set_page_config(page_title="Climat App", page_icon="🌍", layout="wide")



# Initialisation de l'état dans session_state
if 'choices_validated' not in st.session_state:
    st.session_state['choices_validated'] = False
if 'selected_data' not in st.session_state:
    st.session_state['selected_data'] = []
if 'selected_attributes' not in st.session_state:
    st.session_state['selected_attributes'] = []
if 'processing' not in st.session_state:
    st.session_state['processing'] = {}
if "outliers_handled" not in st.session_state:
        st.session_state["outliers_handled"] = {}  # Créer un dictionnaire pour chaque attribut


class ClimatApp:
    def __init__(self):
        self.plot_type = None
        self.preprocess_options = {}
        self.list_dataset = {}
        self.dataset_choice = None
        self.data_analyzer = DataAnalyzer()  # Instance de la classe DataAnalyzer
        self.Data_visualizer= DataVisualizer()
        self.Data_preprocessing = DataProcessor()
        self.Data_manipulation = DataManipulation()
        self.plot_attributes = {}
        self.climat_path = 'data/Combined_Climate_Seasonal_columns.csv'
        self.soil_path = 'data/soil_dz_allprops.csv'
        

    def get_image_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def show_welcome_image(self, placeholder):
        # Affiche l'image si les choix ne sont pas validés
        if not st.session_state['choices_validated']:
            welcome_image_path = os.path.join("assets", "XfQ8.gif")
            if os.path.exists(welcome_image_path):
                placeholder.markdown(
                    f"""
                    <div style="display: flex; flex-direction: column; align-items: center; padding-top: 100px;">
                        <img src="data:image/gif;base64,{self.get_image_base64(welcome_image_path)}" 
                            alt="Bienvenue ! Configurez vos paramètres pour commencer." width="300"/>
                        <p style="text-align: center; padding-top: 20px; font-size: 18px;">
                            Bienvenue ! Configurez vos paramètres pour commencer.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("L'image de bienvenue n'a pas été trouvée.")
        else:
            placeholder.empty()
        
    def configure_page(self):
        st.sidebar.title("Configuration")

    def select_dataset(self):
        if not self.list_dataset:
            self.list_dataset=self.data_analyzer.list_data()  # Met à jour la liste si elle est vide

        st.sidebar.subheader("Datasets disponibles")
        self.dataset_choice = st.sidebar.selectbox("Choisir un dataset", self.list_dataset)

        # Charger les données sélectionnées
        try:
            if self.dataset_choice == "Combined_Climate_Soil_data":
                # print("dans la daaaaaaaaaaaataaa new")
                try:
                    dataClimat = pd.read_csv(self.climat_path)
                    data_soil = pd.read_csv(self.soil_path)
                    data = self.Data_manipulation.merge_climate_soil(dataClimat, data_soil)
                except Exception as e:
                    st.error(f"Erreur lors du chargement des donnée combiné : {e}")
            else:
                data = self.data_analyzer.load_data(self.dataset_choice)
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
        # Stocker les données dans l'état de session Streamlit
        if data is not None:  # Vérifie si le chargement a réussi
            st.session_state['selected_data'] = data
        else:
            st.warning("Impossible de charger le dataset sélectionné.")
            st.session_state['selected_data'] = None

        return data


    def highlight_missing_and_outliers(self, data):
        """
        Applique un style pour colorier les valeurs manquantes (NaN) et les outliers,
        tout en excluant certaines colonnes (géométriques, non numériques, etc.).
        """
        # Vérifie si une colonne contient des données géométriques
        def is_polygon_column(column):
            return not column.empty and isinstance(column.dropna().iloc[0], (Polygon, MultiPolygon))

        # Liste des colonnes à exclure
        columns_to_exclude = ['lat', 'lon', 'spatial_ref', 'time']

        # Vérifie si le fichier est un shapefile
        is_shapefile = hasattr(self, 'dataset_choice') and '.shp' in self.dataset_choice

        # Identifie les colonnes numériques valides
        valid_columns = [
            col for col in data.columns
            if pd.api.types.is_numeric_dtype(data[col])
            and col not in columns_to_exclude
            and not is_polygon_column(data[col])
            and col not in st.session_state["outliers_handled"]
            
        ]

        def colorize_column(column):
            if column.name not in valid_columns:
                return [''] * len(column)  # Pas de style pour les colonnes invalides

            # Calculer Q1, Q3 et IQR pour les colonnes numériques valides
            Q1 = column.quantile(0.25)
            Q3 = column.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            # Appliquer les couleurs sur les valeurs de la colonne
            def colorize(val):
                if pd.isna(val):  # Valeurs manquantes
                    return 'background-color: rgba(255, 255, 0, 0.5)'
                elif val < lower or val > upper:  # Outliers
                    return 'background-color: #3c1c21 ;' #rgba(255, 75, 75, 0.1)
                return ''  # Pas de couleur

            return column.apply(colorize)

        # Appliquer le style par colonne
        styled_data = data.style.apply(colorize_column, axis=0)
        return styled_data

                     
    def show_dataset_overview(self):
        
        with open("data/description.json", "r", encoding="utf-8") as file:
            descriptions = json.load(file)

        # Vérification de la structure JSON
        description_section = descriptions.get("description", {})

        
        # Vérifier si le chemin normalisé existe dans la section 'description'
        description = description_section.get(self.dataset_choice, "Description non disponible pour ce dataset.")
        
                
        # Afficher l'aperçu du dataset
        st.subheader("Aperçu du Dataset")
        # st.dataframe(st.session_state['selected_data'])  # Afficher le dataframe
        
        try:
            if st.session_state['selected_data'] is not None:
                styled_data = self.highlight_missing_and_outliers(st.session_state['selected_data'])
                st.dataframe(styled_data)
        except Exception as e:
            st.warning(f"Impossible d'appliquer le style ")
            st.dataframe(st.session_state['selected_data'])
        # Créer des colonnes pour centrer le bouton
        col1, col2, col3 = st.columns([1, 1, 1])  # La colonne du milieu est deux fois plus large

        # Placer le bouton au centre
        with col2:
            if st.button("💾 Sauvegarder les modifications"):
                # Exemple : Logique de sauvegarde ici
                st.session_state['selected_data'].to_csv("data_updated.csv", index=False)
                st.success("Les modifications ont été sauvegardées avec succès.")

        # Afficher la description en dessous
        st.markdown(f"  \n {description}")
        
        
    def show_attributes(self):
        st.sidebar.subheader("Attributs disponibles")
        if st.session_state['selected_data'] is not None:
            # Si le dataset est chargé, afficher les options
            st.session_state['selected_attributes'] = st.sidebar.multiselect(
                "Sélectionnez les attributs", 
                options=st.session_state['selected_data'].columns, 
                default=st.session_state['selected_data'].columns
            )
        else:
            st.warning("Aucun dataset n'est chargé. Veuillez sélectionner un dataset dans la barre latérale.")

        st.session_state['selected_dataset'] = st.session_state['selected_data'][st.session_state['selected_attributes']]
        return st.session_state['selected_attributes']


    


    def validate_button(self):
        # Ajouter du CSS pour centrer le bouton dans la barre latérale
        button_html = """
        <style>
        [data-testid="stSidebar"] button {
            margin: 0 auto;
            display: block;
        }
        </style>
        """
        st.sidebar.markdown(button_html, unsafe_allow_html=True)

        # Bouton dans la barre latérale
        if st.sidebar.button("Valider"):
            st.session_state['choices_validated'] = True
            return True
        return False


    def select_plot_type2(self):
        
        def is_polygon_column(column):
            # Si les valeurs de la colonne sont de type géométrique, retourner True
            # Par exemple, si les valeurs sont des objets Shapely (Polygon, MultiPolygon, etc.)
            from shapely.geometry import Polygon, MultiPolygon
            return isinstance(column.dropna().iloc[0], (Polygon, MultiPolygon))

        st.sidebar.subheader("Visualisation")
        
        # Utiliser la logique qui vous permet de vérifier si le fichier est un shapefile
        is_shapefile = '.shp' in self.dataset_choice  
        
        # Exclure les colonnes spécifiques comme 'att', 'log', et les colonnes non numériques
        columns_to_exclude = ['lat', 'lon','spatial_ref','time']

        numeric_columns = [
            col for col in st.session_state['selected_attributes'] 
            if pd.api.types.is_numeric_dtype(st.session_state['selected_data'][col]) 
            and col not in columns_to_exclude
            and not is_polygon_column(st.session_state['selected_data'][col])
        ]
        # Ajouter une liste des types de visualisation disponibles
        available_plot_types = ["Map","Boxplot", "Scatter"]
        
        
        
        # Choisir le type de visualisation
        self.plot_type = st.sidebar.selectbox("Choisir le type de visualisation", available_plot_types)
        print(f"if shape{is_shapefile}")
        # Attributs pour chaque type de visualisation
        if self.plot_type == "Boxplot":
            if is_shapefile:
                
                # Désactiver le boxplot si le dataset est un shapefile
                st.sidebar.markdown("Boxplot désactivé pour les fichiers .shp")
                self.boxplot_attr = []  # Pas de sélection possible
                self.plot_attributes['boxplot'] = self.boxplot_attr
            else:
                
                # Permettre la sélection multiple (limite à 5 attributs numériques seulement)
                self.boxplot_attr = st.sidebar.multiselect(
                    "Choisir jusqu'à 5 attributs pour le boxplot",
                    numeric_columns,  # Liste des attributs numériques seulement
                    default=[],
                    max_selections=5
                )
                # Stocker les attributs sélectionnés dans plot_attributes
                self.plot_attributes['boxplot'] = self.boxplot_attr

        elif self.plot_type == "Scatter":
            if is_shapefile:
                # Désactiver le scatter si le dataset est un shapefile
                st.sidebar.markdown("Scatter désactivé pour les fichiers .shp")
                self.x_attr = self.y_attr = None  # Pas de sélection possible
                self.plot_attributes['scatter'] = {}
            else:
                
                
                # Permettre la sélection des attributs numériques pour le scatter plot
                self.x_attr = st.sidebar.selectbox("Choisir l'attribut pour l'axe X", numeric_columns)
                self.y_attr = st.sidebar.selectbox("Choisir l'attribut pour l'axe Y", numeric_columns)
                self.plot_attributes['scatter'] = {'x': self.x_attr, 'y': self.y_attr}

        elif self.plot_type == "Map":
            if is_shapefile:
               
                
                # Permettre uniquement le choix d'attributs géographiques pour le shapefile
                self.x_attr = None  # Pas de sélection d'attribut pour le shapefile
                self.plot_attributes['map'] = "Aucune colonne sélectionnée pour la carte"
            else:
               
                
                # Permettre la sélection des colonnes numériques pour la carte
                self.x_attr = st.sidebar.selectbox("Choisir l'attribut pour la carte", numeric_columns)
                self.plot_attributes['map'] = self.x_attr
                
        # return self.plot_type, self.plot_attributes
     
     
    def select_plot_type(self):
            
            def is_polygon_column(column):
                # Si les valeurs de la colonne sont de type géométrique, retourner True
                # Par exemple, si les valeurs sont des objets Shapely (Polygon, MultiPolygon, etc.)
                from shapely.geometry import Polygon, MultiPolygon
                return isinstance(column.dropna().iloc[0], (Polygon, MultiPolygon))

            st.sidebar.subheader("Visualisation")
            
            # Utiliser la logique qui vous permet de vérifier si le fichier est un shapefile
            is_shapefile = '.shp' in self.dataset_choice  
            
            # Exclure les colonnes spécifiques comme 'att', 'log', et les colonnes non numériques
            columns_to_exclude = ['lat', 'lon','spatial_ref','time']

            numeric_columns = [
                col for col in st.session_state['selected_attributes'] 
                if pd.api.types.is_numeric_dtype(st.session_state['selected_data'][col]) 
                and col not in columns_to_exclude
                and not is_polygon_column(st.session_state['selected_data'][col])
            ]
            # Ajouter une liste des types de visualisation disponibles
            available_plot_types = ["Map","Boxplot", "Scatter"]
            
            
            # Choisir le type de visualisation
            self.plot_type = st.sidebar.selectbox("Choisir le type de visualisation", available_plot_types)

            # Attributs pour chaque type de visualisation
            if self.plot_type == "Boxplot":
                if is_shapefile:
                    # Désactiver le boxplot si le dataset est un shapefile
                    st.sidebar.markdown("Boxplot désactivé pour les fichiers .shp")
                    self.boxplot_attr = []  # Pas de sélection possible
                    self.plot_attributes['boxplot'] = self.boxplot_attr
                else:
                    # Permettre la sélection multiple (limite à 5 attributs numériques seulement)
                    self.boxplot_attr = st.sidebar.multiselect(
                        "Choisir jusqu'à 5 attributs pour le boxplot",
                        numeric_columns,  # Liste des attributs numériques seulement
                        default=[],
                        max_selections=5
                    )
                    # Stocker les attributs sélectionnés dans plot_attributes
                    self.plot_attributes['boxplot'] = self.boxplot_attr

            elif self.plot_type == "Scatter":
                if is_shapefile:
                    # Désactiver le scatter si le dataset est un shapefile
                    st.sidebar.markdown("Scatter désactivé pour les fichiers .shp")
                    self.x_attr = self.y_attr = None  # Pas de sélection possible
                    self.plot_attributes['scatter'] = {}
                else:
                    # Permettre la sélection des attributs numériques pour le scatter plot
                    self.x_attr = st.sidebar.selectbox("Choisir l'attribut pour l'axe X", numeric_columns)
                    self.y_attr = st.sidebar.selectbox("Choisir l'attribut pour l'axe Y", numeric_columns)
                    self.plot_attributes['scatter'] = {'x': self.x_attr, 'y': self.y_attr}

            elif self.plot_type == "Map":
                if is_shapefile:
                    # Permettre uniquement le choix d'attributs géographiques pour le shapefile
                    self.x_attr = None  # Pas de sélection d'attribut pour le shapefile
                    self.plot_attributes['map'] = "Aucune colonne sélectionnée pour la carte"
                else:
                    # Permettre la sélection des colonnes numériques pour la carte
                    self.x_attr = st.sidebar.selectbox("Choisir l'attribut pour la carte", numeric_columns)
                    self.plot_attributes['map'] = self.x_attr
                    
            return self.plot_type, self.plot_attributes
        
     
     
     
    def show_column_information(self):
        st.subheader("Informations sur les Colonnes")
        for col in st.session_state['selected_data'].columns:
            with st.expander(f"Détails sur la colonne : {col}"):
                # Divise l'espace en deux colonnes : Infos à gauche, Histogramme à droite
                info_col, plot_col = st.columns([1, 2])
                
                
                # Afficher les informations de base pour chaque colonne dans la première colonne
                with info_col:
                    st.write(f"Type : {st.session_state['selected_data'][col].dtype}")
                    st.write(f"Valeurs uniques : {st.session_state['selected_data'][col].nunique()}")
                    st.write(f"Valeurs manquantes : {st.session_state['selected_data'][col].isna().sum()}")
                    
                    # Afficher les statistiques descriptives si la colonne est numérique
                    if pd.api.types.is_numeric_dtype(st.session_state['selected_data'][col]):
                        stats = self.data_analyzer.calculate_central_tendency(st.session_state['selected_data'], col)
                        dispersion = self.data_analyzer.calculate_dispersion(st.session_state['selected_data'], col,st.session_state["outliers_handled"])
                        
                        st.write(f"Moyenne : {stats['mean']:.5f}")
                        st.write(f"Mode : {stats['mode']:.5f}")
                        st.write(f"Min : {st.session_state['selected_data'][col].min():.5f}")
                        st.write(f"Q1 : {dispersion['Q1']:.5f}")  
                        st.write(f"Médiane : {stats['median']:.5f}") 
                        st.write(f"Q3 : {dispersion['Q3']:.5f}")                     
                        st.write(f"Max : {st.session_state['selected_data'][col].max():.5f}")
                        st.write(f"Écart-type : {dispersion['std_dev']:.5f}")
                        st.write(f"Variance : {dispersion['variance']:.5f}")
                        st.write(f"IQR : {dispersion['iqr']:.5f}")                     
                        st.write(f"lower_bound : {dispersion['lower_bound']:.5f}")
                        st.write(f"upper_bound : {dispersion['upper_bound']:.5f}")                     
                        st.write(f"Outliers: {len(dispersion['outliers'])}")                     
                        
                        st.write(f"Symétrie : {stats['symmetry']}")
                        st.write(f"Asymétrie : {stats['skewness']:.5f}")
                       


                        
                
                # with plot_col:
                #     if pd.api.types.is_numeric_dtype(st.session_state['selected_data'][col]):
                #         # Histogramme
                #         fig = go.Figure()
                #         hist_data = st.session_state['selected_data'][col].dropna()
                #         fig.add_trace(go.Histogram(
                #             x=hist_data,
                #             nbinsx=30,
                #             name="Histogramme",
                #             marker=dict(
                #                 color="skyblue",                 # Couleur de remplissage
                #                 line=dict(color="black", width=1)  # Bordure noire
                #             ),
                #             opacity=0.75,
                #         ))

                #         # Calculer la densité et ajouter la courbe avec gestion des erreurs
                #         try:
                #             selected_column = hist_data.values.reshape(-1, 1)
                #             kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(selected_column)

                #             # Générer des points pour la densité
                #             x_vals = np.linspace(hist_data.min(), hist_data.max(), 100).reshape(-1, 1)
                #             log_density = kde.score_samples(x_vals)
                #             density_vals = np.exp(log_density)

                #             # Ajouter la courbe de densité
                #             fig.add_trace(go.Scatter(
                #                 x=x_vals.flatten(),
                #                 y=density_vals * len(hist_data) * (x_vals[1] - x_vals[0]).item(),  # Mise à l'échelle
                #                 mode="lines",
                #                 name="Densité",
                #                 line=dict(color="orange", width=2)
                #             ))
                #         except Exception as e:
                #             st.warning(f"Impossible d'afficher la courbe de densité pour la colonne {col}. Erreur : {e}")

                #         # Mettre à jour la mise en page
                #         fig.update_layout(
                #             title=f"Distribution de {col}",
                #             xaxis_title=col,
                #             yaxis_title="Count",
                #             template="plotly_dark"
                #         )
                #         st.plotly_chart(fig, use_container_width=True)
               
                with plot_col:
                    # if pd.api.types.is_numeric_dtype(st.session_state['selected_data'][col]):
                    #     # Récupérer les données numériques de la colonne
                    #     hist_data = st.session_state['selected_data'][col].dropna()

                    #     # Vérifier si les données ne sont pas vides
                    #     if not hist_data.empty:
                    #         # Créer la distribution avec Plotly
                    #         fig = ff.create_distplot(
                    #             [hist_data],  # Les données doivent être une liste de listes
                    #             group_labels=[col],  # Nom pour la légende
                    #             bin_size=0.1  # Taille des bins
                    #         )

                    #         # Afficher le graphique avec Streamlit
                    #         st.plotly_chart(fig, use_container_width=True)
                    #     else:
                    #         st.warning(f"La colonne {col} ne contient pas de données valides pour créer une distribution.")

                    if pd.api.types.is_numeric_dtype(st.session_state['selected_data'][col]):
                        # Histogramme
                        fig = go.Figure()
                        hist_data = st.session_state['selected_data'][col].dropna()
                        fig.add_trace(go.Histogram(
                            x=hist_data,
                            nbinsx=30,
                            name="Histogramme",
                            marker=dict(
                                color="skyblue",                 # Couleur de remplissage
                                line=dict(color="black", width=1)  # Bordure noire
                            ),
                            opacity=0.75,
                        ))

                        # Calculer la densité KDE
                        try:
                            selected_column = hist_data.values.reshape(-1, 1) #KernelDensityattendez des données en 2 dimensions (format [n_samples, n_features]). La méthode reshapetransforme une colonne 1D en une matrice 2D avec une seule colonne

                            # Vérifier que les données ont suffisamment de points pour KDE
                            if len(selected_column) > 1:
                                kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(selected_column)

                                # Générer des points pour la densité
                                x_vals = np.linspace(hist_data.min(), hist_data.max(), 90).reshape(-1, 1)
                                log_density = kde.score_samples(x_vals) 
                                density_vals = np.exp(log_density)

                                # Ajouter la courbe de densité
                                fig.add_trace(go.Scatter(
                                    x=x_vals.flatten(),
                                    y=density_vals * len(hist_data) * (x_vals[1] - x_vals[0]).item(),  # Mise à l'échelle
                                    mode="lines",
                                    name="Densité",
                                    line=dict(color="orange", width=2)
                                ))
                            else:
                                st.warning(f"Pas assez de données pour calculer la densité KDE pour {col}.")
                        except Exception as e:
                            st.warning(f"Impossible d'afficher la courbe de densité pour la colonne {col}. Erreur : {e}")

                        # Mettre à jour la mise en page
                        fig.update_layout(
                            title=f"Distribution de {col}",
                            xaxis_title=col,
                            yaxis_title="Count",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
 
    def select_preprocessing(self):
        
        
        def is_polygon_column(column):
            # Si les valeurs de la colonne sont de type géométrique, retourner True
            # Par exemple, si les valeurs sont des objets Shapely (Polygon, MultiPolygon, etc.)
            from shapely.geometry import Polygon, MultiPolygon
            return isinstance(column.dropna().iloc[0], (Polygon, MultiPolygon))

       
        
        # Utiliser la logique qui vous permet de vérifier si le fichier est un shapefile
        is_shapefile = '.shp' in self.dataset_choice  
        
        # Exclure les colonnes spécifiques comme 'att', 'log', et les colonnes non numériques
        columns_to_exclude = ['lat', 'lon','spatial_ref','time']


        numeric_columns = [
            col for col in st.session_state['selected_attributes'] 
            if pd.api.types.is_numeric_dtype(st.session_state['selected_data'][col]) 
            and col not in columns_to_exclude
            and not is_polygon_column(st.session_state['selected_data'][col])
        ]
        options = ["All"] + numeric_columns
        if not is_shapefile :
            st.sidebar.subheader("Pré-traitement")
            
            # Handling Mising value
            
            Missing_val_methode = st.sidebar.selectbox("Choisir une méthode de traitement des valeurs manquantes", ["None", "Delete", "Mean", "Median", "Mode"])
            missing_attributes = st.sidebar.multiselect("Attributs avec des valeurs manquantes", options) if Missing_val_methode != "None" else []
            if missing_attributes :
                if "All" in missing_attributes:
                    missing_attributes = numeric_columns
                st.session_state['selected_data'] = self.Data_preprocessing.handle_missing_data(st.session_state['selected_data'] , missing_attributes,  Missing_val_methode)
            
            
            # outiers
            outliers_method = st.sidebar.selectbox("Choisir une méthode de traitement des valeurs aberrantes", ["None", "Delete", "Mean", "Median", "Mode"])
            outlier_attributes = st.sidebar.multiselect("Attributs avec des valeurs aberrantes", options ) if outliers_method != "None" else []
            # if outlier_attributes :
            #     st.session_state['selected_data'] = self.Data_preprocessing.handle_outliers(st.session_state['selected_data'] , outlier_attributes, outliers)
            if outliers_method != "None" and outlier_attributes:
                if "All" in outlier_attributes:
                    outlier_attributes = numeric_columns
                # Appliquer le traitement des outliers sur les attributs sélectionnés
                # unhandled_attributes = [attribute for attribute in outlier_attributes if attribute not in st.session_state["outliers_handled"]]
                
                st.session_state["selected_data"] = self.Data_preprocessing.handle_outliers(
                    st.session_state["selected_data"],  outlier_attributes  , outliers_method
                )
                

                # Marquer les attributs traités
                for attribute in outlier_attributes:
                    st.session_state["outliers_handled"][attribute] = True
                # print( st.session_state["outliers_handled"])
                    
                    
            # Réduction (vertical)
            reduction = st.sidebar.selectbox("Choisir une méthode de réduction", ["None","Horizontal", "Vertical"])
            red_attributes = st.sidebar.multiselect("Attributs à réduire", options) if reduction != "None" else []
            if red_attributes :
                if "All" in outlier_attributes:
                    outlier_attributes = numeric_columns
                st.session_state['selected_data'] = self.Data_preprocessing.eliminate_redundancies(data = st.session_state['selected_data'] , method=reduction , columns=red_attributes)
            
            # Normalisation
            normalisation = st.sidebar.selectbox("Choisir une méthode de normalisation", ["None","Z-score", "Min-Max"])
            norm_attributes = st.sidebar.multiselect("Attributs à normaliser", numeric_columns) if normalisation != "None" else []
            if norm_attributes :
                st.session_state['selected_data'] = self.Data_preprocessing.normalize_data(st.session_state['selected_data'], norm_attributes, normalisation)
            
            # Discrétisation
            discretisation = st.sidebar.selectbox("Choisir une méthode de discrétisation", ["None", "Equal Frequency", "Amplitude"])
            disc_attribute = st.sidebar.selectbox("Attributs à discrétiser", numeric_columns) if discretisation != "None" else []
            if disc_attribute :
                st.session_state['selected_data'] = self.Data_preprocessing.discretize_data(st.session_state['selected_data'], disc_attribute, discretisation)

        
           
    def visualizer(self):
        # st.write("Visualisation des données")
        
        if self.plot_type and  self.plot_attributes:
            if st.session_state['selected_data'] is not None and self.plot_type is not None :
                if self.plot_type == "Map":
                    
                    # si la dataset soil:
                    if self.dataset_choice == "data\soil_dz_allprops.csv":
                        fig = self.Data_visualizer.plot_soil_map(
                            st.session_state['selected_data'], 
                            self.plot_attributes['map'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif os.path.normpath(self.dataset_choice.strip()) == os.path.normpath("data/Algeria/algeria.shp") or os.path.normpath(self.dataset_choice.strip()) == os.path.normpath("data/Country/Country.shp"):
                        # print("-------------------------------------------------")
                        fig = self.Data_visualizer.plot_algeria_map_plotly(self.dataset_choice)
                        # st.pyplot(fig)
                        st.plotly_chart(fig, use_container_width=True)
                           
                    else:
                        fig = self.Data_visualizer.plot_climate_map_plotly(
                            st.session_state['selected_data'], 
                            self.plot_attributes['map']
                        )
                        st.plotly_chart(fig, use_container_width=True)

                elif  self.plot_type == "Scatter":
                    # st.write(self.plot_attributes['scatter'])
                    fig = self.Data_visualizer.plot_scatter(
                            st.session_state['selected_data'], 
                            self.plot_attributes['scatter']['x'],
                            self.plot_attributes['scatter']['y']
                        )
                    # st.pyplot(fig)
                    st.plotly_chart(fig, use_container_width=True)
                elif  self.plot_type == "Boxplot" and len(self.plot_attributes['boxplot']) > 0 :
                    # dans le cas de box plot 
                    
                    fig = self.Data_visualizer.plot_boxplot(
                            st.session_state['selected_data'], 
                            columns=self.plot_attributes['boxplot'],
                            show_outliers=True
                        )
                    # st.pyplot(fig)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Veuillez sélectionner au moins un attribut pour le boxplot.")
                
   
    # def advanced_analysis(self):
    #     """Interface pour l'analyse avancée."""
    #     st.subheader("Advanced Data Mining Options")

    #     # Choix de la méthode de Data Mining
    #     method = st.selectbox("Select the Data Mining method:", 
    #                         ["Custom Random Forest", "Decision Tree", "DBSCAN", "CLARANS"])

    #     # Checkbox pour indiquer si l'instance est dans le dataset
    #     instance_from_dataset = st.checkbox("Use an instance from data_test")

    #     # Charger les données
    #     try:
    #         self.X_test = pd.read_csv("data/test_data/X_test.csv")
    #         self.Y_test = pd.read_csv("data/test_data/Y_test.csv")
    #     except FileNotFoundError:
    #         st.error("Test data files not found. Ensure 'X_test.csv' and 'Y_test.csv' are in 'data/test_data/'.")
    #         return

    #     # Charger le modèle en fonction de la méthode sélectionnée
    #     model_files = {
    #         "Custom Random Forest": "data/models/random_forest_model.pkl",
    #         "Decision Tree": "decisionTree_model.pkl",
    #         "DBSCAN": "dbscan_model.pkl",
    #         "CLARANS": "clarans_model.pkl"
    #     }

    #     sklearn_model_files = {
    #         "Custom Random Forest": "data/models/random_forest_model_sklearn.pkl",
    #         "Decision Tree": "decisionTree_model_sklearn.pkl",
    #         "DBSCAN": "dbscan_model_sklearn.pkl",
    #         "CLARANS": "clarans_model_sklearn.pkl"
    #     }
    #     method = "Custom Random Forest"
    #     try:
    #         self.model_sklearn = joblib.load(sklearn_model_files[method])
    #         self.model = joblib.load(model_files[method])
    #     except FileNotFoundError:
    #         st.error(f"Model file for {method} not found.")
    #         return

    #     # Prédire et calculer les métriques (Custom Random Forest uniquement)
    #     if method == "Custom Random Forest":
    #         predictions = self.model.predict(self.X_test)
    #         predictions_sklearn = self.model_sklearn.predict(self.X_test)

    #         mae = mean_absolute_error(self.Y_test, predictions)
    #         rmse = np.sqrt(mean_squared_error(self.Y_test, predictions))
    #         r2 = r2_score(self.Y_test, predictions)

    #         mae_sklearn = mean_absolute_error(self.Y_test, predictions_sklearn)
    #         rmse_sklearn = np.sqrt(mean_squared_error(self.Y_test, predictions_sklearn))
    #         r2_sklearn = r2_score(self.Y_test, predictions_sklearn)

    #         test_time_custom = 0  # Exemple (ajouter le calcul du temps si besoin)
    #         test_time_sklearn = 0  # Exemple

    #     # Sélection ou saisie de l'instance
    #     if instance_from_dataset:
    #         instance_index = st.number_input("Select instance index (from data_test):", min_value=0, max_value=len(self.X_test)-1, step=1)
    #         instance = self.X_test.iloc[instance_index].values  # Récupération de l'instance
    #         st.write("Selected instance:", instance)
    #     else:
    #         instance_input = st.text_input("Enter a new data instance (comma-separated values):")
    #         if instance_input:
    #             try:
    #                 instance = np.array(list(map(float, instance_input.split(',')))).reshape(1, -1)
    #                 st.write("Input instance:", instance)
    #             except ValueError:
    #                 st.error("Invalid input. Please enter comma-separated numeric values.")
    #                 return

    #     # Bouton pour faire la prédiction
    #     if st.button("Predict"):
    #         try:
    #             y_pred = self.model.predict(instance)
    #             st.write(f"Predicted values: {y_pred}")

    #             if instance_from_dataset:
    #                 y_true = self.Y_test.iloc[instance_index]
    #                 st.write(f"True values: {y_true.values}")
    #         except Exception as e:
    #             st.error(f"Prediction failed: {e}")

    #     # Afficher les métriques
    #     if method == "Custom Random Forest":
    #         st.write("### Performance Metrics")
    #         st.write("#### Custom Model")
    #         st.metric("RMSE", f"{rmse:.4f}")
    #         st.metric("MAE", f"{mae:.4f}")
    #         st.metric("Execution Time (Test)", f"{test_time_custom:.2f}s")

    #         st.write("#### Sklearn Model")
    #         st.metric("RMSE", f"{rmse_sklearn:.4f}")
    #         st.metric("MAE", f"{mae_sklearn:.4f}")
    #         st.metric("Execution Time (Test)", f"{test_time_sklearn:.2f}s")

    #     # Hyperparamètres pour Custom Random Forest
    #     if method == "Custom Random Forest":
    #         st.write("### Hyperparameter Optimization")
    #         n_trees = st.slider("Number of Trees:", min_value=10, max_value=200, step=10, value=100)
    #         max_depth = st.slider("Maximum Depth:", min_value=1, max_value=50, step=1, value=10)
    #         st.write(f"Selected parameters: n_trees={n_trees}, max_depth={max_depth}")

    #         if st.button("Retrain with New Parameters"):
    #             start_time = time.time()
    #             self.model.train(n_trees=n_trees, max_depth=max_depth)  # Assurez-vous que la méthode train existe
    #             st.write(f"Model retrained in {time.time() - start_time:.2f}s")


                                    
    def run(self):
        # Configuration de la page latérale et de l'image de bienvenue
        self.configure_page()
        welcome_placeholder = st.empty()
        self.show_welcome_image(welcome_placeholder)
        
        # Choix du dataset et validation des choix
        self.select_dataset()
        self.show_attributes()
        self.select_plot_type()
        self.select_preprocessing()
        
        # Bouton de validation pour confirmer les choix
        if self.validate_button():
            welcome_placeholder.empty()
        
        # Affichage des données si validé
        if st.session_state['choices_validated']:
            # Affiche les onglets pour "Data" et "Columns"
            tabs = st.tabs(["Data", "Columns", "Visualisation", "Advanced Analysis"])
        
            with tabs[0]:  # Onglet "Data"
                self.show_dataset_overview()  # Affiche l'aperçu du dataset
            with tabs[1]:  # Onglet "Columns"
                self.show_column_information()
            with tabs[2]:  # Onglet "Visualisation"
                self.visualizer()
            with tabs[3]:  # Onglet "Advanced Analysis"
                pass
                # self.advanced_analysis()



if __name__ == "__main__":
    app = ClimatApp()
    app.run()


