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


class DataVisualizer:
    def __init__(self):
        self.algeria_path = 'data/Algeria/algeria'
        self.soil_path = 'data/soil_dz_allprops.csv'
        


    def plot_scatter(self, data, column1, column2):
            # Créer un scatter plot avec Plotly
            fig = px.scatter(
                data,
                x=column1,
                y=column2,
                title=f"Scatter plot of {column1} vs {column2}",
                labels={column1: column1, column2: column2},  # Étiquettes des axes
                template="plotly_dark"  # Thème sombre
            )

            # Ajouter une annotation pour la corrélation
            correlation = data[[column1, column2]].corr().iloc[0, 1]
            fig.add_annotation(
                text=f"Correlation: {correlation:.2f}",
                xref="paper", yref="paper",
                x=1, y=1.1, showarrow=False,
                font=dict(size=15, color="#FF4B4B")
            )

            # Personnalisation des couleurs selon votre thème
            fig.update_traces(
                marker=dict(color="#FF4B4B")  # Couleur des points
            )

            fig.update_layout(
                title=dict(
                    font=dict(color="#FAFAFA"),
                    x=0.45,  # Titre aligné au centre
                    xanchor="center",
                ),
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                font=dict(color="#FAFAFA"),
                xaxis=dict(gridcolor="#444444"),
                yaxis=dict(gridcolor="#444444")
            )

            return fig
    def plot_boxplot(self, data, columns, show_outliers=True):
        # Créer une figure Plotly vide
        fig = go.Figure()

        colors_plate2 = ["#FF69B4", "#FFB6C1", "#D3D3D3", "#FFD700", "#003366"]  # Avant Q2
        colors_plate1 = ["#FFB6C1", "#D3D3D3", "#FFD700", "#003366", "#FF69B4"]   # Après Q2
        colors_plate4 = ["#FF6347", "#708090", "#8FBC8F", "#F5DEB3", "#87CEEB"]
        colors_plate5 = ["#B22222", "#D8BFD8", "#FFDB58", "#808000", "#4682B4"]
        colors_plate3 = ["#FFDAB9", "#D2B48C", "#E6E6FA", "#2F4F4F", "#A9A9A9"]
        
        
        
        # Ajouter une trace pour chaque colonne
        for i, column in enumerate(columns):
            # Calcul des statistiques nécessaires pour les bornes
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Ajouter la boîte
            fig.add_trace(go.Box(
                y=data[column],
                name=column,
                boxpoints="outliers" if show_outliers else False,  # Afficher ou non les outliers
                marker_color=colors_plate1[i % len(colors_plate1)],  # Avant Q2
                fillcolor=colors_plate1[i % len(colors_plate1)],  # Après Q2
                line=dict(color="white"),  # Couleur de la ligne de la médiane
                whiskerwidth=1  # Largeur des moustaches
            ))

            # Ajouter des annotations pour les bornes
            fig.add_annotation(
                x=column, y=upper_bound,
                text=f"Upper: {upper_bound:.2f}",
                showarrow=False,
                font=dict(color="white"),
                xanchor="center", yanchor="bottom"
            )
            fig.add_annotation(
                x=column, y=lower_bound,
                text=f"Lower: {lower_bound:.2f}",
                showarrow=False,
                font=dict(color="white"),
                xanchor="center", yanchor="top"
            )

        # Mise à jour du layout pour correspondre au thème
        fig.update_layout(
            title=dict(text="Box Plots with Bounds",
                       x=0.45, 
                       xanchor="center", 
                       font=dict(color="#FAFAFA")),   
            yaxis_title="Values",
            # xaxis_title="Columns",
            template="plotly_dark",  # Base sombre
            paper_bgcolor="#0E1117",  # Fond de la page
            plot_bgcolor="#0E1117",  # Fond de la figure
            font=dict(color="#FAFAFA"),  # Couleur du texte
            xaxis=dict(gridcolor="#444444"),
            yaxis=dict(gridcolor="#444444"),
        )
       
        return fig


    def plot_soil_map(self,soil_df, attribute):
        # Convertir les chaînes WKT en objets de géométrie
        soil_df['geometry'] = soil_df['geometry'].apply(wkt.loads)
        # Convertir en GeoDataFrame
        gdf = gpd.GeoDataFrame(soil_df, geometry='geometry', crs="EPSG:4326")

        # Convertir les données géographiques en GeoJSON
        gdf = gdf.set_geometry("geometry")
        geojson = gdf.__geo_interface__

        # Créer la carte avec Plotly Express
        fig = px.choropleth_mapbox(
            gdf,
            geojson=geojson,
            locations=gdf.index,  # Utilisation des index pour relier les données
            color=attribute,
            color_continuous_scale=["#262730", "#FF4B4B"],  # Palette dégradée selon le thème
            mapbox_style="carto-darkmatter",  # Style de carte sombre
            center={"lat": 28, "lon": 3},  # Centrage approximatif sur l'Algérie
            zoom=3.69,  # Zoom initial
            title=f"{attribute} across Algeria"
        )

        # Personnaliser l'apparence
        fig.update_layout(
            title=dict(
                text=f"{attribute} across Algeria",
                font=dict(size=15, color="#FAFAFA"),  # Couleur du texte
                x=0.45,  # Aligner à droite
                xanchor="center",  # Aligner au centre
            ),
            font=dict(size=10, color="#FAFAFA"),  # Couleur du texte général
            legend=dict(
                orientation="v",
                title_font=dict(size=12, color="#FAFAFA"),
                font=dict(size=10, color="#FAFAFA"),
            ),
            coloraxis_colorbar=dict(
                title=attribute,
                title_side="right",  # Titre de l'échelle à droite
                tickfont=dict(size=9, color="#FAFAFA"),  # Taille des ticks de l'échelle
            ),
            margin={"r": 0, "t": 40, "l": 0, "b": 0},  # Marges ajustées
            paper_bgcolor="#0E1117",  # Fond principal (arrière-plan)
            plot_bgcolor="#0E1117",  # Fond secondaire (carte)
        )

        # Retourner la figure pour affichage dans Streamlit
        return fig

    
    ''' 
    Funtion to plot the climate map 
    Input : A dataframe + The name of the attribute
    Output : A map plot 
    '''
    def plot_climate_map(self,Combined_Climate_Seasonal_columns_df, attribute):

        df_data = Combined_Climate_Seasonal_columns_df[['lon', 'lat', attribute]]
        pivoted_data = df_data.pivot_table(index="lat", columns="lon", values=attribute, aggfunc="mean")

        # Define the geographic extent of the heatmap (min and max longitude and latitude)
        extent = [df_data['lon'].min(), df_data['lon'].max(), df_data['lat'].min(), df_data['lat'].max()]

        # Plot the heatmap using imshow with the defined extent
        plt.figure()
        ax = plt.gca()
        im = ax.imshow(pivoted_data, cmap="coolwarm", extent=extent, origin="lower", aspect="auto")
        plt.colorbar(im, label=attribute)

        gdf_algerie = gpd.read_file(f'{self.algeria_path}.shp', encoding="utf-8")
        gdf_algerie.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=1)

        # Add titles and labels
        plt.title(f"{attribute} Heatmap")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        plt.show()
       





    def plot_climate_map_plotly(self, Combined_Climate_Seasonal_columns_df, attribute):
        # Préparer les données climatiques
        df_data = Combined_Climate_Seasonal_columns_df[['lon', 'lat', attribute]]
        pivoted_data = df_data.pivot_table(index="lat", columns="lon", values=attribute, aggfunc="mean")
        lons, lats = np.meshgrid(pivoted_data.columns, pivoted_data.index)
        heatmap_values = pivoted_data.values

        # Charger le shapefile de l'Algérie
        gdf_algerie = gpd.read_file(f'{self.algeria_path}.shp', encoding="utf-8")
        geojson = gdf_algerie.__geo_interface__  # Convertir le shapefile en GeoJSON

        # Créer la Heatmap avec Plotly
        fig = go.Figure()

        # Ajouter la Heatmap (valeurs climatiques)
        fig.add_trace(go.Heatmap(
            x=lons[0, :],  # Longitudes
            y=lats[:, 0],  # Latitudes
            z=heatmap_values,  # Valeurs climatiques
            colorscale="Reds",  # Palette de couleurs (dégradé rouge)
            colorbar=dict(
                title=dict(text=attribute, font=dict(size=12, color="#FAFAFA")),
                tickcolor="#FAFAFA",
                tickfont=dict(size=10, color="#FAFAFA"),
                len=0.8  # Taille verticale de la barre de couleur
            ),
            showscale=True,  # Afficher la légende de la heatmap
        ))

        # Ajouter les contours de l'Algérie (au-dessus de la heatmap)
        fig.add_trace(go.Scattermapbox(
            lon=gdf_algerie.geometry.bounds.mean(axis=1).values,  # Longitude moyenne de chaque région
            lat=gdf_algerie.geometry.bounds.mean(axis=0).values,  # Latitude moyenne de chaque région
            mode="lines",
            line=dict(width=1, color="black"),  # Bordure noire
            showlegend=False,  # Pas de légende pour le contour
        ))

        # Personnaliser le layout
        fig.update_layout(
            title=dict(
                text=f"{attribute} Heatmap",
                font=dict(size=15, color="#FAFAFA"),
                x=0.45,  # Alignement à droite
                xanchor="center",
            ),
            mapbox=dict(
                style="carto-darkmatter",  # Style sombre
                center={"lat": 28, "lon": 3},  # Centrage approximatif sur l'Algérie
                zoom=3.69,  # Zoom initial
            ),
            margin={"r": 0, "t": 40, "l": 0, "b": 0},  # Ajustement des marges
            paper_bgcolor="#0E1117",  # Fond principal
            plot_bgcolor="#0E1117",  # Fond de la carte
        )

        # Supprimer les axes X et Y
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        return fig




    def plot_algeria_map_plotly(self, path):
        # Charger le shapefile
        print(path)
        gdf_shp = gpd.read_file(f'{path}', encoding='utf-8')
        # country_name = path.split("/")[-1]  # Extraire le nom du fichier (pays)
        country_name = path.split("\\")[-1].split(".")[0]
        print(country_name.lower())
        # Convertir les données géographiques en GeoJSON
        geojson = gdf_shp.__geo_interface__

        # Créer un dégradé avec rgba
        color_scale = ['rgba(255, 75, 75, 0.3)', 'rgba(255, 75, 75, 1)']  # Dégradé du rouge clair au rouge foncé

        # Configuration spécifique pour l'Algérie ou le reste du monde
        if "algeria" in country_name.lower():
            center = {"lat": 28, "lon": 3}  # Centrage sur l'Algérie
            zoom = 3.69
            title = "Map of Algeria"
        else:
            center = {"lat": 20, "lon": 0}  # Centrage global
            zoom = 1.2
            title = f"Map of {country_name}"

        # Créer la carte avec Plotly
        fig = px.choropleth_mapbox(
            gdf_shp,
            geojson=geojson,
            locations=gdf_shp.index,  # Index pour relier les données
            color_discrete_sequence=color_scale,  # Couleurs
            mapbox_style="carto-darkmatter",  # Style sombre
            center=center,  # Centrage défini dynamiquement
            zoom=zoom,  # Zoom défini dynamiquement
            title=title,  # Titre dynamique
        )

        # Personnaliser l'apparence
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=15, color="#FAFAFA"),  # Couleur et taille du titre
                x=0.45,  # Titre aligné à droite
                xanchor="center",
            ),
            legend=dict(
                orientation="v",
                title=country_name,  # Titre de la légende dynamique
                title_font=dict(size=12, color="#FAFAFA"),
                font=dict(size=10, color="#FAFAFA"),
            ),
            font=dict(size=10, color="#FAFAFA"),  # Couleur du texte
            margin={"r": 0, "t": 40, "l": 0, "b": 0},  # Marges ajustées
            paper_bgcolor="#0E1117",  # Fond principal
            plot_bgcolor="#0E1117",  # Fond secondaire
        )

        return fig





    ''' 
    Funtion to plot a country's map
    Input : The path to the shape file
    Output : A map plot 
    '''
    # def plot_algeria_map_plotly1(self,path):
        #     # Charger le shapefile
        #     gdf_shp = gpd.read_file(f'{path}', encoding='utf-8')
        #     country_name = path.split("/")[-1].split(".")[0]  # Extraire le nom du fichier (pays)

        #     # Convertir les données géographiques en GeoJSON
        #     geojson = gdf_shp.__geo_interface__

        #     # Créer un dégradé avec rgba
        #     color_scale = ['rgba(255, 75, 75, 0.3)', 'rgba(255, 75, 75, 1)']  # Dégradé du rouge clair au rouge foncé

        #     # Créer la carte avec Plotly
        #     fig = px.choropleth_mapbox(
        #         gdf_shp,
        #         geojson=geojson,
        #         locations=gdf_shp.index,  # Index pour relier les données
        #         color_discrete_sequence=color_scale ,#["#5CFB95"],   Couleur des zones
        #         mapbox_style="carto-darkmatter",  # Style sombre
        #         center={"lat": 28, "lon": 3},  # Centrage approximatif sur l'Algérie
        #         zoom=3.69,  # Zoom initial
        #         title=f"Map of {country_name}"  # Titre
        #     )

        #     # Personnaliser l'apparence
        #     fig.update_layout(
        #         title=dict(
        #             text=f"Map of {country_name}",
        #             font=dict(size=15, color="#FAFAFA"),  # Couleur et taille du titre
        #             x=0.45,  # Titre aligné à droite
        #             xanchor="center",
        #         ),
        #         legend=dict(
        #             orientation="v",
        #             title="Algeria",  # Titre de la légende
        #             title_font=dict(size=12, color="#FAFAFA"),
        #             font=dict(size=10, color="#FAFAFA"),
        #         ),
        #         font=dict(size=10, color="#FAFAFA"),  # Couleur du texte
        #         margin={"r": 0, "t": 40, "l": 0, "b": 0},  # Marges ajustées
        #         paper_bgcolor="#0E1117",  # Fond principal
        #         plot_bgcolor="#0E1117",  # Fond secondaire
        #     )

        #     return fig


    
    
    
    def plot_climate_map_plotly(self, Combined_Climate_Seasonal_columns_df, attribute):
        # Préparer les données climatiques
        df_data = Combined_Climate_Seasonal_columns_df[['lon', 'lat', attribute]]

        # Charger le shapefile de l'Algérie et vérifier la projection
        gdf_algerie = gpd.read_file(f'{self.algeria_path}.shp', encoding="utf-8")
        if gdf_algerie.crs != "EPSG:4326":  # Vérifier si le CRS est en WGS 84 (latitude/longitude)
            gdf_algerie = gdf_algerie.to_crs("EPSG:4326")

        # Convertir le shapefile en GeoJSON
        geojson = gdf_algerie.__geo_interface__

        # Créer la carte avec Scattermapbox
        fig = go.Figure()

        # Ajouter les points climatiques
        fig.add_trace(go.Scattermapbox(
            lon=df_data['lon'],  # Longitudes
            lat=df_data['lat'],  # Latitudes
            mode="markers",  # Mode pour afficher uniquement des points
            marker=dict(
                size=7,  # Taille des points
                color=df_data[attribute],  # Couleur basée sur l'attribut
                colorscale="Viridis",  # Palette de couleurs
                cmin=df_data[attribute].quantile(0.05),  # Limite inférieure des couleurs
                cmax=df_data[attribute].quantile(0.95),  # Limite supérieure des couleurs
                opacity=0.8,  # Transparence des points
                colorbar=dict(
                    title=dict(text=attribute, font=dict(size=12, color="#FAFAFA")),
                    tickcolor="#FAFAFA",
                    tickfont=dict(size=10, color="#FAFAFA"),
                    len=0.8
                )
            )
        ))

        # Ajouter les contours de l'Algérie
        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson,
            locations=[0] * len(gdf_algerie),  # Données fictives pour afficher le contour
            z=[0] * len(gdf_algerie),  # Données fictives pour la couleur
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],  # Transparence totale
            marker_line_width=2,
            marker_line_color="gray",  # Bordures gris clair
            showscale=False  # Pas d'échelle pour les contours
        ))

        # Personnaliser le layout
        fig.update_layout(
            title=dict(
                text=f"{attribute} Map",
                font=dict(size=16, color="#FAFAFA"),
                x=0.5,
                xanchor="center",
            ),
            mapbox=dict(
                style="carto-darkmatter",  # Style sombre
                center={"lat": 28, "lon": 3},  # Centrage sur l'Algérie
                zoom=3.7,  # Zoom ajusté
            ),
            margin={"r": 10, "t": 40, "l": 10, "b": 10},  # Ajustement des marges
            paper_bgcolor="#0E1117",  # Fond principal
        )

        return fig


