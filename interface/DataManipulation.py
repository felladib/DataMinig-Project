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
import xarray as xr


class DataManipulation():
    def __init__(self):
        self.path = 'data/Country/Country'
        self.algeria_path = 'data/Algeria/algeria'
        self.soil_path = 'data/soil_dz_allprops.csv'
        self.dataset_info = {
                                'PSurf': 'data/Climate_data/csv_filtered_climat_data/PSurf_WFDE5_CRU_2019*_v2.csv',
                                'Qair': 'data/Climate_data/csv_filtered_climat_data/Qair_WFDE5_CRU_2019*_v2.csv',
                                'Rainf': 'data/Climate_data/csv_filtered_climat_data/Rainf_WFDE5_CRU_2019*_v2.csv',
                                'Snowf': 'data/Climate_data/csv_filtered_climat_data/Snowf_WFDE5_CRU_2019*_v2.csv',
                                'Tair': 'data/Climate_data/csv_filtered_climat_data/Tair_WFDE5_CRU_2019*_v2.csv',
                                'Wind': 'data/Climate_data/csv_filtered_climat_data/Wind_WFDE5_CRU_2019*_v2.csv'
                            }
        
        
    
    def data_filtering(self, nc_path, shp_path, country):
   
        

        # Load the Algeria shapefile and get Algeria geometry
        gdf = gpd.read_file(f'{shp_path}.shp', encoding="utf-8")
        gdf_country = gdf[gdf['CNTRY_NAME'] == country]
        country_geometry = gdf_country.geometry

        # Ensure NetCDF data has a CRS (if missing, set it to WGS84)
        ds = ds.rio.write_crs("EPSG:4326")

        # Clip the NetCDF data to Algeria’s geometry
        ds_country = ds.rio.clip(country_geometry, ds.rio.crs, drop=True)
        return ds_country
    
    
    def get_season(self,date):
        if date >= pd.Timestamp(year=date.year, month=12, day=22) or date < pd.Timestamp(year=date.year, month=3, day=22):
            return 'Winter'
        elif date >= pd.Timestamp(year=date.year, month=3, day=22) and date < pd.Timestamp(year=date.year, month=6, day=22):
            return 'Spring'
        elif date >= pd.Timestamp(year=date.year, month=6, day=22) and date < pd.Timestamp(year=date.year, month=9, day=22):
            return 'Summer'
        else:
            return 'Fall'
        
    
        
    def seasonal_aggregation(self,file_pattern, variable_name):
    # Load all monthly files into a single DataFrame
        all_months = pd.concat([pd.read_csv(f) for f in glob.glob(file_pattern)], ignore_index=True)
        
        # Convert 'time' to datetime (originally 'object' converted to 'datetime64[ns]')
        all_months['time'] = pd.to_datetime(all_months['time'])
        
        # Assign season name based on date
        all_months['season'] = all_months['time'].apply(self.get_season)
        
        # Group by latitude, longitude, season, and calculate min, max, mean
        seasonal_data = all_months.groupby(['lat', 'lon', 'season']).agg(
            min_value=(variable_name, 'min'),
            max_value=(variable_name, 'max'),
            mean_value=(variable_name, 'mean')
        ).reset_index()
        
        return seasonal_data
    
    
    def merge_climate_soil(self,climate_df, soil_gdf):
       
        
        if not isinstance(soil_gdf['geometry'].iloc[0], Point) and isinstance(soil_gdf['geometry'].iloc[0], str):
            soil_gdf['geometry'] = soil_gdf['geometry'].apply(wkt.loads)
        

        soil_gdf = gpd.GeoDataFrame(soil_gdf, geometry='geometry', crs="EPSG:4326")
       
        
        # Create GeoDataFrame for the climate dataset
        climate_gdf = gpd.GeoDataFrame(
            climate_df,
            geometry=[Point(lon, lat) for lon, lat in zip(climate_df['lon'], climate_df['lat'])],
            crs="EPSG:4326"  # Assuming WGS84 coordinate system
        )
        
        # Perform spatial join
       
        
        merged_gdf = gpd.sjoin(climate_gdf, soil_gdf, how='left', predicate='within')
        
        return merged_gdf
       
       
       
        
    '''
        Function to merge the seasonal datasets into one single dataset, and keeping only min - max - mean values

        input : path to the files = Climate_data/Seasonal_filtered_climate_data/*.csv
        output : saves the dataset into a csv file
    '''
    def merge_climate_datasets():
        # Path to the saved seasonal CSV files
        file_paths = glob.glob('Climate_data/Seasonal_filtered_climate_data/*.csv')

        # Initialize an empty list to store each dataset
        dataframes = []

        # Read each file and add it to the list of dataframes
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            variable_name = file_path.split("\\")[-1].split('.')[0] # Extract variable name from filename
            df = df.rename(columns={'min_value': f'{variable_name}_min',
                                    'max_value': f'{variable_name}_max',
                                    'mean_value': f'{variable_name}_mean'})
            dataframes.append(df)

        # Merge all dataframes on common columns
        merged_df = dataframes[0]
        for df in dataframes[1:]:
            merged_df = pd.merge(merged_df, df, on=['lat', 'lon', 'season'], how='outer')

        # Save the merged dataset to a new CSV file
        # merged_df.to_csv('Climate_data/Combined_Seasonal_Climate_Data.csv', index=False)


       
       
       
        
    '''
        Function to pivot the dataset to make seasons as columns and save the new dataset into a csv file
    '''
    def pivot_climate_seasons(self):
        combined_seasoned_climat_df = pd.read_csv('Climate_data/Combined_Seasonal_Climate_Data.csv')
        pivoted_df = combined_seasoned_climat_df.pivot_table(
            index=['lat', 'lon'],
            columns='season',
            values=[col for col in combined_seasoned_climat_df.columns if col not in ['lat', 'lon', 'season']]
        )

        # Flatten the multi-index columns
        pivoted_df.columns = [f"{season}_{metric}" for metric, season in pivoted_df.columns]

        # Reset the index if needed
        pivoted_df = pivoted_df.reset_index()

        # Save the transformed dataframe
        pivoted_df.to_csv('Combined_Climate_Seasonal_columns.csv', index=False)

    # pivot_climate_seasons()