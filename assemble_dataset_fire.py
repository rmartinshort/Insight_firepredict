#!/usr/bin/env python

# Functions for assembling fire dataset

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point

import Dataset_manipulation as DM


def assemble_fires_dataframe(datapath,year_to_predict,SF_blocks,date_to_start='2007-01-01'):

    '''
    Input: Path to fires data, year (or other timeframe) to predict, Dataframe of SF blocks
    Output: Dataframes for training and holdout
    '''

    fire_incidents = pd.read_csv(datapath+'Fire_Incidents.csv',low_memory=False)

    fire_incidents['Incident_Date'] = pd.to_datetime(fire_incidents['Incident Date'])
    fire_incidents.sort_values(by='Incident_Date',inplace=True)

    #Get only the data since 2007 (when we have building information)
    fire_incidents = fire_incidents[fire_incidents['Incident_Date']>=date_to_start]

    #Remove rows that don't have location information
    fires_dataset_cleaned = fire_incidents.dropna(subset=['Location'])

    #### NOTE: May want to edit this
    #Keep only the fire instances that we care about
    #These numbers refer to codes in the Primary Situation column of the fires database
    structure_fire = ['111','112']
    vehicle_fire = ['130','131','132','137','138']
    #maybe need to split these further?
    external_fire = ['140','141','142','143','151','152','153','154','155','160',\
            '161','162','164','170','173']
    ####

    searchfor = structure_fire + vehicle_fire + external_fire
    fires_dataset_cleaned = fires_dataset_cleaned[fires_dataset_cleaned['Primary Situation']\
                                    .str.contains('|'.join(searchfor))]

    #Add columns for the various fire types
    fires_dataset_cleaned['Structure_fire'] = fires_dataset_cleaned['Primary Situation']\
              .str.contains('|'.join(structure_fire))
    fires_dataset_cleaned['Vehicle_fire'] = fires_dataset_cleaned['Primary Situation']\
              .str.contains('|'.join(vehicle_fire))
    fires_dataset_cleaned['External_fire'] = fires_dataset_cleaned['Primary Situation']\
              .str.contains('|'.join(external_fire))

    fires_dataset_cleaned['geometry'] = fires_dataset_cleaned['Location'].apply(DM.convert_to_point)

    #Convert to geodataframe ready to associate fires with census blocks

    fires_dataset_geo = gpd.GeoDataFrame(fires_dataset_cleaned,geometry='geometry')
    fires_dataset_geo.crs = {'init': 'epsg:4326'}

    fires_dataset_geo['Structure_fire'] = fires_dataset_geo['Structure_fire'].astype(int)
    fires_dataset_geo['Vehicle_fire'] = fires_dataset_geo['Vehicle_fire'].astype(int)
    fires_dataset_geo['External_fire'] = fires_dataset_geo['External_fire'].astype(int)
    fires_dataset_geo['Incident_Year'] = fires_dataset_geo['Incident_Date'].apply(lambda x: int(x.year))

    if year_to_predict:
        training = fires_dataset_geo[fires_dataset_geo['Incident_Year']<year_to_predict]
        holdout = fires_dataset_geo[fires_dataset_geo['Incident_Year']==year_to_predict]

        # Merge fires - find the block that contains each fire
        intersections_tr = gpd.sjoin(SF_blocks, training, how="left", op='contains')
        intersections_tr.replace(np.nan,0,inplace=True)

        # Merge fires - find the block that contains each fire
        intersections_ho = gpd.sjoin(SF_blocks, holdout, how="left", op='contains')
        intersections_ho.replace(np.nan,0,inplace=True)

        Fires_per_block_train = intersections_tr[['GISJOIN','Structure_fire','Vehicle_fire','External_fire']].groupby('GISJOIN').sum()
        Fires_per_block_holdout = intersections_ho[['GISJOIN','Structure_fire','Vehicle_fire','External_fire']].groupby('GISJOIN').sum()

    else:
        intersections_tr = gpd.sjoin(SF_blocks, fires_dataset_geo, how="left", op='contains')
        intersections_tr.replace(np.nan,0,inplace=True)
        Fires_per_block_train = training[['GISJOIN','Structure_fire','Vehicle_fire','External_fire']].groupby('GISJOIN').sum()
        Fires_per_block_holdout = None

    return Fires_per_block_train, Fires_per_block_holdout

if __name__ == "__main__":

    census_path = '../datasets/census_blocks/'

    SF_blocks = gpd.read_file(census_path+'SF_block_2010.shp')

    train, holdout = assemble_fires_dataframe('../datasets/fire/',2018,SF_blocks)

    #print(holdout.head())
    print(len(holdout))
    print(len(train))
