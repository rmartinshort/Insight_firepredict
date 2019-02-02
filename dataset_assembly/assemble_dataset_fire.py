#!/usr/bin/env python

# Functions for assembling fire dataset

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point

import Dataset_manipulation as DM

def assemble_fires_dataframe_oneperyear(datapath,SF_blocks,date_to_start='2007-01-01'):

    '''
    This generates a dataframe where each row represents census cell in one year
    The fires are NOT offset. Use the next function to do that
    '''

    fire_incidents = pd.read_csv(datapath+'Fire_Incidents.csv',low_memory=False)
    fire_incidents['Incident_Date'] = pd.to_datetime(fire_incidents['Incident Date'])
    fire_incidents.sort_values(by='Incident_Date',inplace=True)

    #Get only the data since 2007 (when we have building information)
    fire_incidents = fire_incidents[fire_incidents['Incident_Date']>=date_to_start]
    #remove all fire incidents for 2019 since we're trying to predict that year
    #fire_incidents = fire_incidents[fire_incidents['Incident_Date']<='2019-01-01']
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

    ### Generate the predictors dataset

    fires_dataset_geo = gpd.GeoDataFrame(fires_dataset_cleaned,geometry='geometry')
    fires_dataset_geo.crs = {'init': 'epsg:4326'}

    fires_dataset_geo['Structure_fire'] = fires_dataset_geo['Structure_fire'].astype(int)
    fires_dataset_geo['Vehicle_fire'] = fires_dataset_geo['Vehicle_fire'].astype(int)
    fires_dataset_geo['External_fire'] = fires_dataset_geo['External_fire'].astype(int)
    fires_dataset_geo['Incident_Year'] = fires_dataset_geo['Incident_Date'].apply(lambda x: int(x.year))

    intersections = gpd.sjoin(SF_blocks, fires_dataset_geo, how="left", op='contains')

    Fires_per_block = intersections[['GISJOIN','Incident_Year','Structure_fire','Vehicle_fire','External_fire']].groupby(['GISJOIN','Incident_Year']).sum()
    Fires_per_block.reset_index(inplace=True)
    Fires_per_block['Year'] = Fires_per_block['Incident_Year']
    Fires_per_block.drop('Incident_Year',axis=1,inplace=True)

    Fires_per_block['GISYEARJOIN'] = Fires_per_block.apply(DM.generateGISyearjoin,axis=1)

    #Generate temp SF_blocks_years DF

    IDs = []
    blocks = []
    newyears = []
    block_polys = list(SF_blocks['geometry'])
    block_IDs = list(SF_blocks['GISJOIN'])

    years = Fires_per_block['Year'].unique().astype(int)

    for i in range(len(block_IDs)):
        block = block_polys[i]
        ID = str(block_IDs[i])
        for year in years:
            IDs.append(ID+str(year))
            blocks.append(block)
            newyears.append(year)

    SF_blocks_years = gpd.GeoDataFrame({'GISYEARJOIN':IDs,'geometry':blocks,'IDyear':newyears})

    Fires_per_block_year = SF_blocks_years.merge(Fires_per_block,how='outer',on='GISYEARJOIN')

    Fires_per_block_year.replace(np.nan,0,inplace=True)

    return Fires_per_block_year

def assemble_fires_target(fires):

    '''
    Assemble the target dataset, which is the fires dataset shifted one year into the future
    '''

    fires_to_predict = fires.copy()

    #Reduce the value of GISYEARJOIN by 1 so that the fires can be joined to the previous year's
    #block
    fires_to_predict['GISYEARJOIN'] = fires_to_predict['GISYEARJOIN'].apply(lambda x: x[:-4]+str(int(x[-4:])-1))
    fires_to_predict.columns = ['GISYEARJOIN','IDyear','GISJOIN','SF_pred','VF_pred','EF_pred','Year']

    allfires_with_pred = allfires.merge(fires_to_predict,on='GISYEARJOIN',how='left')

    return allfires_with_pred

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

    years=np.arange(2007,2019)

    SF_blocks_years = DM.generateSF_blocks_years(SF_blocks,years)

    #train, holdout = assemble_fires_dataframe('../datasets/fire/',2018,SF_blocks)

    #print(holdout.head())
    #print(len(holdout))
    #print(len(train))

    merged_fires = assemble_fires_dataframe_oneperyear('../datasets/fire/',SF_blocks,SF_blocks_years)
    merged_fires.drop('geometry',axis=1).to_csv('All_associated_fires_test.csv',index=False)

    print(merged_fires)
    print(len(merged_fires))
