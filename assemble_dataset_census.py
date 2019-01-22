#!/usr/bin/env python

# Functions for assembling fire dataset

import pandas as pd
import geopandas as gpd
import numpy as np
import os
from shapely.geometry import Polygon, Point


def withinSF(row):

    lon = float(row['INTPTLON10'])
    lat = float(row['INTPTLAT10'])

    if (-122.53 < lon < -122.35 ) and (37.7 < lat < 37.84 ):

        return 1

    else:

        return 0

def generate_blocks_from_GIS(census_path):

    '''
    Input: Path to census data
    Output: Geodataframe containing shape files corresponding to each SF census block
    '''

    print("\nLoading census data shape files\n")

    if os.path.isfile(census_path+'SF_block_2010.shp'):

        print("Found SF_block_2010.shp")

        SF_blocks = gpd.read_file(census_path+'SF_block_2010.shp')

    else:

        print("Generating SF_block_2010.shp")

        #load census dataset (this is large - need to cut just SF)
        CA_blocks = gpd.read_file(census_path+'CA_block_2010.shp')
        CA_census = pd.read_csv(census_path+'nhgis0002_ds172_2010_block.csv',low_memory=False)

        SF_data = CA_census[CA_census['COUNTY']=='San Francisco County']
        merged_SF = CA_blocks.merge(SF_data,on='GISJOIN')

        merged_SF.crs = {'init' : 'ESRI:102003'}
        merged_SF = merged_SF.to_crs({'init': 'epsg:4326'})

        #Get just the census blocks. We will use this to build out the full dataframe
        #print(merged_SF.columns)
        merged_SF['withincity'] = merged_SF.apply(withinSF,axis=1)
        SF_blocks = merged_SF[merged_SF['withincity']==1][['geometry','GISJOIN']]
        SF_blocks.to_file(census_path+'SF_block_2010.shp')

    return SF_blocks

def generate_yearblocks_from_GIS(census_path):

    '''
    Input: Path to census data
    Output: Geodataframe containing shape files corresponding to each SF census block
    '''

    print("\nLoading census data shape files\n")

    if os.path.isfile(census_path+'SF_block_years_2010.shp'):

        print("Found SF_block_2010.shp")

        SF_blocks_years = gpd.read_file(census_path+'SF_block_years_2010.shp')

    else:

        print("Generating SF_block_2010.shp")

        #load census dataset (this is large - need to cut just SF)
        CA_blocks = gpd.read_file(census_path+'CA_block_2010.shp')
        CA_census = pd.read_csv(census_path+'nhgis0002_ds172_2010_block.csv',low_memory=False)

        SF_data = CA_census[CA_census['COUNTY']=='San Francisco County']
        merged_SF = CA_blocks.merge(SF_data,on='GISJOIN')

        merged_SF.crs = {'init' : 'ESRI:102003'}
        merged_SF = merged_SF.to_crs({'init': 'epsg:4326'})

        #Get just the census blocks. We will use this to build out the full dataframe
        #print(merged_SF.columns)
        merged_SF['withincity'] = merged_SF.apply(withinSF,axis=1)
        SF_blocks = merged_SF[merged_SF['withincity']==1][['geometry','GISJOIN','INTPTLAT10','INTPTLON10','ALAND10']]

        ## Assemble yearblocks dataframe

        years = np.arange(2007,2019)
        IDs = []
        blocks = []
        newyears = []
        newlats = []
        newlons = []
        newareas = []
        block_polys = list(SF_blocks['geometry'])
        block_IDs = list(SF_blocks['GISJOIN'])
        block_lats = list(SF_blocks['INTPTLAT10'])
        block_lons = list(SF_blocks['INTPTLON10'])
        block_areas = list(SF_blocks['ALAND10'])

        for i in range(len(block_IDs)):
            block = block_polys[i]
            ID = str(block_IDs[i])
            lon = block_lons[i]
            lat = block_lats[i]
            area = block_areas[i]
            for year in years:
                IDs.append(ID+str(year))
                blocks.append(block)
                newyears.append(year)
                newlats.append(lat)
                newlons.append(lon)
                newareas.append(area)
        SF_blocks_years = gpd.GeoDataFrame({'GISYEARJOIN':IDs,'geometry':blocks,\
        'IDyear':newyears,'LAT':newlats,'LON':newlons,'AREA':newareas})

        SF_blocks_years.crs = {'init' : 'ESRI:102003'}
        SF_blocks_years.to_file(census_path+'SF_block_years_2010.shp')

    return SF_blocks_years

def assemble_census_dataframe(datapath,SF_blocks):

    '''
    Input: Path to data, dataframe of SF census blocks
    Output: Dataframe containing the cenusus block data
    '''

    pop_household_type = pd.read_csv(datapath+"nhgis0002_ds172_2010_block.csv",
                                 low_memory=False)

    housing_tenure = pd.read_csv(datapath+"nhgis0003_ds172_2010_block.csv",
                             low_memory=False)


    # To do - add te additional census information

    prices_1990 = pd.read_csv(datapath+"nhgis0005_ds120_1990_block.csv",low_memory=False)

    SF_pop_housing = pop_household_type[pop_household_type['COUNTY']=='San Francisco County']
    SF_tenure = housing_tenure[housing_tenure['COUNTY']=='San Francisco County']

    SF_pop_housing = SF_pop_housing[['GISJOIN','URBRURALA','H7X001', 'H7X002', 'H7X003', \
    'H7X004', 'H7X005', 'H7X006', 'H7X007','H7X008', 'H8C001', 'H8C002', 'H8C003', \
    'H8C004', 'H8C005', 'H8C006','H8C007', 'H8C008', 'H8C009']]

    SF_housing_type = SF_tenure[['GISJOIN','IFC001',
       'IFF001', 'IFF002', 'IFF003', 'IFF004']]

    merged_data = SF_pop_housing.merge(SF_housing_type,how='left',on='GISJOIN')

    merged_data['Urban'] = pd.get_dummies(merged_data['URBRURALA'],drop_first=True)
    merged = merged_data.drop('URBRURALA',axis=1)

    return merged

if __name__ == "__main__":

    ##TEST 1

    #census_path = '../datasets/census_blocks/'

    #SF_blocks = gpd.read_file(census_path+'SF_block_2010.shp')

    #train  = assemble_property_dataframe('../datasets/census_blocks/',2018,SF_blocks)

    #print(train.head())
    #print(len(train))

    ##TEST2

    #blocks = generate_blocks_from_GIS('../datasets/census_blocks/')
    #print(blocks.head)

    blocks = generate_yearblocks_from_GIS('../datasets/census_blocks/')
    print(blocks.head)
