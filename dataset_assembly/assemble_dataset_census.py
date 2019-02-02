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

        print("Found SF_block_years_2010.shp")

        SF_blocks_years = gpd.read_file(census_path+'SF_block_years_2010.shp')

    else:

        print("Generating SF_block_years_2010.shp")

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

def assemble_census_dataframe_oneperyear(datapath,SF_blocks):

    '''
    Assemble census data on the block level
    '''

    pop_household_type = pd.read_csv(datapath+"nhgis0002_ds172_2010_block.csv",
                                 low_memory=False)
    housing_tenure = pd.read_csv(datapath+"nhgis0003_ds172_2010_block.csv",
                             low_memory=False)
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

    merged2 = SF_blocks.merge(merged,on='GISJOIN',how='left')

    #This is probably a terrible and inefficient way of generate a yearblock
    #dataframe, but it works

    o1 = list(merged['GISJOIN'])
    o2 = list(merged['H7X001'])
    o3 = list(merged['H7X002'])
    o4 = list(merged['H7X003'])
    o5 = list(merged['H7X004'])
    o6 = list(merged['H7X005'])
    o7 = list(merged['H7X006'])
    o8 = list(merged['H7X007'])
    o9 = list(merged['H7X008'])
    o10 = list(merged['H8C001'])
    o11 = list(merged['H8C002'])
    o12 = list(merged['H8C003'])
    o13 = list(merged['H8C004'])
    o14 = list(merged['H8C005'])
    o15 = list(merged['H8C006'])
    o16 = list(merged['H8C007'])
    o17 = list(merged['H8C008'])
    o18 = list(merged['H8C009'])
    o19 = list(merged['IFC001'])
    o20 = list(merged['IFF001'])
    o21 = list(merged['IFF002'])
    o22 = list(merged['IFF003'])
    o23 = list(merged['IFF004'])
    o24 = list(merged['Urban'])
    n1 = []
    n2 = []
    n3 = []
    n4 = []
    n5 = []
    n6 = []
    n7 = []
    n8 = []
    n9 = []
    n10 = []
    n11 = []
    n12 = []
    n13 = []
    n14 = []
    n15 = []
    n16 = []
    n17 = []
    n18 = []
    n19 = []
    n20 = []
    n21 = []
    n22 = []
    n23 = []
    n24 = []
    indexcount = []

    j = 0
    for i in range(len(merged)):

        a1 = o1[i]
        a2 = o2[i]
        a3 = o3[i]
        a4 = o4[i]
        a5 = o5[i]
        a6 = o6[i]
        a7 = o7[i]
        a8 = o8[i]
        a9 = o9[i]
        a10 = o10[i]
        a11 = o11[i]
        a12 = o12[i]
        a13 = o13[i]
        a14 = o14[i]
        a15 = o15[i]
        a16 = o16[i]
        a17 = o17[i]
        a18 = o18[i]
        a19 = o19[i]
        a20 = o20[i]
        a21 = o21[i]
        a22 = o22[i]
        a23 = o23[i]
        a24 = o24[i]

        years = np.arange(2007,2019)

        for year in years:

            n1.append(str(a1)+str(year))
            n2.append(a2)
            n3.append(a3)
            n4.append(a4)
            n5.append(a5)
            n6.append(a6)
            n7.append(a7)
            n8.append(a8)
            n9.append(a9)
            n10.append(a10)
            n11.append(a11)
            n12.append(a12)
            n13.append(a13)
            n14.append(a14)
            n15.append(a15)
            n16.append(a16)
            n17.append(a17)
            n18.append(a18)
            n19.append(a19)
            n20.append(a20)
            n21.append(a21)
            n22.append(a22)
            n23.append(a23)
            n24.append(a24)
            j += 1
            indexcount.append(j)

    census_blocks_years = pd.DataFrame({'index':indexcount,'GISYEARJOIN':n1,'H7X001':n2, 'H7X002':n3, 'H7X003':n4,\
       'H7X004':n5, 'H7X005':n6,
       'H7X006':n7, 'H7X007':n8, 'H7X008':n9, 'H8C001':n10, 'H8C002':n11, 'H8C003':n12, 'H8C004':n13,
       'H8C005':n14, 'H8C006':n15, 'H8C007':n16, 'H8C008':n17, 'H8C009':n18, 'IFC001':n19, 'IFF001':n20,
       'IFF002':n21, 'IFF003':n22, 'IFF004':n23, 'Urban':n24})

    return census_blocks_years



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

    SFblocks = generate_blocks_from_GIS('../datasets/census_blocks/')
    print(SFblocks.head)

    #SFyearblocks = generate_yearblocks_from_GIS('../datasets/census_blocks/')
    #print(blocks.head)

    census_blocks_years = assemble_census_dataframe_oneperyear('../datasets/census_blocks/',SFblocks)
    print(census_blocks_years)
