#!/usr/bin/env python

import pandas as pd
import geopandas as gpd
import numpy as np

import Dataset_manipulation as DM

def assemble_violations_dataframe_oneperyear(datapath,SF_blocks,date_to_start='2007-01-01'):

    fire_violations = pd.read_csv(datapath+"Fire_Violations.csv",low_memory=False)

    fire_violations.dropna(subset=['Location'],inplace=True)
    fire_violations['Violation Date'] = pd.to_datetime(fire_violations['Violation Date'])
    fire_violations = fire_violations[fire_violations['Violation Date']>'2007-01-01']

    fire_violations['geometry'] = fire_violations['Location'].apply(DM.convert_to_point)
    fire_violations['Year'] = fire_violations['Violation Date'].apply(lambda x: int(x.year))

    #Convert to geo dataframe
    fire_violations_geo = gpd.GeoDataFrame(fire_violations,geometry='geometry')
    fire_violations_geo.crs = {'init': 'epsg:4326'}

    # Merge violations - find the block that contains each fire
    intersections = gpd.sjoin(SF_blocks, fire_violations_geo, how="inner", op='contains')
    intersections.replace(np.nan,0,inplace=True)
    nviolations_per_block = intersections[['GISJOIN','Year','index_right']].groupby(['GISJOIN','Year']).count()
    nviolations_per_block.reset_index(inplace=True)

    nviolations_per_block['GISYEARJOIN'] = nviolations_per_block.apply(DM.generateGISyearjoin,axis=1)

    nviolations_per_block.columns = ['GISJOIN','Year','Nviolations',"GISYEARJOIN"]

    return nviolations_per_block

def assemble_violations_dataframe(datapath,year_to_predict,SF_blocks,date_to_start='2007-01-01'):

    fire_violations = pd.read_csv(datapath+"Fire_Violations.csv",low_memory=False)

    fire_violations.dropna(subset=['Location'],inplace=True)
    fire_violations['Violation Date'] = pd.to_datetime(fire_violations['Violation Date'])
    fire_violations = fire_violations[fire_violations['Violation Date']>date_to_start]
    fire_violations['Violation Year'] = fire_violations['Violation Date'].apply(lambda x: int(x.year))

    fire_violations['geometry'] = fire_violations['Location'].apply(DM.convert_to_point)

    fire_violations_geo = gpd.GeoDataFrame(fire_violations,geometry='geometry')
    fire_violations_geo.crs = {'init': 'epsg:4326'}

    if year_to_predict:

        training = fire_violations_geo[fire_violations_geo['Violation Year']<year_to_predict]
        holdout = fire_violations_geo[fire_violations_geo['Violation Year']==year_to_predict]

        # Merge violations - find the block that contains each fire
        intersections_tr = gpd.sjoin(SF_blocks, training, how="inner", op='contains')
        intersections_tr.replace(np.nan,0,inplace=True)

        ## Note that here we could also group by year and then flatten so that we have more examples
        ## i.e. one row per year to predict.

        nviolations_per_block_tr = intersections_tr[['GISJOIN','index_right']].groupby('GISJOIN').count()
        nviolations_per_block_tr['GridCellID'] = list(nviolations_per_block_tr.index)
        nviolations_per_block_tr.reset_index(inplace=True)

        Violations_per_block_tr = SF_blocks.merge(nviolations_per_block_tr,how='left').drop('GridCellID',axis=1)
        Violations_per_block_tr.replace(np.nan,0,inplace=True)
        Violations_per_block_tr.columns = ['GISJOIN','geometry','nviolations']

        #intersections_ho = gpd.sjoin(SF_blocks, holdout, how="inner", op='contains')
        #intersections_ho.replace(np.nan,0,inplace=True)

        #nviolations_per_block_ho = intersections_ho[['GISJOIN','index_right']].groupby('GISJOIN').count()
        #nviolations_per_block_ho['GridCellID'] = list(nviolations_per_block_ho.index)
        #nviolations_per_block_ho.reset_index(inplace=True)

        #Violations_per_block_ho = SF_blocks.merge(nviolations_per_block_ho,how='left').drop('GridCellID',axis=1)
        #Violations_per_block_ho.replace(np.nan,0,inplace=True)
        #Violations_per_block_ho.columns = ['geometry','GISJOIN','nviolations']

    else:
        # Merge violations - find the block that contains each fire
        intersections_tr = gpd.sjoin(SF_blocks, fire_violations_geo, how="inner", op='contains')
        intersections_tr.replace(np.nan,0,inplace=True)

        # Merge violations - find the block that contains each fire
        intersections_tr = gpd.sjoin(SF_blocks, training, how="inner", op='contains')
        intersections_tr.replace(np.nan,0,inplace=True)

        nviolations_per_block_tr = intersections_tr[['GISJOIN','index_right']].groupby('GISJOIN').count()
        nviolations_per_block_tr['GridCellID'] = list(nviolations_per_block_tr.index)
        nviolations_per_block_tr.reset_index(inplace=True)

        Violations_per_block_tr = SF_blocks.merge(nviolations_per_block_tr,how='left').drop('GridCellID',axis=1)
        Violations_per_block_tr.replace(np.nan,0,inplace=True)
        Violations_per_block_tr.columns = ['GISJOIN','geometry','nviolations']
        #Violations_per_block_ho = None

    return Violations_per_block_tr #Violations_per_block_ho


if __name__ == "__main__":

    census_path = '../datasets/census_blocks/'

    SF_blocks = gpd.read_file(census_path+'SF_block_2010.shp')

    #train = assemble_violations_dataframe('../datasets/fire/',2018,SF_blocks)

    #print(holdout.head())
    #print(train.head())
    #print(len(holdout))
    #print(len(train))

    violations_blocks_years = assemble_violations_dataframe_oneperyear('../datasets/fire/',SF_blocks)

    print(violations_blocks_years)
