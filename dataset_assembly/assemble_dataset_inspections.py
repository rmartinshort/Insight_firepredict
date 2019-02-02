#!/usr/bin/env python

import pandas as pd
import geopandas as gpd
import numpy as np

import Dataset_manipulation as DM


def assemble_inspections_dataframe_oneperyear(datapath,SF_blocks,date_to_start='2017-01-01'):

    fire_inspections = pd.read_csv(datapath+"Fire_Inspections.csv",low_memory=False)
    fire_inspections.dropna(subset=["Location","Inspection Start Date","Inspection End Date"],inplace=True)

    fire_inspections['Inspection Start Date'] = pd.to_datetime(fire_inspections['Inspection Start Date'])

    #Get only the data since 2007 (when we have building information)
    fire_inspections = fire_inspections[fire_inspections['Inspection Start Date']>='2007-01-01']
    #Get the number of complaints
    fire_inspections['Was complaint'] = fire_inspections['Inspection Type Description']=='complaint inspection'

    #Convert each location to a shapely point and fix date
    fire_inspections['geometry'] = fire_inspections['Location'].apply(DM.convert_to_point)
    fire_inspections['Year'] = fire_inspections['Inspection Start Date'].apply(lambda x: x.year)

    #Convert to geo dataframe
    fire_inspections_geo = gpd.GeoDataFrame(fire_inspections,geometry='geometry')
    fire_inspections_geo.crs = {'init': 'epsg:4326'}

    # Merge inspections - find the block that contains each inspection. Merge inner because we want to count (not sum)
    intersections = gpd.sjoin(SF_blocks, fire_inspections_geo, how="inner", op='contains')
    intersections.replace(np.nan,0,inplace=True)

    ninspections_per_block = intersections[['GISJOIN','Inspection Number','Year']].groupby(['GISJOIN','Year']).count()
    ninspections_per_block.reset_index(inplace=True)

    ninspections_per_block['GISYEARJOIN'] = ninspections_per_block.apply(DM.generateGISyearjoin,axis=1)

    ncomplaints_per_block = intersections[['GISJOIN','Was complaint','Year']].groupby(['GISJOIN','Year']).sum()
    ncomplaints_per_block.reset_index(inplace=True)

    ncomplaints_per_block['GISYEARJOIN'] = ncomplaints_per_block.apply(DM.generateGISyearjoin,axis=1)

    Inspections_per_year_block = ncomplaints_per_block.merge(ninspections_per_block,on='GISYEARJOIN',how='outer')
    Inspections_per_year_block.drop(['GISJOIN_x','Year_x','GISJOIN_y','Year_y'],axis=1,inplace=True)

    return Inspections_per_year_block



def assemble_inspections_dataframe(datapath,year_to_predict,SF_blocks,date_to_start='2007-01-01'):

    '''
    Input: Path to inspection data, year (or other timeframe) to predict, Dataframe of SF blocks
    Output: Dataframes for training and holdout
    '''

    fire_inspections = pd.read_csv(datapath+'Fire_Inspections.csv',low_memory=False)

    fire_inspections.dropna(subset=["Location","Inspection Start Date","Inspection End Date"],inplace=True)
    fire_inspections['Inspection Start Date'] = pd.to_datetime(fire_inspections['Inspection Start Date'])

    #Get only the data since 2007 (when we have building information)
    fire_inspections = fire_inspections[fire_inspections['Inspection Start Date']>=date_to_start]

    #Get the number of complaints
    fire_inspections['Was complaint'] = \
    fire_inspections['Inspection Type Description']=='complaint inspection'

    fire_inspections['geometry'] = fire_inspections['Location'].apply(DM.convert_to_point)

    #Convert to geo dataframe
    fire_inspections_geo = gpd.GeoDataFrame(fire_inspections,geometry='geometry')
    fire_inspections_geo.crs = {'init': 'epsg:4326'}

    fire_inspections_geo['Inspection Start Year'] = fire_inspections_geo['Inspection Start Date'].apply(lambda x: int(x.year))

    if year_to_predict:
        training = fire_inspections_geo[fire_inspections_geo['Inspection Start Year']<year_to_predict]
        holdout = fire_inspections_geo[fire_inspections_geo['Inspection Start Year']==year_to_predict]

        # Merge fires - find the block that contains each fire (training)
        intersections_tr = gpd.sjoin(SF_blocks, training, how="inner", op='contains')
        intersections_tr.replace(np.nan,0,inplace=True)

        ninspections_per_block_tr = intersections_tr[['GISJOIN','Inspection Number']].groupby('GISJOIN').count()
        ninspections_per_block_tr['GridCellID'] = list(ninspections_per_block_tr.index)
        ninspections_per_block_tr.reset_index(inplace=True)

        ncomplaints_per_block_tr = intersections_tr[['GISJOIN','Was complaint']].groupby('GISJOIN').sum()
        ncomplaints_per_block_tr['GridCellID'] = list(ncomplaints_per_block_tr.index)
        ncomplaints_per_block_tr.reset_index(inplace=True)

        Inspections_per_block_tr = ncomplaints_per_block_tr.merge(ninspections_per_block_tr).drop('GridCellID',axis=1)

        #Do a left join with the geometry to get all the blocks back
        Inspections_per_block_tr['GridCellID'] = list(Inspections_per_block_tr.index)
        Inspections_per_block_tr.reset_index(inplace=True)
        Inspections_per_block_tr = SF_blocks.merge(Inspections_per_block_tr,how='left').drop('GridCellID',axis=1)
        Inspections_per_block_tr.replace(np.nan,0,inplace=True)
        Inspections_per_block_tr.drop('index',axis=1,inplace=True)

        # Merge fires - find the block that contains each fire (holdout)
        #intersections_ho = gpd.sjoin(SF_blocks, holdout, how="inner", op='contains')
        #intersections_ho.replace(np.nan,0,inplace=True)

        #ninspections_per_block_ho= intersections_ho[['GISJOIN','Inspection Number']].groupby('GISJOIN').count()
        #ninspections_per_block_ho['GridCellID'] = list(ninspections_per_block_ho.index)
        #ninspections_per_block_ho.reset_index(inplace=True)

        #ncomplaints_per_block_ho = intersections_ho[['GISJOIN','Was complaint']].groupby('GISJOIN').sum()
        #ncomplaints_per_block_ho['GridCellID'] = list(ncomplaints_per_block_ho.index)
        #ncomplaints_per_block_ho.reset_index(inplace=True)

        #Inspections_per_block_ho = ncomplaints_per_block_ho.merge(ninspections_per_block_ho).drop('GridCellID',axis=1)

        #Do a left join with the geometry to get all the blocks back
        #Inspections_per_block_ho['GridCellID'] = list(Inspections_per_block_ho.index)
        #Inspections_per_block_ho.reset_index(inplace=True)
        #Inspections_per_block_ho = SF_blocks.merge(Inspections_per_block_ho,how='left').drop('GridCellID',axis=1)
        #Inspections_per_block_ho.replace(np.nan,0,inplace=True)
        #Inspections_per_block_ho.drop('index',axis=1,inplace=True)

    else:
        intersections_tr = gpd.sjoin(SF_blocks, fire_inspections_geo, how="inner", op='contains')
        intersections_tr.replace(np.nan,0,inplace=True)

        ninspections_per_block_tr = intersections_tr[['GISJOIN','Inspection Number']].groupby('GISJOIN').count()
        ninspections_per_block_tr['GridCellID'] = list(ninspections_per_block_tr.index)
        ninspections_per_block_tr.reset_index(inplace=True)

        ncomplaints_per_block_tr = intersections_tr[['GISJOIN','Was complaint']].groupby('GISJOIN').sum()
        ncomplaints_per_block_tr['GridCellID'] = list(ncomplaints_per_block_tr.index)
        ncomplaints_per_block_tr.reset_index(inplace=True)

        Inspections_per_block_tr = ncomplaints_per_block_tr.merge(ninspections_per_block_tr).drop('GridCellID',axis=1)

        #Do a left join with the geometry to get all the blocks back
        Inspections_per_block_tr['GridCellID'] = list(Inspections_per_block_tr.index)
        Inspections_per_block_tr.reset_index(inplace=True)
        Inspections_per_block_tr = SF_blocks.merge(Inspections_per_block_tr,how='left').drop('GridCellID',axis=1)
        Inspections_per_block_tr.replace(np.nan,0,inplace=True)
        Inspections_per_block_tr.drop('index',axis=1,inplace=True)
        #Inspections_per_block_ho = None

    return Inspections_per_block_tr #Inspections_per_block_ho

if __name__ == "__main__":

    census_path = '../datasets/census_blocks/'

    SF_blocks = gpd.read_file(census_path+'SF_block_2010.shp')

    #train  = assemble_inspections_dataframe('../datasets/fire/',2018,SF_blocks)

    #print(holdout.head())
    #print(train.head())
    #print(len(holdout))
    #print(len(train))

    yearblock = assemble_inspections_dataframe_oneperyear('../datasets/fire/',SF_blocks)

    print(yearblock)
    print(len(yearblock))
