#!/usr/bin/env python
#RMS 2019

#Functions used to assemble properties dataframe for SF fire prediction project

import pandas as pd
import geopandas as gpd
import numpy as np

import Dataset_manipulation as DM


def setpropertyna(value):

    '''Set property built value to nan if it is unrealistic'''

    if value > 2019:
        return np.nan
    else:
        return value

def fillunknown(sumval):

    '''Return 1 if all use types were NaN - the means we have an unknown type'''

    if sumval == 0:
        return 1
    else:
        return 0

def return_training(intersections,SF_blocks):

        means = intersections[['GISJOIN','Year Property Built','Assessed Land Value','Lot Area']].groupby('GISJOIN').mean()
        sums= intersections[['GISJOIN','Number of Units']].groupby(['GISJOIN']).sum()
        counts_1 = intersections[['GISJOIN','Use Code','Construction Type']].groupby(['GISJOIN','Use Code']).count()
        counts_2 = intersections[['GISJOIN','Use Code','Construction Type']].groupby(['GISJOIN','Construction Type']).count()

        #Dealing with counts of Use Code
        a = counts_1.unstack(1)
        a.columns = a.columns.droplevel(0)
        a.columns = [str(cname) for cname in list(a.columns)]
        a.fillna(0,inplace=True)
        b = a.div(a.sum(axis=1), axis=0)
        b['GISJOIN'] = list(b.index)
        b.reset_index(drop=True)

        counts_use_per_block = SF_blocks.merge(b,how='left')
        counts_use_per_block.replace(np.nan,0,inplace=True)
        cols = list(counts_use_per_block.columns)
        counts_use_per_block['Sum'] = counts_use_per_block[cols[1:]].sum(axis=1)
        counts_use_per_block['UnkownUseType']=counts_use_per_block['Sum'].apply(fillunknown)

        #Dealing with counts of Property Type
        a = counts_2.unstack(1)
        a.columns = a.columns.droplevel(0)
        a.columns = [str(cname) for cname in list(a.columns)]
        a.fillna(0,inplace=True)
        b = a.div(a.sum(axis=1), axis=0)
        b['GISJOIN'] = list(b.index)
        b.reset_index(drop=True)

        #Sum over all the unknown types
        b['S'] = b[['1','?','BRI','F','R','REI','ROW','S','STE','WOO']].sum(axis=1)
        #Drop the other unknown types
        b = b.drop(['1','?','BRI','F','R','REI','ROW','STE','WOO'],axis=1)

        counts_structure_per_block = SF_blocks.merge(b,how='left')

        counts_structure_per_block.replace(np.nan,0,inplace=True)
        cols = list(counts_structure_per_block.columns)
        counts_structure_per_block['Sum'] = counts_structure_per_block[cols[1:]].sum(axis=1)
        counts_structure_per_block['UnkownStructureType']=counts_structure_per_block['Sum'].apply(fillunknown)

        #Dealing with means columns (these will have NaNs)
        means['GISJOIN'] = list(means.index)
        means_per_block = SF_blocks.merge(means,how='left')

        #Dealing with sums columns (these will have NaNs)
        sums['GISJOIN'] = list(sums.index)
        sums_per_block = SF_blocks.merge(sums,how='left')

        return sums_per_block, means_per_block, counts_structure_per_block, counts_use_per_block


def assemble_property_dataframe(datapath,year_to_predict,SF_blocks,yr_to_start=2007):

    '''
    Input: path to dataset, year to predict (for data redaction), dataframe of SF shapefile
    Output: Dataframe containing property features for each census region
    '''

    properties = pd.read_csv(datapath+'Property_Tax.csv',low_memory=False)

    properties = properties[properties['Closed Roll Year']>=yr_to_start]

    #May want to make some more feature engineering choices here
    properties = properties[['Assessed Land Value','Number of Units',\
    'Year Property Built','Use Code','Construction Type','the_geom','Lot Area','Closed Roll Year']]

    properties['Year Property Built'] = properties['Year Property Built'].apply(setpropertyna)
    properties.dropna(inplace=True)

    properties['geometry'] = properties['the_geom'].apply(DM.convert_to_point)

    properties_geo = gpd.GeoDataFrame(properties,geometry='geometry')
    properties_geo.crs = {'init': 'epsg:4326'}

    if year_to_predict:

        training = properties_geo[properties_geo['Closed Roll Year']<year_to_predict]

        intersections = gpd.sjoin(SF_blocks, training, how="inner", op='contains')

        sums_per_block, means_per_block, counts_structure_per_block, counts_use_per_block = \
        return_training(intersections,SF_blocks)

    else:

        intersections = gpd.sjoin(SF_blocks, properties_geo, how="inner", op='contains')

        sums_per_block, means_per_block, counts_structure_per_block, counts_use_per_block = \
                return_training(intersections,SF_blocks)

    m1 = counts_structure_per_block.drop('geometry',axis=1).\
    merge(counts_use_per_block.drop('geometry',axis=1),how='left',on='GISJOIN')

    m2 = m1.merge(sums_per_block.drop('geometry',axis=1),how='left',on='GISJOIN')
    m3 = m2.merge(means_per_block.drop('geometry',axis=1),how='left',on='GISJOIN')
    m4 = SF_blocks.merge(m3,how='left',on='GISJOIN')
    m4.drop(['Sum_x','Sum_y'],axis=1,inplace=True)

    return m4

if __name__ == "__main__":

    census_path = '../datasets/census_blocks/'

    SF_blocks = gpd.read_file(census_path+'SF_block_2010.shp')

    train  = assemble_property_dataframe('../datasets/property/',2018,SF_blocks)

    print(train.head())
    print(len(holdout))
    print(len(train))
