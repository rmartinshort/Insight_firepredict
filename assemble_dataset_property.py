#!/usr/bin/env python
#RMS 2019

#Functions used to assemble properties dataframe for SF fire prediction project

import pandas as pd
import geopandas as gpd
import numpy as np

import Dataset_manipulation as DM


def assemble_landuse(SF_blocks):

    '''
    Called within assemble_property_dataframe_oneperyear
    Makes a dataframe containing information from the landuse dataset (note that this is static)
    '''

    landusepoint = gpd.read_file('/Users/rmartinshort/Documents/Insight/Project/datasets/landuse/geo_export_f48c6d46-b3af-4695-b4fd-9151e793f636.shp')

    landusepoint['central_point'] = landusepoint['geometry'].apply(DM.convert_poly_to_point)

    landusepoint = landusepoint.drop('geometry',axis=1).copy()

    landusepoint['geometry'] = landusepoint['central_point']

    landusepoint = landusepoint[(landusepoint['yrbuilt']<=2018) & (landusepoint['yrbuilt']>=1850)]

    #Find the census blocks that can be associated with this data
    censusblocks_containing_blocks = gpd.sjoin(SF_blocks,landusepoint,how='left',op='contains')

    censusblocks_containing_blocks_grouped = censusblocks_containing_blocks[['GISJOIN','geometry','index_right']].groupby('GISJOIN').count()

    #List of GISJOIN blocks for which we have data
    GISJOINs = list(censusblocks_containing_blocks_grouped[censusblocks_containing_blocks_grouped['index_right']>0].index)

    Censusdata = censusblocks_containing_blocks[censusblocks_containing_blocks['GISJOIN'].isin(GISJOINs)]

    Censusdata = Censusdata[['GISJOIN','block_num','landuse','resunits','retail','yrbuilt']]

    minyearperblock = Censusdata[['GISJOIN','yrbuilt']].groupby('GISJOIN').min().reset_index()
    minyearperblock.columns = ['GISJOIN','minyr']
    maxyearperblock = Censusdata[['GISJOIN','yrbuilt']].groupby('GISJOIN').max().reset_index()
    maxyearperblock.columns = ['GISJOIN','maxyr']
    rangeyearperblock = Censusdata[['GISJOIN','yrbuilt']].groupby('GISJOIN').std().reset_index()
    rangeyearperblock.columns = ['GISJOIN','varyr']
    nresunits = Censusdata[['GISJOIN','resunits']].groupby('GISJOIN').count().reset_index()

    a = censusblocks_containing_blocks_grouped.merge(minyearperblock,on='GISJOIN',how='left')
    b = a.merge(maxyearperblock,on='GISJOIN',how='left')
    c = b.merge(rangeyearperblock,on='GISJOIN',how='left')
    d = c.merge(nresunits,on='GISJOIN',how='left')
    d.dropna(inplace=True)

    return d


def assemble_property_dataframe_oneperyear(datapath,SF_blocks,year_to_start=2007):

    properties = pd.read_csv(datapath+'Property_Tax.csv',low_memory=False)

    properties = properties.dropna(subset=['the_geom','Closed Roll Year'])

    #Representing year is one year ahead of the roll year
    properties['Representing year'] = properties['Closed Roll Year'].apply(DM.add_year)
    #properties['Representing year'] = properties['Closed Roll Year']

    properties_by_year = properties.groupby("Representing year").count()

    properties = properties[['Representing year','Assessed Land Value','Number of Units','Year Property Built','Use Code','Construction Type','the_geom','Lot Area']]
    properties.dropna(inplace=True)

    properties['geometry'] = properties['the_geom'].apply(DM.convert_to_point)

    properties_geo = gpd.GeoDataFrame(properties,geometry='geometry')
    properties_geo.crs = {'init': 'epsg:4326'}
    # Merge properties - find the block that contains each property
    intersections = gpd.sjoin(SF_blocks, properties_geo, how="left", op='contains')

    intersections.dropna(inplace=True)

    #Tidy up property build year distributions
    intersections = intersections[(intersections['Year Property Built']<=2018) & (intersections['Year Property Built']>=1890)]
    intersections = intersections[intersections['Assessed Land Value'] < intersections['Assessed Land Value'].quantile(.95)]

    #Removed everything above 95 percetile of lot area, and lots with recorded area of 0
    intersections = intersections[intersections['Lot Area'] < intersections['Lot Area'].quantile(.95)]
    intersections = intersections[intersections['Lot Area'] > 0]

    #properties_by_year = properties.groupby("Representing year").count()
    #Determine the mean of these features across each census block
    means = intersections[['GISJOIN','Representing year','Year Property Built',\
    'Assessed Land Value','Number of Units','Lot Area']].groupby(['GISJOIN','Representing year']).mean()

    means.reset_index(inplace=True)
    Means_per_block_year = means
    Means_per_block_year['Year'] = Means_per_block_year['Representing year']
    Means_per_block_year.drop('Representing year',axis=1,inplace=True)
    Means_per_block_year['GISYEARJOIN'] = Means_per_block_year.apply(DM.generateGISyearjoin,axis=1)

    # Count the number of properties for each use code
    counts_use = intersections[['GISJOIN','Representing year','Use Code','Construction Type']].groupby(['GISJOIN','Representing year','Use Code']).count()
    counts_use = counts_use.unstack(2)
    counts_use.columns = counts_use.columns.droplevel(0)
    counts_use.columns = [str(cname) for cname in list(counts_use.columns)]
    counts_use.fillna(0,inplace=True)
    b = counts_use.div(counts_use.sum(axis=1), axis=0)
    b.reset_index(inplace=True)

    cols = list(b.columns)
    b['Sum'] = b[cols[2:]].sum(axis=1)
    b['UnkownUseType']=b['Sum'].apply(DM.fillunknown)

    Use_per_block_year = b
    Use_per_block_year['Year'] = Use_per_block_year['Representing year']
    Use_per_block_year.drop(['Representing year','Sum'],axis=1,inplace=True)
    Use_per_block_year['GISYEARJOIN'] = Use_per_block_year.apply(DM.generateGISyearjoin,axis=1)

    #Count the number of property types per cell
    counts_type = intersections[['GISJOIN','Representing year','Use Code','Construction Type']].groupby(['GISJOIN','Representing year','Construction Type']).count()
    a = counts_type.unstack(2)
    a.columns = a.columns.droplevel(0)
    a.columns = [str(cname) for cname in list(a.columns)]
    a.fillna(0,inplace=True)
    b = a.div(a.sum(axis=1), axis=0)
    b.reset_index(inplace=True)
    print(b.columns)
    #Sum over all the unknown types
    b['S'] = b[['1','BRI','F','R','REI','ROW','S','STE','WOO']].sum(axis=1)
    #Drop the other unknown types
    b = b.drop(['1','BRI','F','R','REI','ROW','STE','WOO'],axis=1)

    Type_per_block_year = b
    Type_per_block_year['Year'] = Type_per_block_year['Representing year']
    Type_per_block_year.drop('Representing year',axis=1,inplace=True)
    Type_per_block_year['GISYEARJOIN'] = Type_per_block_year.apply(DM.generateGISyearjoin,axis=1)

    ## Merge

    d1 = Means_per_block_year.merge(Use_per_block_year,how='outer',on='GISYEARJOIN')
    d2 = d1.merge(Type_per_block_year,how='outer',on='GISYEARJOIN')
    All_properties = d2.drop(['GISJOIN_x','Year_x','GISJOIN_y','Year_y'],axis=1)
    #All_properties.rename(index=str, columns={"GISJOIN_x": "GISJOIN"})

    landuse = assemble_landuse(SF_blocks)
    landuse.to_csv("landuse_on_GISJOIN.csv",index=False)

    return All_properties, landuse

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

    #train  = assemble_property_dataframe('../datasets/property/',2018,SF_blocks)

    #print(train.head())
    #print(len(holdout))
    #print(len(train))

    properties,landuse = assemble_property_dataframe_oneperyear('../datasets/property/',SF_blocks)
    #properties.to_csv('Properties_with_offset_year.csv',index=False)
    print(properties.head())
    print(len(properties))
