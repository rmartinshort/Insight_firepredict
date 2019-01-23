#!/usr/bin/env python

#Tools for routine manipulation of the datasets

from shapely.geometry import Point, Polygon
import geopandas as gpd


def convert_to_point(coords):

    coords = coords.split(',')
    lat = float(coords[0][1:])
    lon = float(coords[1][:-1])

    return Point(lon,lat)

def generateGISyearjoin(row):

    return str(row['GISJOIN'])+str(int(row['Year']))

def add_year(val):

    '''Time timeline classifiction for the tax data is different from all other datasets.
    The tax year runs from July 1st to June 30th. A record with roll year of 2008 means the year 2008-2009,
    for example. Because we want to predict the future, we need to use 2017's taxroll information with 2018's
    other features. Thus we need to add 1 to each year represented here
    '''

    return int(val)+1

def generateGISyearjoin_predict(row):

    '''
    We want to predict the fires one year in advance, so we need to join each row of fires
    to the previous year. 2018 fires will be associated with 2017, for example. There will be no fires
    associated with 2018
    '''

    return str(row['GISJOIN'])+str(int(row['Year'])-1)

def generateSF_blocks_years(SF_blocks,years):

    IDs = []
    blocks = []
    newyears = []
    block_polys = list(SF_blocks['geometry'])
    block_IDs = list(SF_blocks['GISJOIN'])

    for i in range(len(block_IDs)):
       block = block_polys[i]
       ID = str(block_IDs[i])
       for year in years:
          IDs.append(ID+str(year))
          blocks.append(block)
          newyears.append(year)

    SF_blocks_years = gpd.GeoDataFrame({'GISYEARJOIN':IDs,'geometry':blocks,'IDyear':newyears})

    return SF_blocks_years

def fillunknown(sumval):
    if sumval == 0:
        return 1
    else:
        return 0
