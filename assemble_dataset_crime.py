#!/usr/bin/env python

#Functions used to assemble crime dataframe for SF fire prediction project

import pandas as pd
import geopandas as gpd
import numpy as np

import Dataset_manipulation as DM


def assemble_crimes_dataframe_oneperyear(datapath,SF_blocks,date_to_start='2007-01-01'):

    crime = pd.read_csv(datapath+"Crime.csv",low_memory=False)
    crime.dropna(subset=['Date','Location','Category'],inplace=True)
    crime['Date'] = pd.to_datetime(crime['Date'])
    crime = crime[crime['Date']>'2007-01-01']
    crime = crime[~crime['Category'].str.contains('NON-CRIMINAL|SECONDARY CODES|RECOVERED VEHICLE')]

    #Ensure that these are type int
    crime['CrimeIsArson'] = crime['Category']=='ARSON'
    crime['CrimeIsOther'] = crime['Category']!='ARSON'
    crime['CrimeIsArson'] = crime['CrimeIsArson'].astype(int)
    crime['CrimeIsOther'] = crime['CrimeIsOther'].astype(int)

    crime['geometry'] = crime['Location'].apply(DM.convert_to_point)
    crime['Year'] = crime['Date'].apply(lambda x: int(x.year))

    crime_geo = gpd.GeoDataFrame(crime,geometry='geometry')
    crime_geo.crs = {'init': 'epsg:4326'}

    # Merge crimes - find the block that contains each crime
    intersections = gpd.sjoin(SF_blocks, crime_geo, how="left", op='contains')
    intersections.replace(np.nan,0,inplace=True)
    ncrimes_per_block = intersections[['GISJOIN','Year','CrimeIsArson','CrimeIsOther']].groupby(['GISJOIN','Year']).sum()
    ncrimes_per_block.reset_index(inplace=True)
    ncrimes_per_block['GISYEARJOIN'] = ncrimes_per_block.apply(DM.generateGISyearjoin,axis=1)

    return ncrimes_per_block




def assemble_crimes_dataframe(datapath,year_to_predict,SF_blocks,date_to_start='2007-01-01'):

    crime = pd.read_csv(datapath+"Crime.csv",low_memory=False)

    crime.dropna(subset=['Date','Location','Category'],inplace=True)
    crime['Date'] = pd.to_datetime(crime['Date'])
    crime = crime[crime['Date']>date_to_start]
    crime['Year'] = crime['Date'].apply(lambda x: int(x.year))

    #May need to make additional decisions about crimes to include here
    crime = crime[~crime['Category'].str.contains('NON-CRIMINAL|SECONDARY CODES|RECOVERED VEHICLE')]

    crime['CrimeIsArson'] = crime['Category']=='ARSON'
    crime['CrimeIsOther'] = crime['Category']!='ARSON'

    crime['geometry'] = crime['Location'].apply(DM.convert_to_point)

    crime_geo = gpd.GeoDataFrame(crime,geometry='geometry')
    crime_geo.crs = {'init': 'epsg:4326'}

    if year_to_predict:

        training = crime_geo[crime_geo['Year']<year_to_predict]
        holdout = crime_geo[crime_geo['Year']==year_to_predict]

        intersections_tr = gpd.sjoin(SF_blocks, training, how="left", op='contains')
        intersections_tr.replace(np.nan,0,inplace=True)

        Crimes_per_block_tr = intersections_tr[['GISJOIN','CrimeIsArson','CrimeIsOther']].groupby('GISJOIN').sum()
        Crimes_per_block_tr['CrimeIsArson'] = pd.to_numeric(Crimes_per_block_tr['CrimeIsArson'])
        Crimes_per_block_tr['CrimeIsOther'] = pd.to_numeric(Crimes_per_block_tr['CrimeIsOther'])

        #intersections_ho = gpd.sjoin(SF_blocks, holdout, how="left", op='contains')
        #intersections_ho.replace(np.nan,0,inplace=True)

        #Crimes_per_block_ho = intersections_ho[['GISJOIN','CrimeIsArson','CrimeIsOther']].groupby('GISJOIN').sum()
        #Crimes_per_block_ho['CrimeIsArson'] = pd.to_numeric(Crimes_per_block_ho['CrimeIsArson'])
        #Crimes_per_block_ho['CrimeIsOther'] = pd.to_numeric(Crimes_per_block_ho['CrimeIsOther'])

    else:

        intersections_tr = gpd.sjoin(SF_blocks, crime_geo, how="left", op='contains')
        intersections_tr.replace(np.nan,0,inplace=True)

        Crimes_per_block_tr = intersections_tr[['GISJOIN','CrimeIsArson','CrimeIsOther']].groupby('GISJOIN').sum()
        Crimes_per_block_tr['CrimeIsArson'] = pd.to_numeric(Crimes_per_block_tr['CrimeIsArson'])
        Crimes_per_block_tr['CrimeIsOther'] = pd.to_numeric(Crimes_per_block_tr['CrimeIsOther'])
        #Crimes_per_block_ho = None

    return Crimes_per_block_tr #Crimes_per_block_ho

if __name__ == "__main__":

    census_path = '../datasets/census_blocks/'

    SF_blocks = gpd.read_file(census_path+'SF_block_2010.shp')

    train  = assemble_crimes_dataframe_oneperyear('../datasets/crime/',SF_blocks)

    print(train.head())
    print(len(train))
