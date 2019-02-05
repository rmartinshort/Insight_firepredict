#!/usr/bin/env python
#RMS 2019

#Contains functions needed to assemble the datasets needed for modelling SF fire
#prediction. The user needs to enter paths to the relevant datasets

###
# Master file for dataset assembly

# Run this to assemble a dataframe of fire data ready for modelling
# The user can make a few choices about what features end up getting selected
# The aim is to make a dataframe that contains features for each of the census
# blocks. We will then predict the chance of fire (and/or the number of fires)
# for each block.

import pandas as pd
import geopandas as gpd
import numpy as np

import assemble_dataset_census as ADC
import assemble_dataset_fire as ADFire
import assemble_dataset_crime as ADCrime
import assemble_dataset_violations as ADViolations
import assemble_dataset_property as ADProperty
import assemble_dataset_inspections as ADInspections


def assemble_dataframe_yearjoin():

    '''
    Assemble dataframe based on yearjoined census blocks
    '''

    census_datapath = '../datasets/census_blocks/'
    fires_datapath = '../datasets/fire/'
    crime_datapath = '../datasets/crime/'
    property_datapath = '../datasets/property/'

    print("Assembling SF blocks")

    SF_blocks = ADC.generate_blocks_from_GIS(census_datapath)

    print("Assembling SF yearblocks")

    SF_yearblocks = ADC.generate_yearblocks_from_GIS(census_datapath)

    SF_yearblocks.columns = ['GISYEARJOIN', 'IDyear', 'LAT', 'LON', 'AREA', 'geometry']

    print("Assembling Violations")

    Violations_per_block = ADViolations.assemble_violations_dataframe_oneperyear(fires_datapath,SF_blocks)

    a = SF_yearblocks.merge(Violations_per_block,how='left',on='GISYEARJOIN')
    a.replace(np.nan,0,inplace=True)

    print("Assembling Inspections")

    Inspections_per_block = ADInspections.assemble_inspections_dataframe_oneperyear(fires_datapath,SF_blocks)

    b = a.merge(Inspections_per_block,how='left',on='GISYEARJOIN')
    b.replace(np.nan,0,inplace=True)

    print("Assembling Crimes")

    Crimes_per_block = ADCrime.assemble_crimes_dataframe_oneperyear(crime_datapath,SF_blocks)

    c = b.merge(Crimes_per_block,how='left',on='GISYEARJOIN')
    c.replace(np.nan,0,inplace=True)

    print("Assembling fires")

    Fires_per_block_year = ADFire.assemble_fires_dataframe_oneperyear(fires_datapath,SF_blocks)

    d = c.merge(Fires_per_block_year,how='left',on='GISYEARJOIN')

    print(d.isna().sum())

    d.replace(np.nan,0,inplace=True)

    print("Assembling properties")

    Properties_per_yearblock, landusejoin = ADProperty.assemble_property_dataframe_oneperyear(property_datapath,SF_blocks)

    e = d.merge(Properties_per_yearblock,how='left',on='GISYEARJOIN')

    print(e.isna().sum())

    e.drop(['GISJOIN_y','Year_y','IDyear_y','geometry_y'],axis=1,inplace=True)

    print("Assembling census data")

    Census_blocks_years = ADC.assemble_census_dataframe_oneperyear(census_datapath,SF_blocks)

    print("Generating target and merging")

    ALLDATA = e.merge(Census_blocks_years,on='GISYEARJOIN',how='left')

    print(ALLDATA.isna().sum())

    ALLDATA.drop(['GISJOIN_x','Year_x','index'],inplace=True,axis=1)
    ALLDATA['LAT'] = ALLDATA['LAT'].astype(float)
    ALLDATA['LON'] = ALLDATA['LON'].astype(float)
    ALLDATA['AREA'] = ALLDATA['AREA'].astype(float)
    ALLDATA['IDyear'] = ALLDATA['IDyear_x'].astype(int)
    ALLDATA.drop(['geometry_x'],axis=1,inplace=True)

    print(ALLDATA.isna().sum())

    Fires_per_block_year.drop('geometry',axis=1,inplace=True)
    #prepare to shift this dataframe one year forward in time
    fires_to_predict = Fires_per_block_year.copy()

    print(fires_to_predict.columns)

    fires_to_predict['GISYEARJOIN'] = fires_to_predict['GISYEARJOIN'].apply(lambda x: x[:-4]+str(int(x[-4:])-1))
    fires_to_predict.columns = ['GISYEARJOIN','IDyear','GISJOIN','SF_pred','VF_pred','EF_pred','Year']

    #allfires_with_pred = Fires_per_block_year.merge(fires_to_predict,on='GISYEARJOIN',how='left')

    ALLDATA2 = ALLDATA.merge(fires_to_predict,on='GISYEARJOIN',how='left')

    #merge landuse data (since we have GISJOIN)

    years = np.arange(2007,2019)
    o1 = list(landusejoin['GISJOIN'])
    o2 = list(landusejoin['minyr'])
    o3 = list(landusejoin['maxyr'])
    o4 = list(landusejoin['varyr'])
    o5 = list(landusejoin['resunits'])
    n1 = []
    n2 = []
    n3 = []
    n4 = []
    n5 = []
    indexcount = []
    j = 0
    for i in range(len(landusejoin)):

        a1 = o1[i]
        a2 = o2[i]
        a3 = o3[i]
        a4 = o4[i]
        a5 = o5[i]
        for year in years:
            n1.append(a1+str(year))
            n2.append(a2)
            n3.append(a3)
            n4.append(a4)
            n5.append(a5)
            indexcount.append(j)
            j += 1

    landuse_blocks_years = pd.DataFrame({'GISYEARJOIN':n1,'minyr':n2, 'maxyr':n3, 'sdyr':n4, 'resunits':n5})
    dataset2 = ALLDATA2.merge(landuse_blocks_years,how='left',on='GISYEARJOIN')
    dataset2.drop(['IDyear_y','Year','GISJOIN'],axis=1,inplace=True)
    
    dataset2.to_csv("Fully_merged_dataset_Autogenerated_plus.csv",index=False)


def assemble_dataframe():

    #If year_to_predict = None, then we use the full dataset as training
    #and attempt to predict some time into the future
    year_to_predict = 2018
    date_to_start = '2017-01-01'

    census_datapath = '../datasets/census_blocks/'
    fires_datapath = '../datasets/fire/'
    crime_datapath = '../datasets/crime/'
    property_datapath = '../datasets/property/'

    SF_blocks = ADC.generate_blocks_from_GIS(census_datapath)

    #Generate hold out and training datasets (Y)

    print("Assembling Fires dataframe (Y)")
    Fires_per_block_train, Fires_per_block_holdout = \
    ADFire.assemble_fires_dataframe(fires_datapath,year_to_predict,SF_blocks,date_to_start=date_to_start)

    print("Assembling Violations dataframe (X part)")

    Violations_per_block = \
    ADViolations.assemble_violations_dataframe(fires_datapath,year_to_predict,SF_blocks,date_to_start=date_to_start)

    print("Assembling Inspections dataframe (X part)")

    Inspections_per_block = \
    ADInspections.assemble_inspections_dataframe(fires_datapath,year_to_predict,SF_blocks,date_to_start=date_to_start)

    print("Assembling Crimes dataframe (X part)")

    Crimes_per_block = \
    ADCrime.assemble_crimes_dataframe(crime_datapath,year_to_predict,SF_blocks,date_to_start=date_to_start)

    print("Assembling Properties dataframe (X part)")

    Property_per_block = \
    ADProperty.assemble_property_dataframe(property_datapath,year_to_predict,SF_blocks)

    print("Assembling Census dataframe (X part)")

    Census_data = \
    ADC.assemble_census_dataframe(census_datapath,SF_blocks)

    print("Joining dataframes (X)")

    Fires_per_block_train['GISJOIN'] = list(Fires_per_block_train.index)

    print(Fires_per_block_train.columns)
    print(Inspections_per_block.columns)
    print(Violations_per_block.columns)
    print(Crimes_per_block.columns)
    print(Census_data.columns)

    Crimes_per_block['GISJOIN'] = list(Crimes_per_block.index)
    Crimes_per_block.reset_index(drop=True,inplace=True)

    a = Fires_per_block_train.merge(Inspections_per_block.drop('geometry',axis=1),how='outer',on='GISJOIN')
    b = a.merge(Violations_per_block.drop('geometry',axis=1),how='outer',on='GISJOIN')
    c = b.merge(Crimes_per_block,how='outer',on='GISJOIN')
    d = c.merge(Census_data,how='outer',on='GISJOIN')
    d.dropna(inplace=True)
    alldata = d.merge(Property_per_block,how='outer',on='GISJOIN')

    #Merge with the block information
    SFpolygeom = pd.read_csv(census_datapath+'/SF_polygeom.csv')
    alldata = alldata.merge(SFpolygeom,on='GISJOIN',how='outer')
    #This will need to be converted back into a Geodataframe for plotting
    alldata.to_csv("All_census_block_data.csv",index=False)

    return alldata

if __name__ == "__main__":

    #d = assemble_dataframe()
    #print(d)
    #print(len(d))
    #d.to_csv("Census_block_data.csv")

    assemble_dataframe_yearjoin()
