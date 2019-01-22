#!/usr/bin/env python
#RMS 2019

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

    d = assemble_dataframe()
    print(d)
    print(len(d))
    d.to_csv("Census_block_data.csv")
