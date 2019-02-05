#!/usr/bin/env python
from flask import Flask
import geopandas as gpd
import pandas as pd
import pickle

app = Flask(__name__)


print("Loading datasets and models\n")
#load what we can to save time when making a new map 
SF_blocks = gpd.read_file('firescapeapp/models/SFblocks/SF_block_years_2010.shp')

#load the models
SFmodelname2018 = 'firescapeapp/models/RC_model_SF_2018.sav'
EFmodelname2018 = 'firescapeapp/models/RC_model_EF_2018.sav'
VFmodelname2018 = 'firescapeapp/models/RC_model_VF_2018.sav'

SFmodelname2017 = 'firescapeapp/models/RC_model_SF_2017.sav'
EFmodelname2017 = 'firescapeapp/models/RC_model_EF_2017.sav'
VFmodelname2017 = 'firescapeapp/models/RC_model_VF_2017.sav'

SFmodel2018 = pickle.load(open(SFmodelname2018, 'rb'))
EFmodel2018 = pickle.load(open(EFmodelname2018, 'rb'))
VFmodel2018 = pickle.load(open(VFmodelname2018, 'rb'))

SFmodel2017 = pickle.load(open(SFmodelname2017, 'rb'))
EFmodel2017 = pickle.load(open(EFmodelname2017, 'rb'))
VFmodel2017 = pickle.load(open(VFmodelname2017, 'rb'))

#load the data 
pred2019data = pd.read_csv('firescapeapp/models/datasets/2019_predictfires.csv')
pred2018data = pd.read_csv('firescapeapp/models/datasets/2018_predictfires.csv')
pred2017data = pd.read_csv('firescapeapp/models/datasets/2017_predictfires.csv')
print("\nDone loading datasets and models")

#Number of blocks to designate high risk
blocks_SF = 290
blocks_EF = 560
blocks_VF = 180

from firescapeapp import views
