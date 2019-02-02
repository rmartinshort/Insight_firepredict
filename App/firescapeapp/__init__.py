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
SFmodelname = 'firescapeapp/models/Model_RC_SF_N1.sav'
EFmodelname = 'firescapeapp/models/Model_RC_EF_N1.sav'
VFmodelname = 'firescapeapp/models/Model_RC_VF_N1.sav'

SFmodel = pickle.load(open(SFmodelname, 'rb'))
EFmodel = pickle.load(open(EFmodelname, 'rb'))
VFmodel = pickle.load(open(VFmodelname, 'rb'))

pred2019data = pd.read_csv('firescapeapp/models/datasets/2019_predictfires.csv')
pred2018data = pd.read_csv('firescapeapp/models/datasets/2018_predictfires.csv')
print("\nDone loading datasets and models")

#Number of blocks to designate high risk
blocks_SF = 290
blocks_EF = 560
blocks_VF = 180

from firescapeapp import views
