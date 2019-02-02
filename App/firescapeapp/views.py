#!/usr/bin/env python

from flask import render_template, request
from geopy.geocoders import Nominatim
import pickle
import geopandas as gpd
import pandas as pd
import numpy as np

import folium
from branca.utilities import split_six
from shapely.geometry import Point
import firescapeapp 

app = firescapeapp.app


def generate_hazard_map_html(model,X,mapdata,html_map_name,plat,plon,firetype='structure'):

    '''
    Generate new hazard map as a html file
    INPUTS
    model: an estimator object
    X: the data for which we want to predict fires
    mapdata: block data in the form of SF_blocks_years
    html_map_name: name of the html file to be produced

    '''

    GISCELLS = list(X['GISYEARJOIN'])
    hazdata = X.drop(['GISYEARJOIN','SF_pred','VF_pred','EF_pred'],axis=1)
    mapgeom = mapdata[mapdata['GISYEARJOI'].isin(GISCELLS)]
    fires_holdout_predict = model.predict_proba(hazdata)

    fscore = 10.0*fires_holdout_predict[:,1]/max(fires_holdout_predict[:,1])
    if firetype == 'structure':
        vallim = sorted(fscore)[::-1][firescapeapp.blocks_SF]
    elif firetype == 'vehicle':
        vallim = sorted(fscore)[::-1][firescapeapp.blocks_VF]
    elif firetype == 'external':
        vallim = sorted(fscore)[::-1][firescapeapp.blocks_EF]

    fscoreHI = fscore.copy()
    fscoreHI[fscoreHI<=vallim]=0
    fscoreHI[fscoreHI>vallim]=1

    riskmap = gpd.GeoDataFrame({'geometry':mapgeom['geometry'],'fire_prob':fscore,'highrisk':fscoreHI})

    #Find out which polygon contains a point of interest
    mypoint = Point(plon,plat)
    geoms = list(riskmap['geometry'])
    for i in range(len(geoms)):
        ping = geoms[i].contains(mypoint)
        if ping == True:
            break

    #prob of fire at the entered address
    prob = riskmap['fire_prob'].values[i]
    if prob >= vallim:
        high_risk = 1
    else:
        high_risk = 0


    riskmap['BLOCKID'] = np.arange(len(riskmap))
    gdf_wgs84 = riskmap.copy()
    gdf_wgs84.crs = {'init': 'epsg:4326', 'no_defs': True}

    m = folium.Map(location=[37.76, -122.42],zoom_start=13,
               tiles="CartoDB positron",
              width='100%',
              height='100%')

    #Fill in the high risk zone
    folium.Choropleth(
        name = 'High risk zone',
        geo_data=gdf_wgs84[gdf_wgs84['highrisk']==1].to_json(),
        key_on = 'feature.properties.{}'.format('BLOCKID'),
        max_zoom=16,
        fill_opacity=0.5,
        fill_color='#ff0000',
        line_opacity=0.1,
        highlight=True,
        ).add_to(m)

    #Fill in the risk score
    folium.Choropleth(
        show=False,
        name='Fire risk score',
        geo_data=gdf_wgs84.to_json(),data=riskmap,columns=['BLOCKID','fire_prob'],
        key_on = 'feature.properties.{}'.format('BLOCKID'),
        max_zoom=16,
        fill_opacity=0.8,
        fill_color='YlOrRd',
        line_opacity=0.1,
        highlight=True,
        legend_name='Fire risk score').add_to(m)

    m.add_child(folium.LatLngPopup())

    popup = 'Fire risk score at your location: %.2f' %prob
    folium.Marker([mypoint.y,mypoint.x], 
             popup=popup,
             icon=folium.Icon(color='white', 
                              icon_color='green', 
                              icon='male', 
                              angle=0,
                              prefix='fa')).add_to(m)

    folium.LayerControl().add_to(m)

    m.save('firescapeapp/templates/'+html_map_name)

    return prob, high_risk

def withinSF(lon,lat):

    '''
    Check if a user's address is within the region of interest
    '''

    try:
        if (-122.53 < lon < -122.35 ) and (37.7 < lat < 37.84 ):
            return 1
        else:
            return 0
    except:
        return 0

@app.route('/')
def index():
    return render_template('FireRisk-index.html')

@app.route('/<firetype>/<yearval>')
def displayamap(firetype='structure',yearval='2019'):

    rendertemp = 'FireRisk-map.html'
    foliummap = '%s_%s.html' %(firetype,yearval)

    return render_template(rendertemp,year=yearval,firetype=firetype,mapname=foliummap)

#Need functions here to generate the maps given some user-entered coordinates and 
#to display pages specific to a user's area

@app.route('/address/<firetype>/<yearval>')
def displaymapwithaddress(firetype='structure',yearval='2019'):

    geolocator = Nominatim(user_agent="FirescapeSF")

    #Load SF_yearblocks
    #SF_blocks = gpd.read_file('firescapeapp/models/SFblocks/SF_block_years_2010.shp')

    rendertemp = 'FireRisk-map.html'

    print(rendertemp,firetype,yearval)

    #Eventually want to call Folium map that has been generated for this specific address
    foliummap = '%s_%s_address.html' %(firetype,yearval)

    if firetype == 'structure':
        model = firescapeapp.SFmodel
    elif firetype == 'external':
        model = firescapeapp.EFmodel
    elif firetype == 'vehicle':
        model = firescapeapp.VFmodel

    if yearval == '2019':
        data = firescapeapp.pred2019data
    elif yearval == '2018':
        data = firescapeapp.pred2018data

    #modeltoload = 'firescapeapp/models/Model_RC_'+firetype+'.sav' #Specific model for the fire type
    #datatoload = 'firescapeapp/models/datasets/'+yearval+'_predictfires.csv' #Specific model for the year of fire

    address = request.args.get('myaddress')

    if 'San Francisco' not in address:
        address = address + ' San Francisco'

    #print("You entered address %s" %address)
    location = geolocator.geocode(address)
    try:
        loclat = location.latitude
        loclon = location.longitude
    except:
        return render_template('error404.html')

    if not withinSF(loclon,loclat):
        return render_template('error404.html')

    #Now we generate the appropriate map

    #Load the appropriate model and appropriate data 
    #model = pickle.load(open(modeltoload, 'rb'))
    #data = pd.read_csv(datatoload)

    #Generate the hazard map
    prob, high_risk = generate_hazard_map_html(model,data,firescapeapp.SF_blocks,foliummap,plat=loclat,plon=loclon)

    #convert to string of reasonable accuracy 
    prob = '%.2f' %prob

    return render_template(rendertemp,year=yearval,firetype=firetype,
        mapname=foliummap,high_risk=high_risk,address_flag=1,address=address,riskscore=prob)
