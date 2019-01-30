#!/usr/bin/env python

from flask import Flask, render_template, request
from geopy.geocoders import Nominatim
import pickle
import geopandas as gpd
import pandas as pd
import numpy as np

import folium
from branca.utilities import split_six
from shapely.geometry import Point


app = Flask(__name__)

geolocator = Nominatim(user_agent="FirescapeSF")


#Load SF_yearblocks
SF_blocks = gpd.read_file('models/SFblocks/SF_block_years_2010.shp')


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
    X.drop(['GISYEARJOIN','SF_pred','VF_pred','EF_pred'],axis=1,inplace=True)
    mapgeom = mapdata[mapdata['GISYEARJOI'].isin(GISCELLS)]
    fires_holdout_predict = model.predict_proba(X)

    fscore = 10.0*fires_holdout_predict[:,1]

    riskmap = gpd.GeoDataFrame({'geometry':mapgeom['geometry'],'fire_prob':fscore})

    #Find out which polygon contains a point of interest
    mypoint = Point(plon,plat)
    geoms = list(riskmap['geometry'])
    for i in range(len(geoms)):
        ping = geoms[i].contains(mypoint)
        if ping == True:
            break

    #prob of fire at the entered address
    prob = riskmap['fire_prob'].values[i]

    if firetype == 'structure':
        if prob > 0.954:
            high_risk = 1
        else:
            high_risk = 0

    elif firetype == 'vehicle':
        if prob > 0.779:
            high_risk = 1
        else:
            high_risk = 0

    elif firetype == 'external':
        if prob > 1.80:
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

    m.choropleth(geo_data=gdf_wgs84.to_json(),data=riskmap,columns=['BLOCKID','fire_prob'],
             key_on = 'feature.properties.{}'.format('BLOCKID'),
             max_zoom=16,
             fill_opacity=0.8,
             fill_color='OrRd',
             line_opacity=0.1,
             highlight=True,
             legend_name='Fire risk score',
             )

    m.add_child(folium.LatLngPopup())

    popup = 'Fire risk score at your location: %.2f' %prob
    folium.Marker([mypoint.y,mypoint.x], 
             popup=popup,
             icon=folium.Icon(color='white', 
                              icon_color='green', 
                              icon='male', 
                              angle=0,
                              prefix='fa')).add_to(m)

    m.save('templates/'+html_map_name)

    return prob, high_risk


def withinSF(lon,lat):

    '''
    Check if a user's address is within the region of interest
    '''

    if (-122.53 < lon < -122.35 ) and (37.7 < lat < 37.84 ):

        return 1

    else:

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

    rendertemp = 'FireRisk-map.html'

    print(rendertemp,firetype,yearval)

    #Eventually want to call Folium map that has been generated for this specific address
    foliummap = '%s_%s_address.html' %(firetype,yearval)

    modeltoload = 'models/Model_RC_'+firetype+'.sav' #Specific model for the fire type
    datatoload = 'models/datasets/'+yearval+'_predictfires.csv' #Specific model for the year of fire

    address = request.args.get('myaddress')

    if 'San Francisco' not in address:
        address = address + ' San Francisco'

    #print("You entered address %s" %address)
    location = geolocator.geocode(address)
    loclat = location.latitude
    loclon = location.longitude

    #Now we generate the appropriate map

    #Load the appropriate model and appropriate data 
    model = pickle.load(open(modeltoload, 'rb'))
    data = pd.read_csv(datatoload)

    #Generate the hazard map
    prob, high_risk = generate_hazard_map_html(model,data,SF_blocks,foliummap,plat=loclat,plon=loclon)

    #convert to string of reasonable accuracy 
    prob = '%.2f' %prob


    return render_template(rendertemp,year=yearval,firetype=firetype,
        mapname=foliummap,high_risk=high_risk,address_flag=1,address=address,riskscore=prob)



if __name__ == '__main__':
    app.run(debug=True)
