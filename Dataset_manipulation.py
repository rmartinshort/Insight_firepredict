#!/usr/bin/env python

#Tools for routine manipulation of the datasets

from shapely.geometry import Point, Polygon


def convert_to_point(coords):

    coords = coords.split(',')
    lat = float(coords[0][1:])
    lon = float(coords[1][:-1])

    return Point(lon,lat)
