#for computing total number of trees, trees per acre, 
#and estimating confidence for accuracy

dir = '~/AeroTract/xmas_trees/NMTF_Fall2022'

#dependencies
import geopandas as gpd
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
import pandas as pd

#load iCreekFiren data
roi_file  = '/ROIs/12_CisternNorth_GF2020.shp'
model_points = '/results/12_CisternNorth_results_manual_clean.shp'
#gr ROI file ino 1 acre cells

def grid_acres (poly, pts):
    extent = gpd.read_file(poly)
    points = gpd.read_file(pts)
    
    extent = extent.to_crs('epsg:26910')
    points = points.to_crs('epsg:26910')
    
    #print(len(points))
    
    points = gpd.clip(points, extent)
    #print(len(points))
    
    name = Path(poly).stem
    bound_points = extent.total_bounds
    xmin, ymin, xmax, ymax = bound_points

    width = (208.710325571113/(3.28084*4))
    height = (208.710325571113/(3.28084*4))

    rows = int(np.ceil((ymax-ymin) /  height))
    cols = int(np.ceil((xmax-xmin) / width))
    XleftOrigin = xmin
    XrightOrigin = xmin + width
    YtopOrigin = ymax
    YbottomOrigin = ymax - height
    polygons = []
    for i in range(cols):
        Ytop = YtopOrigin
        Ybottom =YbottomOrigin
        for j in range(rows):
            polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)])) 
            Ytop = Ytop - height
            Ybottom = Ybottom - height
        XleftOrigin = XleftOrigin + width
        XrightOrigin = XrightOrigin + width

    grid = gpd.GeoDataFrame({'geometry':polygons})
    grid = grid.set_crs('epsg:26910')
    
    intersect = gpd.overlay(extent, grid, how='intersection')
    intersect["acres"] = (intersect['geometry'].area *0.0002473)
    intersect['row_num'] = np.arange(len(intersect))
    intersect['Trees'] = 'Count'
    sum_acres = intersect["acres"].sum(axis = 0, skipna = True)
    extent_acres = (extent["geometry"].area *0.0002473)
    
    total_tpa = len(points) / sum_acres
    
    #Spatial join Points to polygons
    dfsjoin = gpd.sjoin(intersect, points) 
    #print(dfsjoin)
    dfpivot = pd.pivot_table(dfsjoin,index='row_num',columns='Trees',aggfunc={'Trees':len})
    dfpivot.columns = dfpivot.columns.droplevel()
    dfpolynew = intersect.merge(dfpivot, how='left', on='row_num')
    dfpolynew["TPA"] = dfpolynew["Count"] / dfpolynew["acres"]
    
    #print(type(dfpolynew))
    #print(dfpolynew)
    
    #intersect = intersect[intersect['area']>minarea]
    print("The total area in acres of the ROI is:")
    print(sum_acres)
    print("The total number of seedlings located by the model is:")
    print(len(points))
    print("The estimated density (trees per acre) is:")
    print(total_tpa)
    
    #print(intersect["area"])
    #intersect = intersect.to_crs('epsg:3857')
    #intersect.to_file(name + "_grid2.shp")
    dfpolynew.to_file('./grids/' + name + "_grid_TPA.shp")

grid_acres(dir + roi_file, dir+ model_points)


#count total number of trees in ROI by polygon and total

#compute total area of ROI and trees per acre across stand, within grid

#calculate confidence based on site score and % subsample

#print out results