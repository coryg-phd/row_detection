import random
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
from skimage.feature import canny
from skimage import data

from shapely.geometry import Point, LineString

from math import *

import matplotlib.pyplot as plt
from matplotlib import cm

# Data Cleaning
shp = gpd.read_file('D:/ChristmasTreesRows/tile_cen_26910.shp')
x_list = list((shp["geometry"].x*10).astype(int))
y_list = list((shp["geometry"].y*10).astype(int))

x_min = min(x_list)
y_min = min(y_list)

norm_x = [x-x_min for x in x_list]
norm_y = [x-y_min for x in y_list]

array_y = np.array(norm_y)
array_x = np.array(norm_x)

x_shape = max(norm_x) - min(norm_x)
y_shape = max(norm_y) - min(norm_y)

im = np.zeros((x_shape+1, y_shape+1))
im_rot = np.rot90(im)
indices = np.stack([array_x-1,array_y-1], axis =1).astype(int)
im_rot[indices[:,1], indices[:,0]] = 1

image = im_rot

theta = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)

# Hough Transform
lines1 = probabilistic_hough_line(image, threshold=5, line_length=30, line_gap=45, theta=theta)
lines2 = probabilistic_hough_line(image, threshold=4, line_length=40, line_gap=35, theta=theta)

'''
lines : list
List of lines identified, lines in format ((x0, y0), (x1, y1)),
indicating line start and end.
'''

def angleThrsh(angle_threshold, lines):
    myList = []
    for line in lines:
        p0, p1 = line
        if p0[0] == p1[0]:
            continue
        
        opp = p1[1] - p0[1]
        adj = p1[0] - p0[0]
        
        # If line is completely flat, cannot measure angle due to dividing by zero
        try:
            angle = degrees(atan(opp/adj))
            tupleX = tuple((x,y) for (x,y) in line if not angle > angle_threshold and not angle < -(angle_threshold))
            if len(tupleX) > 0:
                myList.append(tupleX)
        # Append completely flat line
        except:
            myList.append((p0,p1))
    return myList #List of tuples
    
    
def lines_to_shpfile(lineList, xmin, ymin):
    pointList = []
    res_gdf = gpd.GeoDataFrame()
    for line in lineList:
        p0, p1 = line

        first_x = (p0[0] + xmin)/10
        next_x = (p1[0] + xmin)/10
        first_y = (p0[1] + ymin)/10
        next_y = (p1[1] + ymin)/10
        
        p0 = (first_x, first_y)
        p1 = (next_x, next_y)
        
        pointList.append((p0,p1))
    
    for point_pair in pointList:
        lineSTR = LineString([Point(point_pair[0]), Point(point_pair[1])])
        gdf = gpd.GeoDataFrame(index=[0], crs="epsg:26910", geometry=[lineSTR])
        res_gdf = res_gdf.append(gdf)
    
    res_gdf.to_file('D:/ChristmasTreesRows/hough_output.shp', crs=26910, driver='ESRI Shapefile')
   
    
lineList = angleThrsh(10, lines1)
lines_to_shpfile(lineList, x_min, y_min)
        
        
# Plots
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# ax = axes.ravel()

# ax[0].imshow(image, cmap=cm.gray)
# ax[0].set_title('Input image')

# ax.imshow(image * 0)
# for line in angleThrsh(10, lines1):
    # p0, p1 = line
    # ax.plot((p0[0], p1[0]), (p0[1], p1[1]))
# ax.set_xlim((0, image.shape[1]))
# ax.set_ylim((0, image.shape[0]))
# ax.set_title('Probabilistic Hough')

# ax[1].imshow(image * 0)
# for line in angleThrsh(10, lines2):
    # p0, p1 = line
    # ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
# ax[1].set_xlim((0, image.shape[1]))
# ax[1].set_ylim((image.shape[0], 0))
# ax[1].set_title('Probabilistic Hough')

# for a in ax:
    # a.set_axis_off()

# plt.tight_layout()
# plt.show()
                                 
 
