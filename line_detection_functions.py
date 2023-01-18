#dependencies
import geopandas as gpd
import rasterio
from rasterio import features
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)

#input paths

vect_path = './results/12_CisternNorth_DF_only_t50.shp'
points_path = './results/12_CisternNorth_DF_only_t50_centroids.shp'

template_path = './orthos/12_CisternNorth_12062022_ortho_2in.tif'
rasterized_path = './rasters/gtiff/12_CisternNorth_DF_only_t50_2in.tif'

base_path = './rasters/gtiff/'
new_path = "./rasters/jpgs/"

#change crs to utm for measuring distances and compute centroids
def polys_to_centroids(vect, write = False):
    trees = gpd.read_file(vect)

    trees = trees.to_crs('epsg:26910')
    #append ones for rasterizing points later
    trees['vals'] = 1

    treepoints = trees.copy()
    treepoints['geometry'] = trees['geometry'].centroid
    print('completed centroid creation')

    if write:
        treepoints.to_file('.' + vect.split('.')[1] + '_centroids.shp')

    return treepoints



#rasterize points using ortho for reference  
#points are points gdf and rasters are full file paths
def points_to_raster(points, inraster, outpath):

    rst = rasterio.open(inraster)

    meta = rst.meta.copy()
    meta.update(compress='lzw')

    with rasterio.open(outpath, 'w+', **meta) as out:

        out_arr = out.read(1)

        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = ((geom,value) for geom, value in zip(points.geometry, points.vals))

        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write_band(1, burned)
        out.write_band(2, burned)
        out.write_band(3, burned)

        out.close()
        return



#convert all tif in folder to jpegs for opencv
def tif_to_jpg(tifdir, jpgdir):
    for infile in os.listdir(tifdir):
        print ("file : " + infile)
        read = cv2.imread(tifdir + infile)
        outfile = infile.split('.')[0] + '.jpg'
        cv2.imwrite(jpgdir+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 200])



#Straight Line Hough Transform
#takes in filepath to points shapefile UTM 

def straight_HT(pointsdir, multiplier = 1):

    pts = gpd.read_file(pointsdir).geometry

    pts = pts.to_crs('epsg:26910')

    a = (pts.y * 10 )  - (np.min(pts.y) * 10)
    b = (pts.x * 10 ) - (np.min(pts.x) * 10)

    #create an image from point coordinates
    y_shape = int(np.max(a) - np.min(a))
    x_shape = int(np.max(b) - np.min(b))

    image = np.zeros((y_shape+1, x_shape+1))

    indices = np.stack([a-1,b-1], axis =1).astype(int)

    image[indices[:,0], indices[:,1]] = 1

    # plt.xlim(0, np.max(b))
    # plt.ylim(0, np.max(a))
    # plt.imshow(image)
    # plt.show()

    # Classic straight-line Hough transform
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180 * multiplier, endpoint=False)
    h, theta, d = hough_line(image, theta=tested_angles)

    i = len(h)
    k=0
    l=0

    #print(len(h[0]))
    for line in h:
        if sum(line) == 0:
            k = k + 1
        if sum(line) > 0:
            # print('index = ' + str(k))
            # print('sum = ' + str(sum(line)))
            # print('distance = ' + str(d[k]))
            l = l + 1
    
    # print(f'Total number of angles tested: ' + str(len(theta)))



    # print('number of zero arrays: ' + str(k))
    # print('number of non-zero arrays: ' + str(l))
    # print('length of distances: ' + str(len(d)))


    #print(d[0])

    # Generating figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_xlim(0, np.max(b))
    ax[0].set_ylim(0, np.max(a))
    #ax[0].set_axis_off()

    ax[1].imshow(np.log(1 + h),
                 #extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                 cmap=cm.gray, aspect='auto')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(image, cmap=cm.gray)

    h2, t2, d2 = hough_line_peaks(h, theta, d)

    print(t2)

    peak_angles = len(t2)

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
        ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
    
    
    
    # ax[2].set_xlim((0, image.shape[1]))
    # ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_xlim(0, np.max(b))
    ax[2].set_ylim(0, np.max(a))
    #ax[2].set_axis_off()
    ax[2].set_title('Detected lines = ' + str(peak_angles))

    plt.tight_layout()
    plt.show()

    print(f'Total number of lines (rows): ' + str(peak_angles))

#centroids = polys_to_centroids(vect_path, write=False)

#points_to_raster(centroids, template_path, rasterized_path)

#tif_to_jpg(base_path, new_path)

straight_HT('./results/12_CisternNorth_centroids_subset_mini.shp', multiplier=10)

#straight_HT('./results/12_CisternNorth_DF_only_t50_centroids.shp', multiplier=12)

