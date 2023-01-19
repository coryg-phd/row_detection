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
from shapely.geometry import Point, LineString
import glob
import math

#input paths

single_model_result = './inputs/14_Dudley_12152022_results_t50.shp'

complete_polygons = './inputs/sample_results_polygons_complete_61983.shp'

complete_centroids = './inputs/sample_results_polygons_complete_61983_centroids.shp'
subset_centroids = './inputs/sample_results_centroids_subset_19521.shp'
mini_centroids = './inputs/sample_results_centroids_subset_1382.shp'

template_path = './inputs/12_CisternNorth_12062022_ortho_2in.tif'
rasterized_path = './outputs/12_CisternNorth_DF_only_t50_2in.tif'

base_path = './inputs/'
new_path = "./outputs/"

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

        out.close()
        return


#convert all tif in folder to jpegs for opencv
def tif_to_jpg(tifdir, jpgdir):
    for infile in os.listdir(tifdir):
        print ("file : " + infile)
        read = cv2.imread(tifdir + infile)
        outfile = infile.split('.')[0] + '.jpg'
        cv2.imwrite(jpgdir+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 200])


#Filter points by categorical variable and create new file for each
def split_categorical_vector(vect):
    gdf = gpd.read_file(vect)
    cohorts = gdf['cohort'].unique()

    print(len(gdf))

    for cohort in cohorts:

        new_gdf = gdf[gdf.cohort == cohort]

        outfilename = f'./outputs/cohorts/{cohort}.shp'

        new_gdf.to_file(outfilename)

#primary Hough Transform to derive angles of majority of rows



#Straight Line Hough Transform
#takes in path to points shapefile 
def straight_HT(points, multiplier = 1):

    if isinstance(points, str):

        gdf_input = gpd.read_file(points)
 
        pts = gdf_input.geometry
        if gdf_input.columns.isin(['cohort']).any():
                cohort = gdf_input['cohort'][0]
        else:
            cohort=''
        print(cohort)
    else:
        print('the path should point to a folder with shapefiles for each cohort')
        return

    pts = pts.to_crs('epsg:26910')

    a = (pts.y * 10 )  - (np.min(pts.y) * 10)
    b = (pts.x * 10 ) - (np.min(pts.x) * 10)

    #create an image from point coordinates
    y_shape = int(np.max(a) - np.min(a))
    x_shape = int(np.max(b) - np.min(b))

    image = np.zeros((y_shape+1, x_shape+1))

    indices = np.stack([a-1,b-1], axis =1).astype(int)

    image[indices[:,0], indices[:,1]] = 1

    # Classic straight-line Hough transform
    #tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180 * multiplier, endpoint=False)
    tested_angles = np.linspace((-np.pi / 12), (np.pi / 12), 30 * multiplier, endpoint=False)
    h, theta, d = hough_line(image, theta=tested_angles)

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

    peak_angles = len(t2)

    pointlist = []
    res_gdf = gpd.GeoDataFrame()

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        
        y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)

        ax[2].plot((0, image.shape[1]), (y0, y1), '-r')


        x0 = (0 / 10 + np.min(pts.x)) 
        x1 = (image.shape[1] / 10) + np.min(pts.x)

        y0 = (y0 / 10 + np.min(pts.y))
        if math.isinf(y0):
            y0 = image.shape[0]
        
        y1 = (y1 / 10 + np.min(pts.y))
        if math.isinf(y1):
            y1 = 0

        p0 = (x0, y0)
        p1 = (x1, y1)

        #print(f'the first value is {p0} and the second is {p1}')

        pointlist.append((p0,p1))

    
    for point_pair in pointlist:
        lineSTR = LineString([Point(point_pair[0]), Point(point_pair[1])])
        gdf = gpd.GeoDataFrame(index=[0], crs="epsg:26910", geometry=[lineSTR])
        res_gdf = res_gdf.append(gdf)

    res_gdf['cohort'] = cohort

    outname = points.split('.')[1].split('/')[-1]

    print(outname)
    
    res_gdf.to_file(f'./outputs/hough/hough_lines_{outname}.shp', crs=26910, driver='ESRI Shapefile')

    #ax[2].plot((x0, x1), (y0, y1), '-b')

    # ax[2].set_xlim((0, image.shape[1]))
    # ax[2].set_ylim((image.shape[0], 0))
    #ax[2].set_axis_off()
    ax[2].set_xlim(0, np.max(b))
    ax[2].set_ylim(0, np.max(a))

    ax[2].set_title('Detected lines = ' + str(peak_angles))

    plt.tight_layout()
    plt.show()

    print('Total number of lines: ' + str(peak_angles))

#centroids = polys_to_centroids(single_model_result, write=True)

#points_to_raster(centroids, template_path, rasterized_path)

#tif_to_jpg(base_path, new_path)

#straight_HT(mini_centroids, multiplier=10)

#straight_HT('./inputs/23_Middle_results_t50_centroids.shp', multiplier=12)

split_categorical_vector('./inputs/14_Dudley_12152022_results_t50_centroids.shp')

# absolute path to search all text files inside a specific folder
path = r'./outputs/cohorts/*.shp'
files = glob.glob(path)

for file in files:
    straight_HT(file, multiplier=50)

path2 = r'./outputs/hough/*.shp'
files2 = glob.glob(path2)

merged = gpd.read_file(files2[0])

for file in files2[1:]:
    next = gpd.read_file(file)
    merged = gpd.pd.concat([merged, next])

merged.to_file('./outputs/hough_lines_merged.shp')

mypath = "my_folder" #Enter your path here

#clear out folders in outputs
for i in ['./outputs/cohorts', './outputs/hough']:
    for root, dirs, files in os.walk(i, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))

        # Add this block to remove folders
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
