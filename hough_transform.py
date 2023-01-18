import cv2
import numpy as np
# import os
import geopandas as gpd
# import rasterio
from rasterio import features


vect_path = './results/12_CisternNorth_centroids_subset.shp'
# template_raster_path = './orthos/12_CisternNorth_12062022_ortho_2in.tif'
# rasterized_path = './rasters/gtiff/12_CisternNorth_centroids_2in_mini.tif'

# base_path = './rasters/gtiff/'
# new_path = "./rasters/jpgs/"

treepoints = gpd.read_file(vect_path)

treepoints = treepoints.to_crs('epsg:26910')

treepoints = treepoints.centroid


print(treepoints)

# # treepoints['vals'] = 1

# # rst = rasterio.open(template_raster_path)

# # meta = rst.meta.copy()
# # meta.update(compress='lzw')

# # # with rasterio.open(rasterized_path, 'w+', **meta) as out:
# # #     out_arr = out.read(1)

# # #     # this is where we create a generator of geom, value pairs to use in rasterizing
# # #     shapes = ((geom,value) for geom, value in zip(treepoints.geometry, treepoints.vals))

# # #     burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
# # #     out.write_band(1, burned)

# # # for infile in os.listdir(base_path):
# # #     print ("file : " + infile)
# # #     read = cv2.imread(base_path + infile)
# # #     outfile = infile.split('.')[0] + '.jpg'
# # #     cv2.imwrite(new_path+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 200])

# # img = cv2.imread('./rasters/jpgs/12_CisternNorth_centroids_2in_mini.jpg')

# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # cv2.imshow('iug', img)
# # cv2.waitKey()

# # thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


# # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
# # close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

# # minLineLength = 550
# # maxLineGap = 70
# # lines = cv2.HoughLinesP(close,1,np.pi/180,100,minLineLength,maxLineGap)
# # for line in lines:
# #     for x1,y1,x2,y2 in line:
# #         cv2.line(img,(x1,y1),(x2,y2),(36,255,12),3)

# # cv2.imshow('thresh', thresh)
# # cv2.imshow('close', close)
# # cv2.imshow('image', img)
# # cv2.waitKey()

# # cv2.imwrite('./rasters/jpgs/houghlines5.jpg',img)

# from skimage.transform import (hough_line, hough_line_peaks,
#                                probabilistic_hough_line)
# from skimage.feature import canny
# from skimage import data

# import matplotlib.pyplot as plt
# from matplotlib import cm

# # import random
# # import numpy as np

# a = (treepoints.geometry.y * 10 )  - (np.min(treepoints.geometry.y) * 10)
# b = (treepoints.geometry.x * 10 )- (np.min(treepoints.geometry.x) * 10)

# # a = np.random.randint(1,101,400)  # Random points.
# # b = np.random.randint(1,101,400)  # Random points.

# # for i in range(0, 90, 2):  # A line to detect
# #     a = np.append(a, [i+5])
# #     b = np.append(b, [0.5*i+30])

# plt.plot(b, a, '.')
# plt.show()

# #create an image from list of points
# y_shape = int(np.max(a) - np.min(a))
# x_shape = int(np.max(b) - np.min(b))


# im = np.zeros((y_shape+1, x_shape+1))

# indices = np.stack([a-1,b-1], axis =1).astype(int)

# im[indices[:,0], indices[:,1]] = 1

# plt.xlim(0, np.max(b))
# plt.ylim(0, np.max(a))
# plt.imshow(im)
# plt.show()

# # # Constructing test image
# # #image = np.zeros((100, 100))
# # #idx = np.arange(25, 75)
# # #image[idx[::-1], idx] = 255
# # #image[idx, idx] = 255

# image = im

# # Classic straight-line Hough transform
# tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 5760, endpoint=False)
# h, theta, d = hough_line(image, theta=tested_angles)

# # Generating figure 1
# fig, axes = plt.subplots(1, 3, figsize=(15, 6))
# ax = axes.ravel()

# ax[0].imshow(image, cmap=cm.gray)
# ax[0].set_title('Input image')
# ax[0].set_xlim(0, np.max(b))
# ax[0].set_ylim(0, np.max(a))
# #ax[0].set_axis_off()

# ax[1].imshow(np.log(1 + h),
#              extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
#              cmap=cm.gray, aspect=1/1.5)
# ax[1].set_title('Hough transform')
# ax[1].set_xlabel('Angles (degrees)')
# ax[1].set_ylabel('Distance (pixels)')
# ax[1].axis('image')

# ax[2].imshow(image, cmap=cm.gray)
# for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
#     y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
#     y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
#     ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
# # ax[2].set_xlim((0, image.shape[1]))
# # ax[2].set_ylim((image.shape[0], 0))
# ax[2].set_xlim(0, np.max(b))
# ax[2].set_ylim(0, np.max(a))
# #ax[2].set_axis_off()
# ax[2].set_title('Detected lines')


# plt.tight_layout()
# plt.show()


# # Line finding using the Probabilistic Hough Transform
# image = im
# #edges = canny(image, 2, 1, 25)


# # Generating figure 2
# fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
# ax = axes.ravel()

# lines1 = probabilistic_hough_line(image, threshold=1, line_length=1000,
#                                  line_gap=75)

# ax[0].imshow(im * 0)
# for line in lines1:
#     p0, p1 = line
#     ax[0].plot((p0[0], p1[0]), (p0[1], p1[1]))
# ax[0].set_xlim((0, image.shape[1]))
# ax[0].set_ylim((image.shape[0], 0))
# ax[0].set_title('Probabilistic Hough 1')

# lines2 = probabilistic_hough_line(image, threshold=15, line_length=500,
#                                  line_gap=75)

# ax[1].imshow(im * 0)
# for line in lines1:
#     p0, p1 = line
#     ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
# for line in lines2:
#     p0, p1 = line
#     ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
# ax[1].set_xlim((0, image.shape[1]))
# ax[1].set_ylim((image.shape[0], 0))
# ax[1].set_title('Probabilistic Hough 2')

# lines3 = probabilistic_hough_line(image, threshold=25, line_length=200,
#                                  line_gap=75)

# ax[2].imshow(im * 0)

# for line in lines1:
#     p0, p1 = line
#     ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
# for line in lines2:
#     p0, p1 = line
#     ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
# for line in lines3:
#     p0, p1 = line
#     ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))

# ax[2].set_xlim((0, image.shape[1]))
# ax[2].set_ylim((image.shape[0], 0))
# ax[2].set_title('Probabilistic Hough 3')

# for a in ax:
#     a.set_axis_off()

# plt.tight_layout()
# plt.show()