#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:19:42 2018

@author: jmonroe

This script exists to count area of JJ from SEM images
"""


'''
Open CV's modified watershed algorithm: 
        Watershed: given a potential landscape one slowly increases a height threshold.
                As different local minima are surpassed, area on either side is combined.
                Continuing gives a segmentation hierarchy
        CV's modification:
                Do a bit of filtering for "definite signal" and "definite background"
                Enables smoother watershedding (one "flooding event")
'''

#import PIL
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import kmeans

import cv2 as cv


# string of 4 paramp junctions
#data_dir = "/Users/jmonroe/Projects/machineLearning/areaCounting/data/091718_paramp/"
#data_name = "deviceC_JJ10,12.tif"

# single squid:
data_dir = "/Users/jmonroe/Projects/machineLearning/areaCounting/data/"
data_name = "josephsonJunction_subQ4_C3D2.png"

def my_data(show=False):
    ## load data
    raw_image = cv.imread(data_dir+data_name)

    ## cut off SEM label
    label_width = 64
    image_noLabel= raw_image[:-label_width, :]
   
    ## extract a single JJ 
    left,right = 550, 780
    up, down = 220, 425
    single_JJ = image_noLabel[up:down, left:right]
    single_slice = single_JJ[100, :]
    
    if show:
        plt.figure()
        plt.title("No label")
        plt.imshow(image_noLabel)
        plt.figure()
        plt.title("Single JJ")
        plt.imshow(single_JJ)
        plt.figure()
        plt.title("Single Slice")
        plt.plot(single_slice)
    #return single_squid
    return image_noLabel
##END my_data
    

def calc_distance(xy_tuple,centers_list):
    ## make a separate function for extensibility
    dx = abs(centers_list[:, 0] - xy_tuple[0])
    dy = abs(centers_list[:, 1] - xy_tuple[1])    
    #return np.sqrt( dx**2 + dy**2 )
    return dx+dy    
##END calc_distance
    

def main():
    ## load datqa
    #img = cv.imread("water_coins.jpg")
    img = my_data(show=False)
    plt.figure()
    plt.title("img")
    plt.imshow(img)

    # let's look at a small part
    test = img[200:400, 600:800, 0]
    #test = np.mean(img, axis=2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter( np.arange(200), np.arange(200), test)
    #plt.imshow( test )
    plt.title("Cut")
    plt.show()
    return 0;
    

    ## try some convolutions
    sig = 1280*0.02
    gauss = cv.GaussianBlur(img, (11,11), sig)
    plt.figure()
    plt.title(f"Gauss sig={np.round(sig)}")
    plt.imshow(gauss)
    plt.show()

    bilateral = cv.bilateralFilter(img, d=5, sigmaColor=1000, sigmaSpace=1000)
    plt.figure()
    plt.title("bilateral: d=5, sig=1000")
    plt.imshow(bilateral)
    plt.show()
    return 0;
  
    ## some processing 
    gray =  cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plt.figure()
    plt.title("Gray")
    plt.imshow(gray)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    thresh = gray
    '''
    plt.figure()
    plt.title("thresh")
    plt.imshow(thresh)
    '''
    
    ## first estimation of noise
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    plt.figure()
    plt.imshow(opening)
    plt.title("Opening")
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    plt.figure()
    plt.imshow(sure_bg)
    plt.title("sure_bg")
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    plt.figure()
    plt.imshow(sure_fg)
    plt.title("sure_fg")
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    markers = cv.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    
    plt.figure()
    plt.imshow(markers)
    plt.title("Markers")
##
    
if __name__ == '__main__':
    #my_data(True)
    main()
    plt.show()

