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

import cv2 as cv


data_dir = "/Users/jmonroe/Projects/machineLearning/areaCounting/data/091718_paramp/"
data_name = "deviceC_JJ10,12.tif"

def my_data(show=False):
    ## load data
    tif_data = cv.imread(data_dir+data_name)

    ## cut off SEM label
    label_width = 64
    tif_noLabel = tif_data[:-label_width, :]
    
    left,right = 660, 760
    up, down = 785, 810
    single_squid = tif_data[up:down, left:left+40]
    single_slice = single_squid[:, 20]
    
    cluster_indices, dist = k_means()
    
    
    if show:
        #plt.figure()
        #plt.imshow(tif_noLabel)
        plt.figure()
        plt.imshow(single_squid)
        plt.figure()
        plt.plot(single_slice)
    return single_squid
##END my_data
    

def calc_distance(xy_tuple,centers_list):
    ## make a separate function for extensibility
    dx = abs(centers_list[:, 0] - xy_tuple[0])
    dy = abs(centers_list[:, 1] - xy_tuple[1])    
    #return np.sqrt( dx**2 + dy**2 )
    return dx+dy    
##END calc_distance
    

def k_means(data, k, show=False):
    '''
    data:   (n, dim) array 
    k:      number of clusters
    '''
    data = np.array(data)
    if len(data.shape)>1:
        dim = data.shape[1]
    else:
        dim = 1
    numPoints = data.shape[0]
    color_list = "rgbcmyk"
    
    num_iter = 6
    centers = np.zeros((k,dim))
    cluster_counts = np.zeros(k)
    cluster_ids_fullList = np.zeros((num_iter+1, numPoints) ,dtype="int")    
    distance_fullList = np.zeros(num_iter+1)
    
    cluster_indices = np.random.randint(0,k,size=numPoints)
    cluster_ids_fullList[0] = cluster_indices

    ## Initial calculations
    # centers
    for j,index in enumerate(cluster_indices):
        centers[index] += data[j]
        cluster_counts[index] += 1
    for k_index in range(k):
        centers[k_index] /= cluster_counts[k_index]

    # figure
    if show:
        fig = plt.figure()
        plt.title("Initial Assignment")
        tot_dist = 0
        for i,(x,y) in enumerate(data):
            plt.scatter(x,y,color=color_list[cluster_indices[i]])
            tot_dist += min(calc_distance((0,0), centers))
        plt.scatter(centers[:, 0], centers[:, 1], marker='x',s=20,color='k')
        distance_fullList[0] = tot_dist
    
    ## k-means assignment
    for i in range(1,num_iter+1):
        ## reassign each point to nearest cluster 
        tot_distance = 0        
        #print(i, centers[0], centers[1])
        for j,(x,y) in enumerate(data):
            distances = calc_distance((x,y), centers)
            new_cluster_index = np.argmin(distances)
            cluster_indices[j] = new_cluster_index
            tot_distance += min(distances)
        ##END data loop
            
        ## define clusters
        cluster_list = [ [] for j in range(k)]
        for j,index in enumerate(cluster_indices):
            cluster_list[index].append(data[j])
        for j in range(k):
            if len(cluster_list[j]): centers[j] = np.mean(cluster_list[j],axis=0)
        #print str(i)+ "\t", centers
        #print cluster_list[1]
        ## track progress
        distance_fullList[i] = tot_distance
        cluster_ids_fullList[i] = cluster_indices
        plt.show()
    ##END iterations
    
    ## iteration-wise plots
    if show:
        for i in range(1,num_iter+1):
            plt.figure()
            plt.title(str(i)+"th iteration")
            for j,(x,y) in enumerate(data):
                plt.scatter(x,y,color=color_list[cluster_ids_fullList[i][j]])
            plt.scatter(centers[:,0], centers[:,1], marker='x',s=20,color='k')
        
    return cluster_ids_fullList, distance_fullList;
##END k_means
    
    
def main():
    img = cv.imread("water_coins.jpg")
    img = my_data(True)
    
    gray =  cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    
    ## first estimation of noise
    
    
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
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
    plt.show()
##
    
if __name__ == '__main__':
    my_data(True)
    #main()

'''
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
'''








