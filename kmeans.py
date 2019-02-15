import numpy as np
import matplotlib.pyplot as plt

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
