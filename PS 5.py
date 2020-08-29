#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math


# In[2]:


#euclidian distance
def distance(v1,v2):
    return np.sqrt(np.dot(v1, v1) - 2*np.dot(v1, v2) + np.dot(v2, v2)) 

#is it only single element?
def is_leaf(cluster):
    return len(cluster) == 1

#find children of a cluster
def get_children(cluster):
    if is_leaf(cluster):
        raise TypeError("a leaf cluster has no children")
    else:
        return cluster[1]

#return all values in a cluster
def get_values(cluster):
    if is_leaf(cluster):
        return cluster
    else:
        return [value
                for child in get_children(cluster)
                for value in get_values(child)]

#find minimum distances between two clusters
def cluster_distance(cluster1, cluster2, distance_agg=min):
    return distance_agg([distance(input1, input2)
                         for input1 in get_values(cluster1)
                         for input2 in get_values(cluster2)])


# In[3]:


#agglomerative single-linkage hierarchical clustering
def hierarchical_cluster(data, distance_agg=min):
    clusters = [(input,) for input in data]
    def pair_distance(pair) -> float:
        dist = cluster_distance(pair[0], pair[1], distance_agg)
        #print('distance = ',dist, pair[0], pair[1])
        return cluster_distance(pair[0], pair[1], distance_agg)
    
    while len(clusters) > 1:  #till there is only one cluster left
        c1, c2 = min([(cluster1, cluster2) #two closest clusters
                      for i, cluster1 in enumerate(clusters)
                      for cluster2 in clusters[:i]],key=pair_distance)
        #print("cluster 1 =",c1, " cluster 2 =",c2)
        clusters = [c for c in clusters if c != c1 and c != c2] #remaining clusters
        merged_cluster = (len(clusters), [c1, c2])
        clusters.append(merged_cluster) #add to cluster
    return clusters[0]


# In[4]:


#with my input data
data = [[14,6],[13,13],[-25,2],[-19,-11],[-9,-16],[21,27],[0,15],[26,13],[50,5],[-34,-1],[11,15],[-49,3],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,10],[-11,-6],[-30,-9],[-18,-3]]

base_cluster = hierarchical_cluster(data)
base_cluster
#the output shows the index of the data-point and the cluster that is formed, in total there are 18 data-points


# In[5]:


#with random normal data from 3 distributions
x1 = np.random.normal(loc=np.random.uniform(size=(2,))*10-5,size=(10,2))
x2 = np.random.normal(loc=np.random.uniform(size=(2,))*10-5,size=(10,2))
x3 = np.random.normal(loc=np.random.uniform(size=(2,))*10-5,size=(10,2))
x = np.vstack((x1,x2,x3))
shuffle = np.random.permutation(x.shape[0])
x = x[shuffle,:]
data=x.tolist()


# In[6]:


#the output shows a individual data points which are clustered to form a single cluster in the end along with the index of that data point
base_cluster = hierarchical_cluster(data)
base_cluster


# In[ ]:




