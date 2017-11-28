__author__ = 'robbie'

import numpy as np
import random

dataT = np.genfromtxt ('dataset1.csv',delimiter =';', usecols =[1 ,2 ,3 ,4 ,5 ,6 ,7])
for ele in dataT:
    if ele[6] == -1:
        ele[6] = 0
    if ele[4] == -1:
        ele[0] = 0
datesT = np.genfromtxt ('dataset1.csv',delimiter =';', usecols =[0])
labelsT = []
for label in datesT:
    if label < 20000301:
        labelsT.append('winter')
    elif 20000301 <= label < 20000601:
        labelsT.append('lente')
    elif 20000601 <= label < 20000901:
        labelsT.append('zomer')
    elif 20000901 <= label < 20001201:
        labelsT.append('herfst')
    else: # from 01−12 to end of year
        labelsT.append('winter')

dataV = np.genfromtxt ('validation1.csv',delimiter =';', usecols =[1 ,2 ,3 ,4 ,5 ,6 ,7])
for ele in dataV:
    if ele[6] == -1:
        ele[6] = 0
    if ele[4] == -1:
        ele[0] = 0
datesV = np.genfromtxt ('validation1.csv',delimiter =';', usecols =[0])
labelsV = []
for label in datesV:
    if label < 20010301:
        labelsV.append('winter')
    elif 20010301 <= label < 20010601:
        labelsV.append('lente')
    elif 20010601 <= label < 20010901:
        labelsV.append('zomer')
    elif 20010901 <= label < 20011201:
        labelsV.append('herfst')
    else: # from 01−12 to end of year
        labelsV.append('winter')

# Calculate distance between to rows.
def calculateDistance(instance1, instance2):
    result = 0
    for ele in range(0, len(instance1)):
        delta = (instance2[ele] - instance1[ele])
        result += delta*delta
    return np.sqrt(result)

class Cluster:
    def __init__(self):
        self.nearDataT = []
        self.centroid = []
    def add(self, dataTrow):
        self.nearDataT.append(dataTrow)

# Returns Cluster(s) depending on the given centroids and K
def KmeansClustering(centroids, K):
    # Given the centroids in parameters because we want to change the centroids each time.
    def getNewCentroids(centroids):
        for i in range(len(dataT)):
            # Create a list that will contain distances from dataT to each centroid.
            centroidDistances = []
            for centroid in centroids:
                centroidDistances.append(calculateDistance(dataT[i], centroid))
            # Find nearest centroid distance and assign to cluster.
            clusters[centroidDistances.index(min(centroidDistances))].centroid = centroids[centroidDistances.index(min(centroidDistances))]
            clusters[centroidDistances.index(min(centroidDistances))].add(dataT[i])
        # New possible centroids.
        newCentroids = []
        for cluster in clusters:
            #  Calculate cluster average data row. Next we append in a list for a new centroid.
            newCentroids.append([sum(x) / len(cluster.nearDataT) for x in list(zip(*cluster.nearDataT))])
        return newCentroids
    # List of clusters.
    clusters = [Cluster() for _ in range(K)]
    # Variable to control the While loop.
    canClusterBeChanged = True
    # Stop when non of the cluster assignments change.
    while canClusterBeChanged:
        canClusterBeChanged = False
        for i in range(len(centroids)):
            # We need to calculate the centroids again when they are not equal.
            if [list(map(int, centroid)) for centroid in centroids][i] != [list(map(int, centroid)) for centroid in getNewCentroids(centroids)][i]:
                centroids = getNewCentroids(centroids)
                canClusterBeChanged = True
    return clusters

# Amount of clusters.
K = 2
# List of centroids we will use. The amount of elements is based on K. Indexes can be random.
beginCentroids = [dataV[24], dataV[60]]
for finalCluster in KmeansClustering(beginCentroids, K):
    print(finalCluster.centroid)