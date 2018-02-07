__author__ = 'robbie'

import numpy as np
from functools import reduce
from collections import Counter
import math

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
def calculateDistance(dataV, training):
    result = 0
    for ele in range(0, len(dataV)):
        delta = (training[ele] - dataV[ele])
        result += delta*delta
    return np.sqrt(result)

# Return the indexes of the nearest founded neighbors from the trainingsData list. Indexes could be used to find a match with labels.
def getNeighbors(trainingData, validationRow, k):
    # k must be higher then 1.
    if k < 1 or k > len(trainingData):
        exit(print("K is not correct."))
    # List with calculated distance.
    distanceList = [calculateDistance(validationRow, training) for training in trainingData]
    # Get neighbor index with smallest distance.
    neighbors = []
    for i in range(0, k):
        neighbors.append(min(distanceList))
        distanceList.remove(min(distanceList))
    # Calculate again. Indexing was not correct anymore because we removed items from it.
    distanceList = [calculateDistance(validationRow, training) for training in trainingData]
    # Return indexes of each nearest neighbor(s).
    return [distanceList.index(neighbor) for neighbor in neighbors]

# Returns most common label bases on the indexesList.
def getMostCommonTrainingsLabel(indexesList, labels):
    # List of found labels.
    labelsFound = [labels[i] for i in indexesList]
    c = Counter(labelsFound)
    # Return the most common label in list.
    return c.most_common(1)[0][0]

def getBestK():
    r = {}
    # Test each K.
    for K in range(1, len(dataV)):
        correct = 0
        # Loop each validation data.
        for i in range(len(dataV)):
            # Get closest neighbor(s).
            neighborsIndexes = getNeighbors(dataT, dataV[i], K)
            # Check if label is a match with validation label.
            if labelsV[i] == getMostCommonTrainingsLabel(neighborsIndexes, labelsT):
                correct += 1
        r[K] = correct
        print("K " + str(K) + " with score of " + str(correct) + "\n")
    for k,v in r.items():
        if v == max(r.values()):
            return k

# Determine K. getBestK()
K = getBestK()
correct = 0
# Loop each validation data.
for i in range(len(dataV)):
    # Get closest neighbor(s).
    neighborsIndexes = getNeighbors(dataT, dataV[i], K)
    # Get most common label from founded neighbor(s).
    # Check if label is a match with validation label.
    if labelsV[i] == getMostCommonTrainingsLabel(neighborsIndexes, labelsT):
        correct += 1
# Calculated K.
print(K)
# Amount correct.
print(correct)
# Amount of validation data.
print(len(dataV))