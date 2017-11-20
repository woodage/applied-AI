__author__ = 'robbie'
import numpy as np
import math
import itertools
import operator

def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))
    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index
    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]

data = np.genfromtxt ('dataset1.csv',delimiter =';', usecols =[1 ,2 ,3 ,4 ,5 ,6 ,7])
dataV = np.genfromtxt ('validation1.csv',delimiter =';', usecols =[1 ,2 ,3 ,4 ,5 ,6 ,7])
dates = np.genfromtxt ('dataset1.csv',delimiter =';', usecols =[0])
datesV = np.genfromtxt ('validation1.csv',delimiter =';', usecols =[0])
labels = []
labelsV = []
for label in dates:
    if label < 20000301:
        labels.append('winter')
    elif 20000301 <= label < 20000601:
        labels.append('lente')
    elif 20000601 <= label < 20000901:
        labels.append('zomer')
    elif 20000901 <= label < 20001201:
        labels.append('herfst')
    else: # from 01−12 to end of year
        labels.append('winter')
for label in datesV:
    if label < 20000301:
        labelsV.append('winter')
    elif 20000301 <= label < 20000601:
        labelsV.append('lente')
    elif 20000601 <= label < 20000901:
        labelsV.append('zomer')
    elif 20000901 <= label < 20001201:
        labelsV.append('herfst')
    else: # from 01−12 to end of year
        labelsV.append('winter')

def calculateDistance(x):
    s = 0
    for i in x:
        s+= i * i
    return math.sqrt(s)

# (distance, season) dictonary
trainingDic = {}
testPredictionDict = {}

# 1. CALCULATE DISTANCE TRAINING
for i in range(len(data)):
    trainingDic[calculateDistance(data[i])] = labels[i]
# loop each data valiation
for i in range(len(dataV)):
    # check in training set for the most common distance value and see that as neighbor
    nearestNeighbor = min(trainingDic, key=lambda x: abs(x - calculateDistance(dataV[i])))
    # get index when list is sorted.
    matchedIndex = sorted(trainingDic.keys()).index(nearestNeighbor)
    # determine variable k below for testing
    k = 83
    street = [nearestNeighbor]
    # k must be lower then the training list
    if k < len(trainingDic.keys()):
        #check if index has enough left neighbors and enough right neighbors
        if matchedIndex - (k / 2) >= 0 and matchedIndex + (k / 2) <= len(trainingDic.keys()):
            for j in range(1, int((k / 2))):
                street.append(sorted(trainingDic.keys())[matchedIndex + j])
            for j in range(1, int((k / 2)) ):
                street.append(sorted(trainingDic.keys())[matchedIndex - j])
        else:
            #index does not have enough left neighbors
            if matchedIndex - (k / 2) < 0:
                for j in range(1, int((k / 2))):
                    street.append(sorted(trainingDic.keys())[matchedIndex + j])
            #index does not have enough right neighbors
            if matchedIndex + (k / 2) > len(trainingDic.keys()):
                for j in range(1, int((k / 2))):
                    street.append(sorted(trainingDic.keys())[matchedIndex - j])
    #list of seasons from neighbors including most common neighbor
    avse = []
    for dis in street:
        avse.append(trainingDic[dis])
    # add the season of the nearest neighbor with dictonary
    testPredictionDict[calculateDistance(dataV[i])] = most_common(avse)

correct = 0
labelI = 0
#check if correct
for i in testPredictionDict:
    if testPredictionDict[i] == labelsV[labelI]:
        correct+=1
    labelI +=1
print(correct)