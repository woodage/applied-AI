__author__ = 'robbie'
# Get iris data from the file.
def irisFileData():
    with open("irisDataSet.txt") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        content = [x.split(',') for x in content]
        return content
# Get all the data.
irisData = [x[:4] for x in irisFileData()]
# Get all labels.
irislabels = [x[4] for x in irisFileData()]

# Select some trainigsData.
trainingsData = irisData[0: 40] + irisData[50:90] + irisData[100:140]
# Select the corresponding labels.
trainingsLabels = irislabels[0: 40] + irislabels[50:90] + irislabels[100:140]
# Select some validationData.
validationData = irisData[40: 50] + irisData[90:100] + irisData[140:150]
# Select the corresponding labels.
validationLabels = irislabels[40: 50] + irislabels[90:100] + irislabels[140:150]

# Cost function that use mean squared error.
# param : cW = currentWeights (vector)
# param : eW = expectedWeights (vector)
def meanSquaredError(cW, fW):
    v = [(cW[i] - fW[i]) * (cW[i] - fW[i]) for i in range(len(cW))]
    return (1 / 2 * len(fW)) * sum(v)

# Testing the cost function.
print(meanSquaredError([1,1,1], [1,1,1]))