from numpy import *
import random

class Neuron:

    def __init__(self, weights, bias):
        self.bias = bias
        self.weights = weights
        self.activation = 0
        self.summedInput = 0

    def getWeights(self):
        return self.weights

    def weightMultiplier(self, input):
        ele = 0
        result = []
        for each in self.weights:
            result.append(input[ele] * each)
            ele += 1

        return result


    def updateWeights(self, weights):
        self.weights = weights[:]

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def act(self, input):
        res = self.sigmoid(sum(self.weightMultiplier(input)) )
        self.summedInput = sum(self.weightMultiplier(input)) + self.bias
        self.activation = res  #bias might not work
        return  res

    def getAct(self):
        return self.activation

    def getSum(self):
        return self.summedInput

class Network:

    def __init__(self, it):
        self.iterations = it
        self.nOUT1 = Neuron([random.uniform(0.7, 0.9), random.uniform(0.7, 0.9)], 1)
        self.nHID1 = Neuron([random.uniform(0.7, 0.9), random.uniform(0.7, 0.9)], 1)
        self.nHID2 = Neuron([random.uniform(0.7, 0.9), random.uniform(0.7, 0.9)], 1)

        # exor, and
        self.neuronList = [[self.nOUT1 , "n1 "] , [self.nHID1, "n2 "], [self.nHID2 , "n3 "]]
        self.trainingSet = [[0, 0], [0, 1], [1, 0,], [1, 1]]
        self.trainingAnswer = [0, 1, 1, 0]

    def calculateDistance(self, a, b):
        # calcululate difference between two classes represented by two lists
        result = 0
        for ele in range(0, len(a)):
            delta = (b[ele] - a[ele])
            result += delta * delta
        return sqrt(result)

    def dist(self, a):
        tmp = 0
        for i in a:
            tmp += i*i
        res = sqrt(tmp )
        return res

    def showWeights(self):
        for neuron in self.neuronList:
            print(neuron[1] ,neuron[0].getWeights())

    def train(self):
        print("training")
        print()

        for x in range(0, self.iterations):
            print("-"*25, x , "-"*25)
            i = 0
            cumError = 0

            for data in self.trainingSet:
                #self.showWeights()
                #print()
                #print("desired result = ", self.trainingAnswer[i])

                result = self.think(data)
                print("result = ", result)
                distError = math.pow(self.trainingAnswer[i] - result, 2)
                cumError += distError


                activationA = data[0]
                activationB = data[1]

                lR = 0.1

                errorOUT =    (self.trainingAnswer[i] - self.nOUT1.getAct() ) * ( 1 - self.nOUT1.getAct()) * self.nOUT1.getAct()

                summError1 =  errorOUT * self.nOUT1.getWeights()[0] *  (1 - self.nHID1.getAct()) * self.nHID1.getAct()
                summError2 =  errorOUT * self.nOUT1.getWeights()[1] *  (1 - self.nHID2.getAct()) * self.nHID2.getAct()
                #act
                w1 = self.nOUT1.getWeights()[0] + (lR * self.nHID1.getAct() * errorOUT)
                w2 = self.nOUT1.getWeights()[1] + (lR * self.nHID2.getAct() * errorOUT)

                w3 = self.nHID1.getWeights()[0] + (lR * activationA * summError1)
                w4 = self.nHID1.getWeights()[1] + (lR * activationB * summError1)

                w5 = self.nHID2.getWeights()[0] + (lR * activationA * summError2)
                w6 = self.nHID2.getWeights()[1] + (lR * activationB * summError2)

                self.nOUT1.updateWeights([w1, w2])
                self.nHID1.updateWeights([w3, w4])
                self.nHID2.updateWeights([w5, w6])


                i += 1


                        #0.03 for lr 0.01
            if cumError < 0.0005:
                print("cum error is low")
                return True

            print("cumError(total) = ", cumError)

        print("done traing")

    def think(self,input):

        result = self.nOUT1.act([self.nHID1.act([input[0],input[1]]), self.nHID2.act([input[0],input[1]])  ] )

        return result



a = Network(300000000)

a.showWeights()
a.train()
a.showWeights()
print(a.think([0,0]))#
print()
print(a.think([1,1]))
print()
print(a.think([0,1]))
print()
print(a.think([1,0]))
print()

# result =  0.00325848222347
# result =  0.988683269548
# result =  0.988683263576
# result =  0.0152723778513
# cumError(total) =  0.0005000001430416899
# ------------------------- 3035833 -------------------------
# result =  0.00325848153307
# result =  0.988683271443
# result =  0.988683265471
# result =  0.0152723753055
# cum error is low
# n1  [-57.927244505673208, 46.480783725380874]
# n2  [0.97001617443763311, 0.9700142058858463]
# n3  [8.8983126664996242, 8.8978592552527207]
# 0.00325848084266
#
# 0.015243805359
#
# 0.988683283551
#
# 0.988683061456
