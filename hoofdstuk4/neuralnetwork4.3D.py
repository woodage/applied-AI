from numpy import *
import random
import numpy as np


data   = np.genfromtxt('irisDataset.csv' , delimiter= ',', usecols=[0,1,2,3])
dataLabels = []
for x in range(0,45):
    dataLabels.append([1, 0, 0,])
for x in range(45,90):
    dataLabels.append([0, 1, 0,])
for x in range(90,135):
    dataLabels.append([0, 0, 1,])

test   = np.genfromtxt('validationIris.csv' , delimiter= ',', usecols=[0,1,2,3])
testLabels = []
for x in range(0,5):
    testLabels.append([1, 0, 0,])
for x in range(5,10):
    testLabels.append([0, 1, 0,])
for x in range(10,15):
    testLabels.append([0, 0, 1,])

class Neuron:

    def __init__(self, weights, id):
        a = 0.2
        b = 0.7
        self.id = id
        self.weights = [random.uniform(a, b) for x in range(0 , weights)]
        self.activation = 0
        self.summedInput = 0
        self.error = 0

    def getWeights(self):
        return self.weights

    def weightMultiplier(self, input):
        ele = 0
        result = []
        for each in self.weights:
            result.append(input[ele] * each)
            ele += 1

        return result

    def sigmoid(self, x):
        #sigmoid activation function
        return 1 / (1 + exp(-x))

    def act(self, input):
        # calculate neuron response to it summed inputs and its weights
        res = self.sigmoid(sum(self.weightMultiplier(input)))
        self.summedInput = sum(self.weightMultiplier(input))
        self.activation = res
        return  res

    def getAct(self):
        return self.activation

    def getSum(self):
        return self.summedInput

    def sDerivative(self, x):
        #derivative of the sigmoid function
        return x * (1 - x)

    def getId(self):
        return self.id

    def getError(self):
        return self.error

    def calcErrorOutput(self, answer):
        # weighted derivative of an ooutput neuron
        self.error = (answer[self.id] - self.activation ) * self.sDerivative(self.activation)

    def calcErrorHidden(self, nodes):
        self.error = 0
        #weighted partial derivative
        for node in nodes:
            self.error += (node.getError() * node.getWeights()[self.id] * self.sDerivative(self.activation) )

    def setWeightsOuterAndHiddenLayer(self, rate, nodes):
        #set the weights of an outer and hiddenlayer
        for node in nodes:
            self.weights[node.getId()] += rate * (node.getAct() * self.error)

    def setWeightsInputLayer(self, rate, input):
        #set the weight of neuron connected to the inputs
        index = 0
        for val in input:
            self.weights[index] += rate * (val * self.error)
            index+= 1

class Network:

    def __init__(self, it):
        self.iterations = it
        self.learnRate = 0.4
        self.errorLimit = 0.1

        # self.nOUT1 = Neuron(6, 0)
        # self.nOUT2 = Neuron(6, 1)
        # self.nOUT3 = Neuron(6, 2)

        # self.nHID1 = Neuron(6, 0)
        # self.nHID2 = Neuron(6, 1)
        # self.nHID3 = Neuron(6, 2)
        # self.nHID4 = Neuron(6, 3)
        # self.nHID5 = Neuron(6, 4)
        # self.nHID6 = Neuron(6, 5)

        # self.nHID7 = Neuron(4, 0)
        # self.nHID8 = Neuron(4, 1)
        # self.nHID9 = Neuron(4, 2)
        # self.nHID10 = Neuron(4, 3)
        # self.nHID11 = Neuron(4, 4)
        # self.nHID12 = Neuron(4, 5)

        # self.nHID13 = Neuron(6, 0)
        # self.nHID14 = Neuron(6, 1)
        # self.nHID15 = Neuron(6, 2)
        # self.nHID16 = Neuron(6, 3)
        # self.nHID17 = Neuron(6, 4)
        # self.nHID18 = Neuron(6, 5)



        #network layer index[0] is firsts layer index[1] is second layer and so on and  index[-1] = outputlayer
        self.networkLayers = [[Neuron(4, 0), Neuron(4, 1), Neuron(4, 2), Neuron(4, 3), Neuron(4, 4), Neuron(4, 5)],
                              [Neuron(6, 0), Neuron(6, 1), Neuron(6, 2), Neuron(6, 3), Neuron(6, 4), Neuron(6, 5)],
                              #[Neuron(4, 0), Neuron(4, 1), Neuron(4, 2)],
                              [Neuron(6, 0), Neuron(6, 1), Neuron(6, 2)]
                              ]

        self.trainingSet = data
        self.trainingAnswers = dataLabels

    def calculateDistance(self, a, b):
        # calcululate difference between two classes represented by two lists
        result = 0
        for ele in range(0, len(a)):
            delta = (b[ele] - a[ele])
            result += delta * delta
        return sqrt(result)

    def lenOfVector(self, a):
        #return the magnitude of a vector
        tmp = 0
        for i in a:
            tmp += i*i
        res = sqrt(tmp )
        return res

    def showWeights(self):
        layerId = 0
        for layer in self.networkLayers:
            print("printing layer  = ", layerId)
            layerId += 1
            for neuron in layer:
                print("neuron with Id ", neuron.getId(), " has weights -> " , neuron.getWeights())

    def sDerivative(self, x):
        return x * (1 - x)

    def forwardPropagateFirstLayer(self,  input):
        for node in self.networkLayers[0]:
            node.act([ele for ele in input])

    def forwardPropagateLayer(self, layer, prevLayer):
        # forward propagate each node in the layer
        for node in layer:
            node.act( [prev.activation for prev in prevLayer  ]  )

    def forwardPropagateLayers(self):
        # forwardpropagate 2  connected layers
        for i in range(0, len(self.networkLayers)-1):
            self.forwardPropagateLayer(self.networkLayers[i + 1], self.networkLayers[i])

    def forwardPropagateNetwork(self, input):
        # forard propagate the network starting with the input layer
        self.forwardPropagateFirstLayer(input)
        self.forwardPropagateLayers()
        #return the activation of each node in the output layer
        return [n.activation for n in self.networkLayers[-1]]

    def backPropagateOutputLayer(self, ans):
        # backpropagate each node in the outputlayer
        for node in self.networkLayers[-1]:
            node.calcErrorOutput( ans)

    def backPropagateNodes(self, layer, prevLayer):
        #backpropagate each node in the layer
        for node in layer:
            node.calcErrorHidden(prevLayer)

    def backPropagateLayers(self):
        #backpropagate 2  connected layers
        for i in range(len(self.networkLayers) -1 , 0 , -1):
            self.backPropagateNodes(self.networkLayers[i - 1], self.networkLayers[i])

    def backPropagateNetwork(self, i):
        #back propagate the output layer and each following layer
        self.backPropagateOutputLayer(self.trainingAnswers[i])
        self.backPropagateLayers()

    def updateWeights(self, input):
        #update the weight for each node in each layer
        for node in self.networkLayers[0]:
            node.setWeightsInputLayer(self.learnRate, input)

        for x in range(1, len(self.networkLayers)):
            for node in self.networkLayers[x]:
                node.setWeightsOuterAndHiddenLayer(self.learnRate, self.networkLayers[x - 1])

    def calculateError(self, i , res):
        return math.pow(self.lenOfVector(self.trainingAnswers[i]) - self.lenOfVector(res), 2)

    def think(self,input):
        return self.forwardPropagateNetwork(input)

    def train(self):
        print("training")
        for x in range(0, self.iterations):
            print("-"*25, " epoch ", x , "-"*25)
            # self.showWeights()
            exampleCounter = 0
            cumulativeError = 0

            for data in self.trainingSet:

                result = self.forwardPropagateNetwork(data)
                self.backPropagateNetwork(exampleCounter)
                self.updateWeights(data)

                cumulativeError += self.calculateError(exampleCounter, result)
                exampleCounter += 1

            if cumulativeError < self.errorLimit:
                return True

            print("cumulativeError(total) = ", cumulativeError)
        print("maximum training iterations passed")




def validateAnswer(a , b):
    #check if 2 lists have the same values
    for x in range(0, len(a)):
        if a[x] == b[x]:
            continue
        else:
            return False
    return True

def validateNeuralnetwork(network):
    correctPredictions = 0
    for i in range(0, 15):
        print("starting on example ", i )
        result = network.think(test[i])
        #print("aproximated result of ", test[i], " = " ,result ,)

        # round the result to 0 or 1
        for x in range(0, len(result)):
            if result[x] <= 0.5:
                result[x] = 0
            else:
                result[x] = 1

        if validateAnswer(result, testLabels[i] ) == True:
            correctPredictions += 1

        print(" rounded result = ", result)
        print(" correct result = ", testLabels[i] )
    print("classified correctly ", ((correctPredictions / len(testLabels)) * 100), "%")
# if the network hangs on a local minumum restart or change the random range in the neuron init weights
network = Network(3000000)
network.train()
network.showWeights()
validateNeuralnetwork(network)

# -------------------------  epoch
# 861 - ------------------------
# printing
# layer = 0
# neuron
# with Id  0  has weights ->[0.98865225753863928, 0.50225218395898219, 0.58872152272871137, 0.33696538892014005]
# neuron
# with Id  1  has weights ->[-11.919113341363543, -18.743516035918788, 20.991252873805003, 16.832275843169054]
# neuron
# with Id  2  has weights ->[0.8409018338825851, 0.58113770281011001, 0.77886813625408591, 0.35129680992541612]
# neuron
# with Id  3  has weights ->[0.77972055376859983, 0.60521821546177701, 0.76885041398905685, 0.55545127614285561]
# neuron
# with Id  4  has weights ->[-5.5098784477036817, -2.1144275053017467, 9.2319249098361738, 7.3752432958923926]
# neuron
# with Id  5  has weights ->[-8.3344161727092363, -6.5468892505265117, 10.773676227186089, 9.7202295985245168]
# printing
# layer = 1
# neuron
# with Id  0  has weights ->[3.5080386388900324, -6.5348639231486558, 3.6293038373580129, 3.5445344758354831, -3.9709783446560958, -5.5867127706751898]
# neuron
# with Id  1  has weights ->[0.39063976655527005, -1.7512730496074727, 0.44154463174746905, 0.1145975312098526, -4.594941977727947, -4.244518573752579]
# neuron
# with Id  2  has weights ->[-1.7061115147411654, 4.7114672554734671, -1.6042140851296574, -1.5496013132436399, 3.4690664838726146, 3.4202280956929201]
# neuron
# with Id  3  has weights ->[0.46767398370096591, -2.4603099382758948, 0.66130650349258402, 0.64523244614244979, -5.9315338903664596, -5.6100764574565254]
# neuron
# with Id  4  has weights ->[-0.56821792332354515, -0.62615092890908963, -0.35568002532619464, -0.52371359855078181, -1.5327591876687476, -2.0576132647390426]
# neuron
# with Id  5  has weights ->[-0.1768490133930514, -0.755044535432006, -0.15292229552888228, -0.38488025079007243, -1.7620912815572882, -1.1655375679300981]
# printing
# layer = 2
# neuron
# with Id  0  has weights ->[-2.7916255722711378, 3.3751938567847222, -8.2896072238786473, 6.3402327655894499, 0.022308310123848908, 0.2124857376885701]
# neuron
# with Id  1  has weights ->[5.1749125353735659, -4.6823289467930387, -5.421398806152113, -6.6882628375302202, -2.1061092091499058, -1.6328183801710816]
# neuron
# with Id  2  has weights ->[-4.5717637119288899, -1.6943297162333102, 5.1529107402312402, -3.0703132877143275, -1.4848487423610677, -1.4721260530908187]
# starting
# on
# example
# 0
# rounded
# result = [1, 0, 0]
# correct
# result = [1, 0, 0]
# starting
# on
# example
# 1
# rounded
# result = [1, 0, 0]
# correct
# result = [1, 0, 0]
# starting
# on
# example
# 2
# rounded
# result = [1, 0, 0]
# correct
# result = [1, 0, 0]
# starting
# on
# example
# 3
# rounded
# result = [1, 0, 0]
# correct
# result = [1, 0, 0]
# starting
# on
# example
# 4
# rounded
# result = [1, 0, 0]
# correct
# result = [1, 0, 0]
# starting
# on
# example
# 5
# rounded
# result = [0, 1, 0]
# correct
# result = [0, 1, 0]
# starting
# on
# example
# 6
# rounded
# result = [0, 0, 1]
# correct
# result = [0, 1, 0]
# starting
# on
# example
# 7
# rounded
# result = [0, 1, 0]
# correct
# result = [0, 1, 0]
# starting
# on
# example
# 8
# rounded
# result = [0, 1, 0]
# correct
# result = [0, 1, 0]
# starting
# on
# example
# 9
# rounded
# result = [0, 1, 0]
# correct
# result = [0, 1, 0]
# starting
# on
# example
# 10
# rounded
# result = [0, 0, 1]
# correct
# result = [0, 0, 1]
# starting
# on
# example
# 11
# rounded
# result = [0, 0, 1]
# correct
# result = [0, 0, 1]
# starting
# on
# example
# 12
# rounded
# result = [0, 0, 1]
# correct
# result = [0, 0, 1]
# starting
# on
# example
# 13
# rounded
# result = [0, 0, 1]
# correct
# result = [0, 0, 1]
# starting
# on
# example
# 14
# rounded
# result = [0, 0, 1]
# correct
# result = [0, 0, 1]
# classified
# correctly
# 93.33333333333333 %
    #fully connected with 2 hiddenlayers consisting of 6 nodes each en 3 output nodes


# result of  [ 4.4  2.9  1.4  0.2]  =  [0.99459264307128448, 0.0062962067297559583, 0.00044105950632721599]  correct result =  [1, 0, 0]
# result of  [ 5.1  3.3  1.7  0.5]  =  [0.99455110907319844, 0.0063436500591252934, 0.00043966176131318062]  correct result =  [1, 0, 0]
# result of  [ 5.1  3.4  1.5  0.2]  =  [0.99486030247520463, 0.0059916983160238453, 0.00043466018433652875]  correct result =  [1, 0, 0]
# result of  [ 5.4  3.4  1.5  0.4]  =  [0.99485460138117687, 0.0059982108103686441, 0.00043444573695830221]  correct result =  [1, 0, 0]
# result of  [ 4.8  3.   1.4  0.3]  =  [0.99471617670715962, 0.0061557099314638259, 0.00043789795846137204]  correct result =  [1, 0, 0]
# result of  [ 4.9  2.4  3.3  1. ]  =  [0.0051187815641468301, 0.99446492711109602, 0.0036574380132527767]  correct result =  [0, 1, 0]
# result of  [ 6.2  2.2  4.5  1.5]  =  [1.2423462127866593e-05, 0.037776426563064548, 0.96231336954249624]  correct result =  [0, 1, 0]
# result of  [ 6.4  2.9  4.3  1.3]  =  [0.004769185935911799, 0.99483476939125981, 0.0037059093010943459]  correct result =  [0, 1, 0]
# result of  [ 5.8  2.7  3.9  1.2]  =  [0.0048302521352365946, 0.99477012390566277, 0.0036970424010837515]  correct result =  [0, 1, 0]
# result of  [ 6.1  3.   4.6  1.4]  =  [0.0047280538227298481, 0.99487828241234511, 0.0037124153246295062]  correct result =  [0, 1, 0]
# result of  [ 7.6  3.   6.6  2.1]  =  [1.2353496286522679e-05, 0.037516473050684958, 0.96259263754567559]  correct result =  [0, 0, 1]
# result of  [ 6.3  2.7  4.9  1.8]  =  [1.2358351184385596e-05, 0.037523234156061465, 0.96258135335374351]  correct result =  [0, 0, 1]
# result of  [ 7.9  3.8  6.4  2. ]  =  [4.5095953031438754e-05, 0.19921390969769423, 0.78982151739005302]  correct result =  [0, 0, 1]
# result of  [ 6.7  3.1  5.6  2.4]  =  [1.2353546550158371e-05, 0.037516327417165235, 0.96259306302312286]  correct result =  [0, 0, 1]
# result of  [ 6.7  3.3  5.7  2.5]  =  [1.2353570284622396e-05, 0.037516260530399963, 0.96259288811686372]  correct result =  [0, 0, 1]
#
# Process finished with exit code 0

