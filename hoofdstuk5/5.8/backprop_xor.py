import numpy as np
from math import e
from math import pow
import random

def sigmoid(x):
    """Standard sigmoid; since it relies on ** to do computation, it broadcasts on vectors and matrices"""
    return 1 / (1 - (e**(-x)))

def derivative_sigmoid(x):
    """Expects input x to be already sigmoid-ed"""
    return x * (1 - x)

def tanh(x):
    """Standard tanh; since it relies on ** and * to do computation, it broadcasts on vectors and matrices"""
    return (1 - e ** (-2*x))/ (1 + e ** (-2*x))

def derived_tanh(x):
    """Expects input x to already be tanh-ed."""
    return 1 - tanh(x)

def relu(x):
    x = np.array(x)
    x[x < 0] = 0
    return x

def derived_relu(x):
    x = np.array(x)
    x[x < 0] = 0
    x[x > 0] = 1
    return x



def forward(inputs,weights,function=sigmoid,step=-1):
    """Function needed to calculate activation on a particular layer.
    step=-1 calculates all layers, thus provides the output of the network
    step=0 returns the inputs
    any step in between, returns the output vector of that particular (hidden) layer"""

    if step == -1:
        for w in range(len(weights)):
            if w == 0:
                inputs = np.append(1, inputs)
                act = function(np.dot(weights[w], inputs))
            else:
                act_prime = np.append(1, act)
                act = function(np.dot(weights[w], act_prime))
        return act

    elif step == 0:
        return inputs

    else:
        for w in range(step):
            if w == 0:
                inputs = np.append(1, inputs)
                act = function(np.dot(weights[w], inputs ))
            else:
                act_prime = np.append(1, act)
                act = function(np.dot(weights[w], act_prime))
        return act

def backprop(inputs, outputs, weights, function=sigmoid, derivative=derivative_sigmoid, eta=0.01):
    """
    Function to calculate deltas matrix based on gradient descent / backpropagation algorithm.
    Deltas matrix represents the changes that are needed to be performed to the weights (per layer) to
    improve the performance of the neural net.
    :param inputs: (numpy) array representing the input vector.
    :param outputs:  (numpy) array representing the output vector.
    :param weights:  list of numpy arrays (matrices) that represent the weights per layer.
    :param function: activation function to be used, e.g. sigmoid or tanh
    :param derivative: derivative of activation function to be used.
    :param learnrate: rate of learning.
    :return: list of numpy arrays representing the delta per weight per layer.
    """
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    deltas = []
    layers = len(weights) # set current layer to output layer
    a_now = forward(inputs, weights, function, layers) # activation on current layer
    for i in range(0, layers):
        a_prev = forward(inputs, weights, function, layers-i-1) # calculate activation of previous layer
        if i == 0:
            error = np.array(derivative(a_now) * (outputs - a_now))  # calculate error on output
        else:
            error = derivative(a_now) * (weights[-i].T).dot(error)[1:] # calculate error on current layer
        delta = eta * np.expand_dims(np.append(1, a_prev), axis=1) * error # calculate adjustments to weights
        deltas.insert(0, delta.T) # store adjustments
        a_now = a_prev # move one layer backwards

    return deltas







def learn(thetaList):


    trainingInputs = np.array([
        [1, 1],  # 0
        [0, 1],  # 1
        [1, 0],  # 1
        [0, 0]  # 0
    ])

    trainingOutputs = np.array([
        [0],
        [1],
        [1],
        [0]
    ])


    for l in range(20000):
        print("learning ....")
        cumulativeError = 0
        for x in range(len(trainingInputs)):
            deltas = backprop(trainingInputs[x], trainingOutputs[x], thetaList, relu, derived_relu)

            res = forward(trainingInputs[x], thetaList, relu )
            err = (res - trainingOutputs[x]) * (res - trainingOutputs[x])
            cumulativeError += err

            for index in range(len(thetaList)):
                thetaList[index] = thetaList[index] + deltas[index]
        print("cumulative error ")
        print(cumulativeError)
        if cumulativeError <= 0.001:
            print("network learned")
            return


theta1 = np.array([[random.uniform(0.2, 0.5), random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)],
                       [random.uniform(0.2, 0.5), random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)]
                       ])

theta2 = np.array([[random.uniform(0.2, 0.5), random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)]])

thetaList = [theta1, theta2]


learn(thetaList)
# it might be necessary to run multiple times
print("result")
print("1 1 ", forward(np.array([[1, 1]]), thetaList, relu, -1))
print("0 1 ",forward(np.array([[0, 1]]), thetaList, relu, -1))
print("1 0 ",forward(np.array([[1, 0]]), thetaList, relu, -1))
print("0 0 ",forward(np.array([[0, 0]]), thetaList, relu, -1))




