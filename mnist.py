
import pickle, gzip, os
from urllib import request
from pylab import imshow, show, cm
import numpy as np
from math import e

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

def get_image ( number ):
    (X, y) = [img[ number] for img in train_set]
    return (np.array(X), y)

def view_image ( number ):
    (X, y) = get_image( number )
    print("Label : %s" % y)
    imshow(X.reshape(28 ,28), cmap=cm.gray)
    show()

def lenOfVector( a):
    #return the magnitude of a vector
    tmp = 0
    for i in a:
        tmp += i*i
    res = tmp * tmp
    return res


url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
if not os.path.isfile("mnist.pkl.gz"):
    print("retrieving url")
    request.urlretrieve (url , "mnist.pkl.gz")

print("opening zip")
f = gzip.open('mnist.pkl.gz', 'rb')
train_set , valid_set , test_set = pickle.load(f, encoding='latin1')
f.close()
print("file closed")


print("starting learning")
theta = [np.random.rand(10, 785)]

epoch = 10000
for x in range(epoch):
    print("epoch ", x)
    cumulativeError = 0
    a = int(len(train_set[0]) / 10)
    for i in range(a):
        #print("iterations passed = ", i)
        img, label = get_image(i)
        expected = np.array([0 for _ in range(10)])
        expected[label] = 1

        #deltas = backprop(img, expected, theta, sigmoid, derivative_sigmoid, eta=0.1)
        deltas = backprop(img, np.expand_dims(expected, axis=1).T, theta, sigmoid, derivative_sigmoid)

        res = forward(img, theta, relu)
        err = (lenOfVector(res) - lenOfVector(expected)) * (lenOfVector(res) - lenOfVector(expected))
        cumulativeError += err
        # updating weights:
        # given an array w of numpy arrays per layer and the deltas calculated by backprop, do
        for index in range(len(theta)):
            theta[index] = theta[index] + deltas[index]

    print("cumulative error ")
    print(cumulativeError)
    if cumulativeError <= 0.01:
        break

print("done")

for i in range(len(valid_set[0])):
    img, label = get_image(i)
    expected = np.array([0 for _ in range(10)])
    expected[label] = 1

    output = forward(img, theta)
    print(expected, "  <--->  ", output)

# onderbouwing
# er zijn 10 mogelijke klassificaties dus een single layer network met 10 output neurons is voldoende.
# afbeelding is 28x28 dus er zijn 784 inputs mogelijk hier komt de bias nog bij dus 785 inputs voor elke neuron
# theta = dus matrix met dimensies R(10)(785)
#
# een mogelijk betere implementatie zou gerealiseerd kunnen worden door een hiddenlayer toe te voegen.
# er zit een verband tussen het aantal neuronen in de hiddenlayer, en het onderscheidbaar vermogen van het netwerk.
# een mogelijk goede verbetering zou kunnen zijn het aantal neurons in de 1st hidden layer gelijk te maken aan 2x
# de output layer. de performance(snelheid) gaat hier wel op achteruit
#
# het netwerk zal er dan als volgt uitzien theta1 = R(20)(785) (bias included) en theta2 = R(10)(21) (bias included)
#