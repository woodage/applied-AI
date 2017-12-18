__author__ = 'robbie'
import math

class Neuron(object):

    def __init__(self, weights = []):
        self.weights = weights
        self.threshold = 0.5

    def norGateActPerceptron(self, activations = []):
        weights = sum([activations[i] * self.weights[i] for i in range(len(activations))])
        return weights < self.threshold

    def actSigmoid(self, activations = []):
        z = sum([activations[i] * self.weights[i] for i in range(len(activations))])
        return 1 / ( 1 + math.exp(-z))

n = Neuron([0.5, 0.5, 0.5])

print(n.norGateActPerceptron([0, 0, 0]))
print(n.norGateActPerceptron([0, 0, 1]))
print(n.norGateActPerceptron([0, 1, 0]))
print(n.norGateActPerceptron([0, 1, 1]))
print(n.norGateActPerceptron([1, 0, 0]))
print(n.norGateActPerceptron([1, 0, 1]))
print(n.norGateActPerceptron([1, 1, 0]))
print(n.norGateActPerceptron([1, 1, 1]))

#sigmoid function
print(n.actSigmoid([0, 0, 0]))
print(n.actSigmoid([0, 0, 1]))
print(n.actSigmoid([0, 1, 0]))
print(n.actSigmoid([0, 1, 1]))
print(n.actSigmoid([1, 0, 0]))
print(n.actSigmoid([1, 0, 1]))
print(n.actSigmoid([1, 1, 0]))
print(n.actSigmoid([1, 1, 1]))