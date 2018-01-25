class Neuron:

    def __init__(self, weights, treshold):
        self.treshold = treshold
        self.weights = weights


    def weightMultiplier(self, input):
        ele = 0
        result = []
        for each in self.weights:
            result.append(input[ele] * each)
            ele += 1

        return result

    def update(self):
        pass


    def act(self, input):
        if sum(self.weightMultiplier(input)) - self.treshold >= 0:
            return 1
        else:
            return 0




#adder

# and gates
AndGate1 = Neuron([1, 1], 2)
AndGate2 = Neuron([1, 1], 2)
# or gatess
OrGate1 = Neuron([1, 1], 1)
OrGate2 = Neuron([1, 1], 1)
# inverters

inverter1 = Neuron([-2], -1)
inverter2 = Neuron([-2], -1)

# exor, and
a = 0
b = 0
print("input -> ", a, b,  " = Sum: ",  AndGate1.act([OrGate1.act([a, b]) , OrGate2.act([inverter1.act([a])  , inverter2.act([b])])]), " Carry: " , AndGate2.act([a, b]) )
a = 0
b = 1
print("input -> ", a, b, " = Sum: ",  AndGate1.act([OrGate1.act([a, b]) , OrGate2.act([inverter1.act([a])  , inverter2.act([b])])])," Carry: ", AndGate2.act([a, b]) )
a = 1
b = 0
print("input -> ", a, b, " = Sum: ",  AndGate1.act([OrGate1.act([a, b]) , OrGate2.act([inverter1.act([a])  , inverter2.act([b])])]), " Carry: ", AndGate2.act([a, b]) )
a = 1
b = 1
print("input -> ", a, b, " = Sum: ",  AndGate1.act([OrGate1.act([a, b]) , OrGate2.act([inverter1.act([a])  , inverter2.act([b])])]), " Carry: ", AndGate2.act([a, b]) )


# norgate out of perceptron
norGate = Neuron([-1, -1, -1], 0)
print(norGate.act([0,0,0]))
print(norGate.act([0,1,0]))

