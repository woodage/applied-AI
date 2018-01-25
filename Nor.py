import numpy as np
import math

def sigmoid(x):
     return 1 / (1 + math.e**(-x))

def predict(x, thetaList):
    x = np.array(x)
    input = True

    for theta in thetaList:
        theta = np.array(theta)

        if input == True:
            x_prime = np.append(1, x)
            act = sigmoid(np.dot(theta, x_prime))


            input = False
            if len(thetaList) == 1:
                return act
        else:
            act_prime = np.append(1, act)
            act = sigmoid(np.dot(theta, act_prime))
    return act



theta = np.array([[1, -1, -1, -1]])
thetaList = [theta ] #, theta1]



for q in range(0, 2):
    for r in range(0, 2):
        for s in range(0, 2):
            x = [ q ,r, s]
            print(x,  " --> ", predict(x,thetaList))


