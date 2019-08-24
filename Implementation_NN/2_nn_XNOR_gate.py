#!/bin/python3

import sys
import random
import numpy as np
sys.path.append('..')
from neural_network import NeuralNetwork

inputLen = 2
hiddenLen = 8
outputLen = 1
learningRate = 0.1
n = NeuralNetwork(inputLen, hiddenLen, outputLen)
# n = NeuralNetwork(2, 2, 1)

training_data = {
    1: {'inputs': np.array([[0],[0]]), 'targets': np.array([[1]])},
    2: {'inputs': np.array([[0],[1]]), 'targets': np.array([[0]])},
    3: {'inputs': np.array([[1],[0]]), 'targets': np.array([[0]])},
    4: {'inputs': np.array([[1],[1]]), 'targets': np.array([[1]])},
    }

print("\033[4m" + "\n### Training ###" + "\033[0m")
for i in range(10000):
    x = random.choice(list(training_data.values()))
    inputs =  x.get('inputs')
    targets =  x.get('targets')
    n.trainSVLearing(inputs,targets,learningRate)


print("\033[4m" + "\n### Testing Phase ###" + "\033[0m")
inputs = np.array([
    [0],
    [0]
    ])
print(n.feedForward(inputs))

inputs = np.array([
    [0],
    [1]
    ])
print(n.feedForward(inputs))

inputs = np.array([
    [1],
    [0]
    ])
print(n.feedForward(inputs))

inputs = np.array([
    [1],
    [1]
    ])
print(n.feedForward(inputs))
tkVis = n.nnStructure()
tkVis.mainloop()
