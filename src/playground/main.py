import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lib.activation import Tanh, Linear, ReLU, Sigmoid, Softmax
import numpy as np

activation = Softmax()
print(activation.function(np.array([1, -2, 3])))
print(activation.derivative(np.array([1, -2, 3])))