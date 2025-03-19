import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lib.activation import Tanh, Linear, ReLU, Sigmoid, Softmax
from lib.loss import CCE

from lib.weight_initializer import ZeroInitializer, UniformInitializer, NormalInitializer
from lib.neural import NeuralNetwork

initializer = NormalInitializer(seed=42)
print(initializer.initialize((3)))
