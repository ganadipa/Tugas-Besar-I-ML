import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import lib.activation
import lib.ffnn
import lib.neural
import lib.loss
import lib.weight_initializer

sigma = lib.activation.Sigmoid()
nn = lib.neural.NeuralNetwork(
    [5, 3, 2, 1],
    [sigma] * 3,
    lib.loss.MSE(),
    lib.weight_initializer.NormalInitializer(1, 0.1, 13522022)
)
nn.plot_gradients(layer_indices=[0,2])