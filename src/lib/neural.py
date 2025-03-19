import numpy as np
import lib.activation as act
import lib.loss as loss
from lib.weight_initializer import WeightInitializer
import random
from typing import List

class NetworkLayer:

    nodes: np.array
    activation: act.Activation
    activated_nodes: np.array

    def __init__(self, node_count: int, activation: act.Activation):
        self.activation = None if activation is None else activation()
        self.nodes = np.array([0] * node_count)
        self.activated_nodes = np.array([0] * node_count)


class NeuralNetwork:

    loss_function: loss.Loss
    layers: List[NetworkLayer]
    weights: List[np.array] # 3D array
    gradients: List[np.array] # 3D array
    bias_weights: List[np.array] # 2D array

    def __init__(
        self, 
        node_counts: List[int],
        activations: List[act.Activation],
        loss_function: loss.Loss,
        initialize_method: WeightInitializer
    ):
        if len(activations) != len(node_counts) - 1:
            raise Exception("Error: tried to declare NeuralNetwork with wrong amount of activation functions.")
        
        if len(node_counts) < 2:
            raise Exception("Error: tried to declare NeuralNetwork with less than 2 layers.")

        self.loss_function = loss_function

        activations = [None] + activations # input layer has no activation

        self.layers = [
            NetworkLayer(cnt, activation)
            for cnt, activation in zip(node_counts, activations)
        ]

        self.initialize_weights(initialize_method)
    
    def initialize_weights(self, initializer: WeightInitializer):

        self.weights = []        
        self.gradients = []
        self.bias_weights = []  
        
        for i in range(len(self.layers) - 1):
            next_layer_size = len(self.layers[i + 1].nodes)
            current_layer_size = len(self.layers[i].nodes)
            
            weight_matrix = initializer.initialize((next_layer_size, current_layer_size))
            self.weights.append(weight_matrix)
            
            bias_weights = initializer.initialize((next_layer_size,))
            self.bias_weights.append(bias_weights)
            
            self.gradients.append(np.zeros_like(weight_matrix))

