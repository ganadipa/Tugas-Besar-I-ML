import numpy as np
import activation as act
import loss
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
    weights: List[np.array]
    gradients: List[np.array]

    def __init__(
        self, 
        node_counts: List[int],
        activations: List[act.Activation],
        loss_function: loss.Loss,
        initialize_method: str = "zero"
    ):
        if len(activations) != len(node_counts) - 1:
            raise Exception("Error: tried to declare NeuralNetwork with wrong amount of activation functions.")
        
        if len(node_counts) < 2:
            raise Exception("Error: tried to declare NeuralNetwork with less than 2 layers.")

        self.loss_function = loss_function

        activations = [None] + activations # input layer has no activation

        self.layers = [
            NetworkLayer(cnt, activation)
            for cnt,activation in zip(node_counts, activations)
        ]

        self.initialize_weights(initialize_method)
    
    def initialize_weights(self, initialize_method: str):

        self.weights = []        
        self.gradients = []        

        generate: callable
        if initialize_method == 'zero':
            generate = lambda : 0
        
        elif initialize_method == 'uniform':
            inp_str = "Input space separated lower and upper bounds: "
            lo, hi = map(float, input(inp_str).split())
            generate = lambda : lo + ((hi - lo) * random.randint(0, 1000) / 1000)
        
        elif initialize_method == 'normal':
            raise NotImplementedError()
        
        else:
            raise Exception("Unknown initialization method.")

        for i in range(len(self.layers) - 1):
            self.weights.append(np.array(
                [[generate() for j in range(len(self.layers[i].nodes))]
                for k in range(len(self.layers[i + 1].nodes))]
            ))

if __name__ == "__main__":
    nodecounts = list(map(int, input('Input node counts(space separated values): ').split()))

    obj = NeuralNetwork(nodecounts, [act.Sigmoid] * (len(nodecounts) - 1), loss.BCE, 'uniform')

    print(obj)
