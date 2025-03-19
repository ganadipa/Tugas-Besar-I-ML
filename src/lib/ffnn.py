from lib.neural import NeuralNetwork, NetworkLayer
from typing import List

class FFNN:
    
    network: NeuralNetwork
    learning_rate: float

    def __init__(self, network: NeuralNetwork, learning_rate: float):
        """TODO: implement CLI to input all Neural Network parameters:
            - weight initialization method
            - node counts
            - activation functions
            - loss function

            Uhm I think this must be done in the client
        """
        self.network = network
        self.learning_rate = learning_rate

    def forward_prop(self, ):
        """TODO: implement forward prop by updating:
            - layers[i].nodes[j]
            - layers[i].activated_nodes[j]
            for 1 < i < layers[i].length and 0 < j < layers[i].length

            i.e. calculate output of all nodes of each layer (except input layer)
        """
    
    def back_prop(self, ):
        """TODO: implement backward prop by updating all:
            - weights[k][i][j]
            - gradients[k][i][j]
        """
    
    def update_weights(self, learning_rate: float):
        """TODO: implement update weights by updating all weights after back prop
        """
    
    def fit(self, 
        x_train: List[List[float]], 
        y_train: List[List[float]],
        batch_size: int,
        learning_rate: float,
        epoch_count: int
    ):
        """TODO: implement batch fit
        """