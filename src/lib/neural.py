import numpy as np
import lib.activation as act
import lib.loss as loss
from lib.weight_initializer import WeightInitializer
import random
from typing import List
import pickle
import matplotlib.pyplot as plt
import networkx as nx

class NetworkLayer:
    """A layer in a neural network"""

    nodes: np.ndarray
    activation: act.Activation
    activated_nodes: np.ndarray

    def __init__(self, node_count: int, activation: act.Activation):
        """
        Initialize a network layer
        
        Args:
            node_count: Number of nodes in this layer
            activation: Activation function for this layer
        """
        self.activation = None if activation is None else activation()
        self.nodes = np.zeros(node_count)
        self.activated_nodes = np.zeros(node_count)



class NeuralNetwork:

    loss_function: loss.Loss
    layers: List[NetworkLayer]
    weights: List[np.ndarray] # 3D array
    gradients: List[np.ndarray] # 3D array
    bias_weights: List[np.ndarray] # 2D array
    bias_gradients: List[np.ndarray] # 2D array

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
        self.bias_gradients = []  
        
        for i in range(len(self.layers) - 1):
            next_layer_size = len(self.layers[i + 1].nodes)
            current_layer_size = len(self.layers[i].nodes)
            
            weight_matrix = initializer.initialize((next_layer_size, current_layer_size))
            self.weights.append(weight_matrix)
            
            bias_weights = initializer.initialize((next_layer_size,))
            self.bias_weights.append(bias_weights)
            
            self.gradients.append(np.zeros_like(weight_matrix))
            self.bias_gradients.append(np.zeros_like(bias_weights))  

    def show(self):
        """Display the neural network architecture as a graph.
        
        This method visualizes the network structure, including all layers,
        connections between neurons, weights, and gradients.
        """
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Define positions for all neurons
        pos = {}
        neuron_labels = {}
        
        # Add nodes for each layer
        for layer_idx, layer in enumerate(self.layers):
            n_neurons = len(layer.nodes)
            for neuron_idx in range(n_neurons):
                node_id = f"L{layer_idx}_{neuron_idx}"
                G.add_node(node_id)
                pos[node_id] = (layer_idx, neuron_idx - n_neurons/2)
                
                if layer_idx == 0:
                    neuron_labels[node_id] = f"Input {neuron_idx}"
                elif layer_idx == len(self.layers) - 1:
                    neuron_labels[node_id] = f"Output {neuron_idx}"
                else:
                    neuron_labels[node_id] = f"H{layer_idx}_{neuron_idx}"
        
        # Add edges between neurons
        edge_labels = {}
        for layer_idx in range(len(self.layers) - 1):
            for prev_idx in range(len(self.layers[layer_idx].nodes)):
                for next_idx in range(len(self.layers[layer_idx + 1].nodes)):
                    prev_node = f"L{layer_idx}_{prev_idx}"
                    next_node = f"L{layer_idx + 1}_{next_idx}"
                    
                    weight = self.weights[layer_idx][next_idx, prev_idx]
                    gradient = self.gradients[layer_idx][next_idx, prev_idx]
                    
                    G.add_edge(prev_node, next_node)
                    edge_labels[(prev_node, next_node)] = f"W: {weight:.2f}\nG: {gradient:.2f}"
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=False, node_size=700, node_color='skyblue', 
                font_weight='bold', arrowsize=20, edge_color='gray')
        
        nx.draw_networkx_labels(G, pos, labels=neuron_labels, font_size=10)
        
        # Draw edge labels (optional, can be messy for large networks)
        if len(self.layers) < 4 and max([len(l.nodes) for l in self.layers]) < 8:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title('Neural Network Architecture')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_weights(self, layer_indices=None):
        """Plot the distribution of weights for specified layers.
        
        Args:
            layer_indices: List of indices indicating which layers to plot.
                        If None, all layers are plotted.
                        
        Example:
            # Plot weights for the first and second layers
            model.plot_weights([0, 1])
            
            # Plot weights for all layers
            model.plot_weights()
        """

        
        if layer_indices is None:
            layer_indices = range(len(self.weights))
        
        n_layers = len(layer_indices)
        fig, axes = plt.subplots(1, n_layers, figsize=(n_layers * 4, 4))
        
        # Handle case with only one layer
        if n_layers == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(layer_indices):
            if layer_idx >= len(self.weights):
                print(f"Warning: Layer index {layer_idx} out of range")
                continue
                
            weights = self.weights[layer_idx].flatten()
            axes[i].hist(weights, bins=30, alpha=0.7)
            axes[i].set_title(f"Layer {layer_idx+1} Weights")
            axes[i].set_xlabel("Weight Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_gradients(self, layer_indices=None):
        """Plot the distribution of gradients for specified layers.
        
        Args:
            layer_indices: List of indices indicating which layers to plot.
                        If None, all layers are plotted.
                        
        Example:
            # Plot gradients for the first and second layers
            model.plot_gradients([0, 1])
            
            # Plot gradients for all layers
            model.plot_gradients()
        """

        
        if layer_indices is None:
            layer_indices = range(len(self.gradients))
        
        n_layers = len(layer_indices)
        fig, axes = plt.subplots(1, n_layers, figsize=(n_layers * 4, 4))
        
        # Handle case with only one layer
        if n_layers == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(layer_indices):
            if layer_idx >= len(self.gradients):
                print(f"Warning: Layer index {layer_idx} out of range")
                continue
                
            gradients = self.gradients[layer_idx].flatten()
            axes[i].hist(gradients, bins=30, alpha=0.7)
            axes[i].set_title(f"Layer {layer_idx+1} Gradients")
            axes[i].set_xlabel("Gradient Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath: str) -> None:
        """Save the neural network model to a file.
        
        Args:
            filepath: Path where the model will be saved
            
        Example:
            # Save the current model
            model.save("my_neural_network.pkl")
        """
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'NeuralNetwork':
        """Load a neural network model from a file.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            Loaded neural network model
            
        Example:
            # Load a saved model
            loaded_model = NeuralNetwork.load("my_neural_network.pkl")
        """
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
