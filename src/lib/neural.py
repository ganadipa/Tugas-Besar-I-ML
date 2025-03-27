import numpy as np
import lib.activation as act
import lib.loss as loss
from lib.weight_initializer import WeightInitializer
import random
from typing import List, Union
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go

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
        self.activation = activation
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
            initialize_methods: Union[WeightInitializer, List[WeightInitializer]]
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

        # Convert single initializer to list if needed
        if not isinstance(initialize_methods, list):
            initialize_methods = [initialize_methods] * (len(node_counts) - 1)
        
        # Ensure we have the right number of initializers
        if len(initialize_methods) != len(node_counts) - 1:
            raise Exception("Error: number of weight initializers must match number of connections between layers.")
            
        self.initialize_weights(initialize_methods)
    

    def initialize_weights(self, initializers: List[WeightInitializer]):
        self.weights = []        
        self.gradients = []
        self.bias_weights = []  
        self.bias_gradients = []  
        
        for i in range(len(self.layers) - 1):
            next_layer_size = len(self.layers[i + 1].nodes)
            current_layer_size = len(self.layers[i].nodes)
            
            initializer = initializers[i]
            weight_matrix = initializer.initialize((next_layer_size, current_layer_size))
            self.weights.append(weight_matrix)
            
            bias_weights = initializer.initialize((next_layer_size,))
            self.bias_weights.append(bias_weights)
            
            self.gradients.append(np.zeros_like(weight_matrix))
            self.bias_gradients.append(np.zeros_like(bias_weights))  

    
    def neural_plot_networks(self):
        """Display the neural network architecture with better readability."""
        
        G = nx.DiGraph()
        pos = {}

        layer_labels = ["Input Layer"] + [f"Hidden Layer {i}" for i in range(1, len(self.layers) - 1)] + ["Output Layer"]
        neuron_labels = {}
        highest_neuron_positions = {}

        edge_traces = []
        node_traces = []

        # Colors per layer
        layer_colors = {
            "Input Layer": "blue",
            "Hidden Layer": "green",
            "Output Layer": "red"
        }

        max_neurons = max(len(layer.nodes) for layer in self.layers)  # Find the largest layer
        
        # Adjust horizontal spacing
        x_spacing = 2

        # Process Nodes
        for layer_idx, layer in enumerate(self.layers):
            n_neurons = len(layer.nodes)
            layer_height = max_neurons / n_neurons  # Normalize spacing
            highest_y = -float("inf")

            for neuron_idx in range(n_neurons):
                node_id = f"L{layer_idx}_{neuron_idx}"
                G.add_node(node_id)

                neuron_name = f"H{layer_idx}_{neuron_idx}" if 0 < layer_idx < len(self.layers) - 1 else f"I{neuron_idx}" if layer_idx == 0 else f"O{neuron_idx}"
                neuron_labels[node_id] = neuron_name

                x_pos = layer_idx * x_spacing
                y_pos = -(neuron_idx * layer_height) + (n_neurons / 2)  # Reversed ordering, 0 at the top

                pos[node_id] = (x_pos, y_pos)

                highest_y = max(highest_y, y_pos)

                node_traces.append(go.Scatter(
                    x=[x_pos], y=[y_pos],
                    mode="markers+text",
                    text=[neuron_name], textposition="bottom center",
                    marker=dict(size=20, color=layer_colors.get(layer_labels[layer_idx], "gray")),
                    hoverinfo="text",
                    hovertext=f"Neuron: {neuron_name}"
                ))

            highest_neuron_positions[layer_idx] = (x_pos, highest_y + 2)

        # Process Edges
        for layer_idx in range(len(self.layers) - 1):
            for prev_idx in range(len(self.layers[layer_idx].nodes)):
                for next_idx in range(len(self.layers[layer_idx + 1].nodes)):
                    prev_node = f"L{layer_idx}_{prev_idx}"
                    next_node = f"L{layer_idx + 1}_{next_idx}"

                    prev_name = neuron_labels[prev_node]
                    next_name = neuron_labels[next_node]

                    weight = self.weights[layer_idx][next_idx, prev_idx]
                    gradient = self.gradients[layer_idx][next_idx, prev_idx]
                    activation_function = self.layers[layer_idx + 1].activation.__class__.__name__

                    edge_name = f"{prev_name} → {next_name}"

                    G.add_edge(prev_node, next_node)

                    x0, y0 = pos[prev_node]
                    x1, y1 = pos[next_node]

                    # Edge Line (Fix: Add midpoints for better hover interaction)
                    edge_traces.append(go.Scatter(
                        x=[x0, (x0 + x1) / 2, x1],  # Add an intermediate point
                        y=[y0, (y0 + y1) / 2, y1],
                        mode="lines+markers",
                        line=dict(width=1.5, color="rgba(100,100,100,0.6)"),
                        marker=dict(size=5, opacity=0),  # Invisible markers at edge midpoints
                        hoverinfo="text",
                        hovertext=f"Edge: {edge_name}<br>Weight: {weight:.3f}<br>Gradient: {gradient:.3f}<br>Activation: {activation_function}"
                    ))

        # Layer Labels
        annotations = []
        for layer_idx, layer_name in enumerate(layer_labels):
            x, y = highest_neuron_positions[layer_idx]
            annotations.append(
                dict(x=x, y=y, text=layer_name, showarrow=False, font=dict(size=16, color="black"))
            )

        fig = go.Figure(data=edge_traces + node_traces)
        
        fig.update_layout(
            title="Neural Network Architecture",
            showlegend=False,
            hovermode="closest",
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(240,240,240,0.8)"
        )

        fig.show()
    

    def neural_plot_weights(self, layer_indices=None):
        """Plot the distribution of weights for specified layers (starting from 1)."""
        
        if layer_indices is None:
            layer_indices = range(1, len(self.weights) + 1)  # Start from 1
        
        n_layers = len(layer_indices)
        fig, axes = plt.subplots(1, n_layers, figsize=(n_layers * 4, 4))
        
        if n_layers == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(layer_indices):
            if layer_idx < 1 or layer_idx > len(self.weights):  # Ensure within range
                print(f"Warning: Layer index {layer_idx} out of range")
                continue

            weights = self.weights[layer_idx - 1].flatten()  # Adjust for 1-based index
            axes[i].hist(weights, bins=30, alpha=0.7)
            axes[i].set_title(f"Layer {layer_idx} Weights")  # Keep 1-based index in title
            axes[i].set_xlabel("Weight Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


    def neural_plot_gradients(self, layer_indices=None):
        """Plot the distribution of gradients for specified layers (starting from 1)."""
        
        if layer_indices is None:
            layer_indices = range(1, len(self.gradients) + 1)  # Start from 1
        
        n_layers = len(layer_indices)
        fig, axes = plt.subplots(1, n_layers, figsize=(n_layers * 4, 4))
        
        if n_layers == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(layer_indices):
            if layer_idx < 1 or layer_idx > len(self.gradients):  # Ensure within range
                print(f"Warning: Layer index {layer_idx} out of range")
                continue

            gradients = self.gradients[layer_idx - 1].flatten()  # Adjust for 1-based index
            axes[i].hist(gradients, bins=30, alpha=0.7)
            axes[i].set_title(f"Layer {layer_idx} Gradients")  # Keep 1-based index in title
            axes[i].set_xlabel("Gradient Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()