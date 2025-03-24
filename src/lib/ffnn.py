from lib.neural import NeuralNetwork, NetworkLayer
from typing import List, Dict, Optional, Tuple
import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
from lib.loss import Loss, MSE, BCE, CCE
from lib.activation import Sigmoid, Softmax

class FFNN:
    
    network: NeuralNetwork
    learning_rate: float
    loss_history: Dict[str, List[float]]

    def __init__(self, network: NeuralNetwork, learning_rate: float = 0.01):
        """Initialize a Feed-Forward Neural Network.
        
        Args:
            network: The neural network architecture
            learning_rate: Learning rate for gradient descent
        """
        self.network = network
        self.learning_rate = learning_rate
        self.loss_history = {
            'train_loss': [],
            'val_loss': []
        }

    def forward_prop(self, x_batch: np.ndarray) -> np.ndarray:
        """Perform forward propagation through the network.
        
        Args:
            x_batch: Input data batch of shape (batch_size, input_features)
            
        Returns:
            Output predictions of shape (batch_size, output_features)
            
        Example:
            # For a network with 2 input features and 1 output:
            x_sample = np.array([[0.5, 0.8], [0.1, 0.2]])  # 2 samples, 2 features each
            predictions = model.forward_prop(x_sample)
            # predictions shape: (2, 1) e.g., [[0.75], [0.32]]
        """
        # Ensure x_batch is 2D
        if x_batch.ndim == 1:
            x_batch = x_batch.reshape(1, -1)
        
            
        batch_size = x_batch.shape[0]
        
        # Set input layer values
        self.network.layers[0].nodes = x_batch.T  # Transpose to (features, batch_size)
        self.network.layers[0].activated_nodes = self.network.layers[0].nodes  # No activation for input layer
        
        # Forward pass through each layer   
        for i in range(1, len(self.network.layers)):
            current_layer = self.network.layers[i]
            prev_layer = self.network.layers[i-1]
            
            # Compute the weighted sum: weights * prev_activated_nodes + bias
            # Shape becomes (current_layer_nodes, batch_size)
            weighted_sum = np.dot(self.network.weights[i-1], prev_layer.activated_nodes)
            
            # Add bias (broadcasting across batch)
            bias_expanded = self.network.bias_weights[i-1].reshape(-1, 1)
            weighted_sum = weighted_sum + bias_expanded
            
            # Store pre-activation values
            current_layer.nodes = weighted_sum
            
            # Apply activation function
            if current_layer.activation is not None:
                current_layer.activated_nodes = current_layer.activation.function(weighted_sum)
            else:
                current_layer.activated_nodes = weighted_sum
        
        # Return output layer activations transposed back to (batch_size, output_features)
        return self.network.layers[-1].activated_nodes.T
    
    def back_prop(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Perform backward propagation to compute gradients.
        
        Args:
            x_batch: Input data batch of shape (batch_size, input_features)
            y_batch: Target data batch of shape (batch_size, output_features)
            
        Returns:
            The computed loss value
            
        Example:
            # For a batch of 10 samples with 5 input features and 3 output classes
            x_batch = np.random.rand(10, 5)  # 10 samples, 5 features each
            y_batch = np.zeros((10, 3))  # One-hot encoded targets for 3 classes
            # Set the correct class for each sample
            for i in range(10):
                y_batch[i, np.random.randint(0, 3)] = 1
                
            # Perform backward propagation
            loss = model.back_prop(x_batch, y_batch)
            # loss is a float representing the loss for this batch
        """
        # Ensure inputs are properly shaped
        if x_batch.ndim == 1:
            x_batch = x_batch.reshape(1, -1)
        if y_batch.ndim == 1:
            y_batch = y_batch.reshape(1, -1)
            
        batch_size = x_batch.shape[0]
        
        # Forward pass (to compute activations)
        self.forward_prop(x_batch)
        
        # Initialize gradients for this batch
        for i in range(len(self.network.weights)):
            self.network.gradients[i] = np.zeros_like(self.network.weights[i])
            self.network.bias_gradients[i] = np.zeros_like(self.network.bias_weights[i])
        
        # Compute output layer error (delta)
        # Shape: (output_features, batch_size)
        output_layer = self.network.layers[-1]
        output_activations = output_layer.activated_nodes  # Shape: (output_features, batch_size)
        
        # Transpose y_batch to match the shape of output_activations
        y_batch_T = y_batch.T  # Shape: (output_features, batch_size)
        
        # Calculate loss
        loss = self.network.loss_function.function(y_batch_T, output_activations)
        
        # Calculate output layer error based on the loss function and activation
        d_loss = self.network.loss_function.derivative(y_batch_T, output_activations)
        d_activation = output_layer.activation.derivative(output_layer.nodes)
        delta = d_loss * d_activation
        
        # Backpropagate the error through the network
        for l in reversed(range(1, len(self.network.layers))):
            layer = self.network.layers[l]
            prev_layer = self.network.layers[l-1]
            
            # Compute weight gradients for this layer
            # delta shape: (current_layer_size, batch_size)
            # prev_activations shape: (prev_layer_size, batch_size)
            # gradient shape: (current_layer_size, prev_layer_size)
            self.network.gradients[l-1] = np.dot(delta, prev_layer.activated_nodes.T) / batch_size
            
            # Compute bias gradients (average across batch)
            self.network.bias_gradients[l-1] = np.mean(delta, axis=1)
            
            # Backpropagate delta to previous layer (if not input layer)
            if l > 1:
                # Compute delta for previous layer
                # delta shape: (current_layer_size, batch_size)
                # weights shape: (current_layer_size, prev_layer_size)
                # new delta shape: (prev_layer_size, batch_size)
                delta = np.dot(self.network.weights[l-1].T, delta)
                delta *= prev_layer.activation.derivative(prev_layer.nodes)
        
        return loss
    
    def update_weights(self) -> None:
        """Update weights using gradient descent.
        
        Example:
            # After computing gradients with back_prop
            model.update_weights()
            
            # This will update all weights and biases in the network:
            # For each weight matrix and bias vector:
            # w_new = w_old - learning_rate * gradient
        """
        for i in range(len(self.network.weights)):
            self.network.weights[i] -= self.learning_rate * self.network.gradients[i]
            self.network.bias_weights[i] -= self.learning_rate * self.network.bias_gradients[i]
    
    def fit(self, 
        x_train: np.ndarray, 
        y_train: np.ndarray,
        batch_size: int = 32,
        epochs: int = 10,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """Train the neural network using batched gradient descent.
        
        Args:
            x_train: Training data of shape (n_samples, n_features)
            y_train: Training targets of shape (n_samples, n_outputs)
            batch_size: Size of mini-batches
            epochs: Number of training epochs
            validation_data: Optional tuple of (x_val, y_val) for validation
            verbose: 0 for no output, 1 for progress bar and metrics
            
        Returns:
            History dictionary containing training and validation loss
            
        Example:
            # For a dataset with 1000 samples, 20 features, and 3 classes
            x_train = np.random.rand(1000, 20)  # 1000 samples, 20 features each
            y_train = np.zeros((1000, 3))       # One-hot encoded targets for 3 classes
            # Set the correct class for each sample
            for i in range(1000):
                y_train[i, np.random.randint(0, 3)] = 1
                
            # Optional validation data
            x_val = np.random.rand(200, 20)
            y_val = np.zeros((200, 3))
            for i in range(200):
                y_val[i, np.random.randint(0, 3)] = 1
                
            # Train the model
            history = model.fit(
                x_train, 
                y_train,
                batch_size=32,
                epochs=50,
                validation_data=(x_val, y_val),
                verbose=1
            )
            
            # history is a dictionary with keys:
            # - 'train_loss': list of training loss values for each epoch
            # - 'val_loss': list of validation loss values for each epoch
        """
        # Ensure inputs are properly shaped
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
            
        num_samples = x_train.shape[0]
        
        # Reset loss history
        self.loss_history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            
            # Shuffle training data
            indices = np.random.permutation(num_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            # Initialize epoch loss
            epoch_loss = 0.0
            
            # Create mini-batches
            num_batches = int(np.ceil(num_samples / batch_size))
            
            # Progress tracking
            if verbose == 1:
                batch_iterator = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")
            else:
                batch_iterator = range(num_batches)
            
            # Process each batch
            for batch_idx in batch_iterator:
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                
                batch_x = x_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]
                
                # Forward and backward passes
                batch_loss = self.back_prop(batch_x, batch_y)
                
                # Update weights
                self.update_weights()
                
                # Update epoch loss (weighted by batch size)
                batch_weight = (end_idx - start_idx) / num_samples
                epoch_loss += batch_loss * batch_weight
                
                # Update progress bar
                if verbose == 1:
                    batch_iterator.set_postfix({"loss": f"{epoch_loss:.4f}"})
            
            # Record training loss
            self.loss_history['train_loss'].append(epoch_loss)
            
            # Validation phase
            val_loss = None
            if validation_data is not None:
                x_val, y_val = validation_data
                
                # Ensure proper shapes
                if x_val.ndim == 1:
                    x_val = x_val.reshape(-1, 1)
                if y_val.ndim == 1:
                    y_val = y_val.reshape(-1, 1)
                
                # Forward pass on validation data
                val_predictions = self.predict(x_val)
                
                # Calculate validation loss
                val_loss = self.network.loss_function.function(y_val.T, self.network.layers[-1].activated_nodes)
                self.loss_history['val_loss'].append(val_loss)
            
            # Print epoch summary
            if verbose == 1:
                epoch_time = time.time() - start_time
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {epoch_loss:.4f}")
        
        return self.loss_history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate predictions for input samples.
        
        Args:
            x: Input data of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples, n_outputs)
            
        Example:
            # For a dataset with 10 samples, 20 features, and a model with 3 output classes
            x_test = np.random.rand(10, 20)
            
            # Make predictions
            predictions = model.predict(x_test)
            # predictions shape: (10, 3), e.g.:
            # [[0.1, 0.7, 0.2],  # Probabilities for sample 1
            #  [0.8, 0.1, 0.1],  # Probabilities for sample 2
            #  ...]
            
            # For classification, get the class with highest probability
            predicted_classes = np.argmax(predictions, axis=1)
            # predicted_classes: [1, 0, ...]
        """
        return self.forward_prop(x)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the model on test data.
        
        Args:
            x: Test data of shape (n_samples, n_features)
            y: True values of shape (n_samples, n_outputs)
            
        Returns:
            Loss value on test data
            
        Example:
            # For a test dataset with 100 samples, 20 features, and 3 classes
            x_test = np.random.rand(100, 20)
            y_test = np.zeros((100, 3))       # One-hot encoded targets
            for i in range(100):
                y_test[i, np.random.randint(0, 3)] = 1
                
            # Evaluate the model
            test_loss = model.evaluate(x_test, y_test)
            # test_loss is a float, e.g., 0.18
        """
        # Ensure inputs are properly shaped
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        # Forward pass
        predictions = self.predict(x)
        
        # Calculate loss
        return self.network.loss_function.function(y.T, self.network.layers[-1].activated_nodes)
    
    def save(self, filepath: str) -> None:
        """Save the model to a file.
        
        Args:
            filepath: Path to save the model
            
        Example:
            # Save the trained model
            model.save("my_mnist_model.pkl")
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'network': self.network,
                'learning_rate': self.learning_rate,
                'loss_history': self.loss_history
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'FFNN':
        """Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded FFNN model
            
        Example:
            # Load a previously saved model
            loaded_model = FFNN.load("my_mnist_model.pkl")
            
            # Use the loaded model for predictions
            test_predictions = loaded_model.predict(x_test)
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(data['network'], data['learning_rate'])
        model.loss_history = data['loss_history']
        return model
    
    def plot_loss_history(self) -> None:
        """Plot the training and validation loss history.
        
        Example:
            # After training the model
            model.plot_loss_history()
            
            # This will display a graph showing:
            # - Training loss curve (blue line)
            # - Validation loss curve (if validation data was provided, orange line)
            # The x-axis represents epochs, and the y-axis represents loss values
        """
        if not self.loss_history['train_loss']:
            print("No training history to plot.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history['train_loss'], label='Training Loss', marker='o')
        
        if self.loss_history['val_loss']:
            plt.plot(self.loss_history['val_loss'], label='Validation Loss', marker='x')
        
        plt.title('Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()