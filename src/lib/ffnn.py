from lib.neural import NeuralNetwork
from typing import List, Dict, Optional, Tuple
import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize

class FFNN:
    
    network: NeuralNetwork
    regularization_method: str
    loss_history: Dict[str, List[float]]

    def __init__(self, network: NeuralNetwork, regularization_method:str=None):
        """Initialize a Feed-Forward Neural Network.
        
        Args:
            network: The neural network architecture
            loss_history: Dictionary to store training and validation loss
        """
        self.network = network
        self.loss_history = {
            'train_loss': [],
            'val_loss': []
        }

        self.regularization_method = regularization_method
    

    def forward_prop(self, x_batch: np.ndarray) -> np.ndarray:
        """Perform forward propagation through the network.
        
        Args:
            x_batch: Input data batch of shape (batch_size, input_features)
            
        Returns:
            Output predictions of shape (batch_size, output_features)
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
        
        return self.network.layers[-1].activated_nodes.T
    

    def back_prop(self, x_batch: np.ndarray, y_batch: np.ndarray, learning_rate: float) -> float:
        """Perform backward propagation to compute gradients.
        
        Args:
            x_batch: Input data batch of shape (batch_size, input_features)
            y_batch: Target data batch of shape (batch_size, output_features)
            
        Returns:
            The computed loss value
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
        output_layer = self.network.layers[-1]
        output_activations = output_layer.activated_nodes
            
        
        # Transpose y_batch to match the shape of output_activations
        y_batch_T = y_batch.T
        
        # Calculate loss
        loss = self.network.loss_function.function(y_batch_T, output_activations)
        
        reg_const = 1
        if self.regularization_method == "L1":
            reg_const = learning_rate * sum(np.sum(np.abs(w)) for w in self.network.weights)
        if self.regularization_method == "L2":
            reg_const = learning_rate * sum(np.sum(np.square(w)) for w in self.network.weights)
        
        d_loss = reg_const * self.network.loss_function.derivative(y_batch_T, output_activations)
        d_activation = output_layer.activation.derivative(output_layer.nodes)


        if (d_activation.ndim == 3): # softmax yea
            output_sz, batch_sz = d_loss.shape
            delta = np.zeros((output_sz, batch_sz))
            
            for i in range(batch_sz):
                # d_activation[:,:,i] is the Jacobian matrix (output_sz, output_sz)
                # d_loss[:,i] is the loss gradient vector (output_sz,)
                delta[:,i] = np.dot(d_activation[:,:,i], d_loss[:,i])

        else: 
            delta = d_loss * d_activation

        
        # Backpropagate the error through the network
        for l in reversed(range(1, len(self.network.layers))):
            layer = self.network.layers[l]
            prev_layer = self.network.layers[l-1]
            
            # Compute weight gradients for this layer
            self.network.gradients[l-1] = np.dot(delta, prev_layer.activated_nodes.T) / batch_size

            # Compute bias gradients (average across batch)
            self.network.bias_gradients[l-1] = np.mean(delta, axis=1)
            
            # Backpropagate delta to previous layer (if not input layer)
            if l > 1:
                # Compute delta for previous layer
                delta = np.dot(self.network.weights[l-1].T, delta)
                d_actprev = prev_layer.activation.derivative(prev_layer.nodes)
                
                if d_actprev.ndim == 2 :
                    delta *= d_actprev
                
                else: # softmax yea

                    prev_sz = prev_layer.nodes.shape[0]
                    newdelta = np.zeros((prev_sz, batch_sz))
                    
                    for i in range(batch_sz):
                        # For each sample in the batch
                        # d_activation[:,:,i] is the Jacobian matrix (output_sz, output_sz)
                        # d_loss[:,i] is the loss gradient vector (output_sz,)
                        newdelta[:,i] = np.dot(d_actprev[:,:,i], delta[:,i])

                    delta = newdelta
        
        return loss
    

    def update_weights(self, learning_rate) -> None:
        """Update weights using gradient descent.
        """
        for i in range(len(self.network.weights)):
            self.network.weights[i] -= learning_rate * self.network.gradients[i]
            self.network.bias_weights[i] -= learning_rate * self.network.bias_gradients[i]
    

    def fit(self, 
        x_train: np.ndarray, 
        y_train: np.ndarray,
        batch_size: int = 32,
        epochs: int = 10,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        learning_rate: float = 0.01,
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
                batch_loss = self.back_prop(batch_x, batch_y, learning_rate)
                
                # Update weights
                self.update_weights(learning_rate)
                
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
        """
        return self.forward_prop(x)


    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the model on test data.
        
        Args:
            x: Test data of shape (n_samples, n_features)
            y: True values of shape (n_samples, n_outputs)
            
        Returns:
            Loss value on test data
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
    

    def save(self, file_path: str):
        """Save the entire FFNN instance to a file.
        
        Args:
            file_path: Path to save
        
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    

    def load(file_path: str) -> "FFNN":
        """Load a saved FFNN instance from a file.
        
        Args:
            file_path: Path to load from
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    

    def plot_networks(self) -> None:
        """Plot the neural network architecture.
        """
        self.network.neural_plot_networks()
    

    def plot_weights(self, layer_indices=None, title="Weight Distribution") -> None:
        """Plot the weights of the neural network.

        Args:
            layer_indices: List of layer indices to plot weights
        """
        self.network.neural_plot_weights(layer_indices=layer_indices, title=title)    

    def plot_gradients(self, layer_indices=None, title="Gradient Distribution") -> None:
        """Plot the gradients of the neural network.

        Args:
            layer_indices: List of layer indices to plot gradients
        """
        self.network.neural_plot_gradients(layer_indices=layer_indices, title=title)  
    

    def plot_loss_history(self) -> None:
        """Plot the training and validation loss history.
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


# Standalone Utility Functions
def evaluate_model(model, X, y_onehot) -> None:
    """Evaluate model performance and print summary metrics.
    
    Args:
        model: Trained model
        X: Input data of shape (n_samples, n_features)
        y_onehot: One-hot encoded target data of shape (n_samples, n_classes)
    """
    y_pred = model.predict(X)
    predicted_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_onehot, axis=1)
    
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes, average='macro')
    recall = recall_score(true_classes, predicted_classes, average='macro')
    f1 = f1_score(true_classes, predicted_classes, average='macro')

    print(f"\n----- Model Performance Summary -----")
    print(f"{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'Accuracy':<20} {accuracy:.10f}")
    print(f"{'Precision (macro)':<20} {precision:.10f}")
    print(f"{'Recall (macro)':<20} {recall:.10f}")
    print(f"{'F1 Score (macro)':<20} {f1:.10f}")
    print("-" * 30)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
        
    }


def plot_training_loss(history: Dict[str, List[float]], title: str = "Training History") -> None:
    """Plot training and validation loss history.
    
    Args:
        history: Dictionary containing 'train_loss' and 'val_loss'
        title: Title of the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_confusion_matrix(model, X, y_onehot):
    """Plot confusion matrix for a single model."""
    y_true = np.argmax(y_onehot, axis=1)
    y_pred = np.argmax(model.predict(X), axis=1)
    cm = confusion_matrix(y_true, y_pred)

    title = f"{model.__class__.__name__} Confusion Matrix"

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title, fontsize=15)
    plt.colorbar()
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.tight_layout()
    plt.show()