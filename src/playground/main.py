import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

# Import custom FFNN implementation
from lib.ffnn import FFNN
from lib.neural import NeuralNetwork
from lib.activation import ReLU, Sigmoid, Tanh, Softmax
from lib.loss import CCE
from lib.weight_initializer import NormalInitializer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FFNN-Comparison")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Parameters
train_samples = 5000
test_samples = 1000
hidden_layer_sizes = [128, 64]  # Two hidden layers
learning_rate = 0.01
batch_size = 32
epochs = 10

def load_mnist():
    """Load and preprocess MNIST dataset"""
    logger.info("Loading MNIST dataset...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    
    # Shuffle data
    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))
    
    # Convert labels to integers
    y = y.astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_samples, test_size=test_samples, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # One-hot encode labels for custom FFNN
    y_train_onehot = np.zeros((y_train.size, 10))
    y_train_onehot[np.arange(y_train.size), y_train] = 1
    
    y_test_onehot = np.zeros((y_test.size, 10))
    y_test_onehot[np.arange(y_test.size), y_test] = 1
    
    logger.info(f"Data loaded: X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot

class PyTorchModel(nn.Module):
    """PyTorch equivalent of our custom FFNN implementation"""
    def __init__(self, input_size, hidden_sizes, output_size):
        super(PyTorchModel, self).__init__()
        
        # Create layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        
        # Hidden layers with ReLU
        for i in range(len(layer_sizes) - 2):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        # Output layer
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        
        # Initialize weights with normal distribution
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights to match custom FFNN initialization"""
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # Process through hidden layers with ReLU
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))
        
        # Output layer (logits)
        x = self.layers[-1](x)
        return x

def train_pytorch_model(model, X_train, y_train, batch_size, epochs, learning_rate):
    """Train PyTorch model and track performance"""
    logger.info("Training PyTorch model...")
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Class indices
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {'train_loss': []}
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            
            # Loss calculation
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item() * X_batch.size(0)
        
        # Average loss for this epoch
        epoch_loss /= len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        
        logger.info(f"PyTorch - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    training_time = time.time() - start_time
    logger.info(f"PyTorch training completed in {training_time:.2f} seconds")
    
    return model, history, training_time

def train_scratch_model(X_train, y_train_onehot, batch_size, epochs, learning_rate):
    """Train custom FFNN model and track performance"""
    logger.info("Training scratch FFNN model...")
    
    # Network architecture
    input_size = X_train.shape[1]  # 784 for MNIST
    output_size = 10  # 10 digits
    layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
    
    # Create activations (ReLU for hidden, Softmax for output)
    activations = [ReLU()] * len(hidden_layer_sizes)
    activations.append(Softmax())
    
    # Create network with normal weight initialization
    initializer = NormalInitializer(mean=0.0, var=0.1, seed=42)
    
    nn_model = NeuralNetwork(
        node_counts=layer_sizes,
        activations=activations,
        loss_function=CCE(),
        initialize_methods=initializer
    )
    
    model = FFNN(nn_model)
    
    # Train the model
    start_time = time.time()
    
    history = model.fit(
        x_train=X_train,
        y_train=y_train_onehot,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        verbose=1
    )
    
    training_time = time.time() - start_time
    logger.info(f"Scratch FFNN training completed in {training_time:.2f} seconds")
    
    return model, history, training_time

def evaluate_models(scratch_model, torch_model, X_test, y_test, y_test_onehot):
    """Evaluate both models and compare performance"""
    logger.info("Evaluating models...")
    
    # Evaluate scratch model
    scratch_start = time.time()
    scratch_preds = scratch_model.predict(X_test)
    scratch_pred_time = time.time() - scratch_start
    
    scratch_pred_classes = np.argmax(scratch_preds, axis=1)
    scratch_accuracy = np.mean(scratch_pred_classes == y_test)
    
    # Evaluate PyTorch model
    torch_model.eval()
    torch_start = time.time()
    
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        torch_outputs = torch_model(X_test_tensor)
        _, torch_pred_classes = torch.max(torch_outputs, 1)
    
    torch_pred_time = time.time() - torch_start
    torch_accuracy = (torch_pred_classes.numpy() == y_test).mean()
    
    return {
        'scratch': {
            'accuracy': scratch_accuracy,
            'prediction_time': scratch_pred_time
        },
        'pytorch': {
            'accuracy': torch_accuracy,
            'prediction_time': torch_pred_time
        }
    }

def plot_results(scratch_history, torch_history, scratch_time, torch_time, 
                 scratch_accuracy, torch_accuracy):
    """Create visualization of comparison results"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(scratch_history['train_loss'], 'b-', label='Scratch FFNN')
    plt.plot(torch_history['train_loss'], 'r-', label='PyTorch')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training Time
    plt.subplot(1, 3, 2)
    bars = plt.bar([0, 1], [scratch_time, torch_time], 
             color=['blue', 'red'])
    plt.xticks([0, 1], ['Scratch FFNN', 'PyTorch'])
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    
    # Add times above bars
    for i, bar in enumerate(bars):
        time_val = scratch_time if i == 0 else torch_time
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{time_val:.2f}s', ha='center')
    
    # Plot 3: Accuracy
    plt.subplot(1, 3, 3)
    bars = plt.bar([0, 1], [scratch_accuracy, torch_accuracy], 
             color=['blue', 'red'])
    plt.xticks([0, 1], ['Scratch FFNN', 'PyTorch'])
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy Comparison')
    plt.ylim(0, 1)
    
    # Add accuracies above bars
    for i, bar in enumerate(bars):
        acc_val = scratch_accuracy if i == 0 else torch_accuracy
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc_val:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('ffnn_pytorch_comparison.png', dpi=300)
    plt.show()

def run_comparison():
    """Run full comparison between custom FFNN and PyTorch"""
    # Load data
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = load_mnist()
    
    # Train scratch model
    scratch_model, scratch_history, scratch_time = train_scratch_model(
        X_train, y_train_onehot, batch_size, epochs, learning_rate
    )
    
    # Create and train PyTorch model
    torch_model = PyTorchModel(
        input_size=X_train.shape[1],
        hidden_sizes=hidden_layer_sizes,
        output_size=10
    )
    
    torch_model, torch_history, torch_time = train_pytorch_model(
        torch_model, X_train, y_train, batch_size, epochs, learning_rate
    )
    
    # Evaluate models
    eval_results = evaluate_models(
        scratch_model, torch_model, X_test, y_test, y_test_onehot
    )
    
    # Print results
    print("\n----- FFNN Implementation Comparison -----")
    print(f"Network: 784-{'-'.join(map(str, hidden_layer_sizes))}-10")
    print(f"Learning Rate: {learning_rate}, Batch Size: {batch_size}, Epochs: {epochs}")
    
    print("\nTraining:")
    print(f"Scratch FFNN - Time: {scratch_time:.2f}s, Final Loss: {scratch_history['train_loss'][-1]:.4f}")
    print(f"PyTorch - Time: {torch_time:.2f}s, Final Loss: {torch_history['train_loss'][-1]:.4f}")
    print(f"Speed Improvement: {scratch_time/torch_time:.2f}x faster with PyTorch")
    
    print("\nEvaluation:")
    print(f"Scratch FFNN - Accuracy: {eval_results['scratch']['accuracy']:.4f}, Prediction Time: {eval_results['scratch']['prediction_time']:.4f}s")
    print(f"PyTorch - Accuracy: {eval_results['pytorch']['accuracy']:.4f}, Prediction Time: {eval_results['pytorch']['prediction_time']:.4f}s")
    print(f"Prediction Speed Improvement: {eval_results['scratch']['prediction_time']/eval_results['pytorch']['prediction_time']:.2f}x faster with PyTorch")
    
    # Plot comparison
    plot_results(
        scratch_history, 
        torch_history, 
        scratch_time, 
        torch_time,
        eval_results['scratch']['accuracy'],
        eval_results['pytorch']['accuracy']
    )

if __name__ == "__main__":
    run_comparison()