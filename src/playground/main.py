import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lib.activation import Tanh, Linear, ReLU, Sigmoid, Softmax
from lib.loss import CCE

activation = Softmax()
print(activation.function(np.array([1, -2, 3])))
print(activation.derivative(np.array([1, -2, 3])))



# Contoh y_true (One-Hot Encoding)
y_true = np.array([
    [1, 0, 0],  # Sampel 1 adalah kelas 0
    [0, 1, 0],  # Sampel 2 adalah kelas 1
    [0, 0, 1]   # Sampel 3 adalah kelas 2
])

# Contoh y_pred (Probabilitas dari Softmax)
y_pred = np.array([
    [0.7, 0.2, 0.1],  # Prediksi untuk Sampel 1
    [0.1, 0.6, 0.3],  # Prediksi untuk Sampel 2
    [0.2, 0.3, 0.5]   # Prediksi untuk Sampel 3
])

cce_loss = CCE()
loss_value = cce_loss.function(y_true, y_pred)
print(f"Categorical Cross-Entropy Loss: {loss_value}")