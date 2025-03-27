import numpy as np
from abc import ABC, abstractmethod
from typing import Callable


class Activation(ABC):
    """Base class for activation functions"""
    @abstractmethod
    def function(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    

class Linear(Activation):
    """Linear activation function"""
    def __init__(self) -> None:
        pass

    def function(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

class ReLU(Activation):
    """ReLU activation function: f(x) = max(0, x)"""
    def __init__(self) -> None:
        pass

    def function(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)


class Sigmoid(Activation):
    """Sigmoid activation function: f(x) = 1 / (1 + e^(-x))"""
    def __init__(self) -> None:
        pass

    def function(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        fx = self.function(x)
        return fx * (1 - fx)


class Tanh(Activation):
    """Hyperbolic tangent activation function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))"""
    def __init__(self) -> None:
        pass

    def function(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class Softmax(Activation):
    """Softmax activation function: f(x_i) = e^x_i / sum(e^x_j)"""
    def __init__(self) -> None:
        pass

    def function(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        softmax_output = self.function(x)  
        
        output_sz, batch_sz = softmax_output.shape
        
        result = np.zeros((output_sz, output_sz, batch_sz))
        
        for i in range(batch_sz):
            s = softmax_output[:, i]
            result[:, :, i] = np.diag(s) - np.outer(s, s)
        
        return result

class CustomActivation(Activation):
    """Custom activation function"""
    fn: Callable

    def __init__(self, fn: Callable) -> None:
        self.fn = fn
        pass

    def function(self, x: np.ndarray) -> np.ndarray:
        return self.fn(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        # TODO: AldyPy
        pass

class ELU(Activation):
    """Exponential Linear Unit: f(x) = x if x > 0 else alpha * (exp(x) - 1)"""
    
    def __init__(self, alpha=1.0) -> None:
        self.alpha = alpha

    def function(self, x: np.ndarray) -> np.ndarray:
        # Apply clipping to prevent overflow
        x_safe = np.clip(x, -500, 500)
        return np.where(x_safe > 0, x_safe, self.alpha * (np.exp(x_safe) - 1))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        x_safe = np.clip(x, -500, 500)
        return np.where(x_safe > 0, 1, self.alpha * np.exp(x_safe))


class GELU(Activation):    
    def __init__(self) -> None:
        pass

    def function(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        sqrt_2_over_pi = np.sqrt(2 / np.pi)
        tanh_term = np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3))
        sech_squared = 1 - tanh_term**2
        
        inner_derivative = sqrt_2_over_pi * (1 + 3 * 0.044715 * x**2)
        
        return 0.5 * (1 + tanh_term) + 0.5 * x * sech_squared * inner_derivative
