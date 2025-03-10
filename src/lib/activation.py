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
        s = softmax_output.reshape(-1, 1)
        ret = np.diagflat(s) - np.dot(s, s.T)
        return ret

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

