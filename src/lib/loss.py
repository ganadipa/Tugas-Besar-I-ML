import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    """Base class for loss functions"""
    @abstractmethod
    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def error(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MSE(Loss):
    """Mean Squared Error loss"""
    def __init__(self) -> None:
        pass

    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 0.5 * np.mean((y_true - y_pred) ** 2)
    
    def error(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.function(y_true, y_pred)


class BCE(Loss):
    """Binary Cross-Entropy loss"""
    def __init__(self) -> None:
        pass

    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def error(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.function(y_true, y_pred)


class CCE:
    """Categorical Cross-Entropy loss"""
    def __init__(self):
        pass

    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss
    
    def error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:        
        return self.function(y_true, y_pred)