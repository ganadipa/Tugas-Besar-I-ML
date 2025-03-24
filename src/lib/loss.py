import numpy as np
from abc import ABC, abstractmethod

EPSILON = 1e-15

class Loss(ABC):
    @abstractmethod
    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MSE(Loss):
    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=0))
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        batch_size = y_true.shape[1] if y_true.ndim > 1 else 1
        return (-2/batch_size) * (y_true - y_pred)


class BCE(Loss):
    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, EPSILON, 1.0 - EPSILON)
        batch_size = y_true.shape[1] if y_true.ndim > 1 else 1
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / batch_size
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, EPSILON, 1.0 - EPSILON)
        batch_size = y_true.shape[1] if y_true.ndim > 1 else 1
        return (-1/batch_size) * (y_true / y_pred - (1 - y_true) / (1 - y_pred))


class CCE(Loss):
    def function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, EPSILON, 1.0)
        batch_size = y_true.shape[1] if y_true.ndim > 1 else 1
        return -np.sum(y_true * np.log(y_pred)) / batch_size
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, EPSILON, 1.0)
        batch_size = y_true.shape[1] if y_true.ndim > 1 else 1
        return (-1/batch_size) * (y_true / y_pred)