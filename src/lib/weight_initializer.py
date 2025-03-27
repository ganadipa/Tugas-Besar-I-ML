import numpy as np
from abc import ABC, abstractmethod

# file: weight_initializer.py
class WeightInitializer(ABC):
    """Abstract base class for neural network weight initialization."""
    
    @abstractmethod
    def initialize(self, shape):
        raise NotImplementedError


class ZeroInitializer(WeightInitializer):
    """Initialize weights with zeros."""
    
    def initialize(self, shape):
        """
        Initialize weights with zeros.
        
        Args:
            shape (tuple): Shape of the weight matrix/tensor to initialize.
            (rows, cols) for 2D weights, (rows, cols, channels) for 3D weights.
        Returns:
            numpy.ndarray: Zero-initialized weights.
        """
        return np.zeros(shape)


class UniformInitializer(WeightInitializer):
    """Initialize weights with random values from a uniform distribution."""
    
    def __init__(self, low=-0.05, high=0.05, seed=None):
        """
        Args:
            low (float): Lower bound of the uniform distribution.
            high (float): Upper bound of the uniform distribution.
            seed (int, optional): Random seed for reproducibility.
        """
        self.low = low
        self.high = high
        self.seed = seed
    

    def initialize(self, shape):
        """
        Initialize weights with random values from a uniform distribution.
        
        Args:
            shape (tuple): Shape of the weight matrix/tensor to initialize.
            (rows, cols) for 2D weights, (rows, cols, channels) for 3D weights.
        """
        rng = np.random.RandomState(self.seed)
        return rng.uniform(low=self.low, high=self.high, size=shape)


class NormalInitializer(WeightInitializer):
    """Initialize weights with random values from a normal distribution."""
    
    def __init__(self, mean=0.0, var=0.1, seed=None):
        """
        Args:
            mean (float): Mean of the normal distribution.
            var (float): Variance of the normal distribution.
            seed (int, optional): Random seed for reproducibility.
        """
        self.mean = mean
        self.std = np.sqrt(var)
        self.seed = seed
    
    
    def initialize(self, shape):
        """
        Initialize weights with random values from a normal distribution.
        
        Args:
            shape (tuple): Shape of the weight matrix/tensor to initialize.
            (rows, cols) for 2D weights, (rows, cols, channels) for 3D weights.
        """
        rng = np.random.RandomState(self.seed)
        return rng.normal(loc=self.mean, scale=self.std, size=shape)

class XavierInitializer(WeightInitializer):
    """
    Xavier/Glorot initialization for weights.
    """
    
    def __init__(self, distribution='uniform', seed=None):
        """
        Args:
            distribution: 'uniform' or 'normal' distribution to use
            seed: Random seed for reproducibility
        """
        self.distribution = distribution
        self.seed = seed
    
    def initialize(self, shape):
        """
        Initialize weights with Xavier/Glorot initialization.
        
        Args:
            shape (tuple): Shape of the weight matrix/tensor to initialize.
            For dense layers, shape is (fan_out, fan_in).
        
        Returns:
            numpy.ndarray: Xavier-initialized weights.
        """
        fan_in = shape[1] if len(shape) >= 2 else shape[0]
        fan_out = shape[0] if len(shape) >= 2 else shape[0]
        
        rng = np.random.RandomState(self.seed)
        
        if self.distribution == 'uniform':
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=shape)
        else:
            std = np.sqrt(2.0 / (fan_in + fan_out))
            return rng.normal(0, std, size=shape)


class HeInitializer(WeightInitializer):
    """
    He initialization for weights.
    """
    
    def __init__(self, distribution='normal', seed=None):
        """
        Args:
            distribution: 'uniform' or 'normal' distribution to use
            seed: Random seed for reproducibility
        """
        self.distribution = distribution
        self.seed = seed
    
    def initialize(self, shape):
        fan_in = shape[1] if len(shape) >= 2 else shape[0]
        
        rng = np.random.RandomState(self.seed)
        
        if self.distribution == 'uniform':
            limit = np.sqrt(6.0 / fan_in)
            return rng.uniform(-limit, limit, size=shape)
        else:
            std = np.sqrt(2.0 / fan_in)
            return rng.normal(0, std, size=shape)