from abc import ABC, abstractmethod
import os
import pickle
import time
from typing import Any, Tuple, Dict, Optional

from sklearn.datasets import fetch_openml


class DataRepository(ABC):
    @abstractmethod
    def get(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
    
    @abstractmethod
    def save(self, dataset_name: str, data: Dict[str, Any]) -> None:
        raise NotImplementedError

class OnlineDataRepository(DataRepository):
    def __init__(self, cache_dir: str = ".data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get(self, dataset_name: str) -> Dict[str, Any]:
        t0 = time.time()
        
        
        X, y = fetch_openml(dataset_name, version=1, return_X_y=True, as_frame=False)
        data = {'X': X, 'y': y}
        
        fetch_time = time.time() - t0
        data['metadata'] = {
            'dataset_name': dataset_name,
            'version': 1,
            'fetch_time': fetch_time
        }
        
        return data
    
    def save(self, dataset_name: str, data: Dict[str, Any]) -> None:
        cache_path = os.path.join(self.cache_dir, f"{dataset_name}.pkl")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)


class OfflineDataRepository(DataRepository):
    def __init__(self, cache_dir: str = ".data_cache"):
        self.cache_dir = cache_dir
    
    def get(self, dataset_name: str) -> Dict[str, Any]:
        cache_path = os.path.join(self.cache_dir, f"{dataset_name}.pkl")
        
        if not os.path.exists(cache_path):
            error_msg = f"Dataset '{dataset_name}' not found in offline storage"
            raise FileNotFoundError(error_msg)
        
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        return data

    def save(self, dataset_name: str, data: Dict[str, Any]) -> None:
        cache_path = os.path.join(self.cache_dir, f"{dataset_name}.pkl")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        

    


class DataRepositoryFactory:
    """Factory for creating data repositories."""
    
    @staticmethod
    def create_repository(repository_type: str, cache_dir: str = ".data_cache") -> DataRepository:
        if repository_type.lower() == "online":
            return OnlineDataRepository(cache_dir)
        elif repository_type.lower() == "offline":
            return OfflineDataRepository(cache_dir)
        else:
            raise ValueError(f"Invalid repository type: {repository_type}")
        
def get_mnist_data():
    try:
        repo = DataRepositoryFactory.create_repository("offline")
        data = repo.get("mnist_784")
        print("Loaded MNIST data from offline storage")
    except FileNotFoundError:
        repo = DataRepositoryFactory.create_repository("online")
        data = repo.get("mnist_784")
        repo.save("mnist_784", data)
        print("Saved MNIST data to offline storage")
    
    return data


