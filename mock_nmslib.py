"""
Mock implementation of nmslib for testing purposes.
This mock provides basic functionality to enable tests without requiring the full nmslib installation.
"""
import numpy as np
from typing import List, Tuple, Any, Optional
import random

class DataType:
    """Mock DataType class."""
    DENSE_VECTOR = 'dense'
    SPARSE_VECTOR = 'sparse'

class NMSLibError(Exception):
    """Mock NMSLib exception."""
    pass

class MockIndex:
    """Mock NMSLIB index for testing."""
    
    def __init__(self, space: str = 'cosinesimil', method: str = 'hnsw'):
        self.space = space
        self.method = method
        self.vectors = {}
        self.next_id = 0
        self.built = False
        
    def init(self, space: str, method: str = 'hnsw'):
        """Initialize the index with space and method."""
        self.space = space
        self.method = method
        
    def addDataPoint(self, vector_id: int, data: np.ndarray) -> None:
        """Add a data point to the index."""
        self.vectors[vector_id] = data
        
    def addDataPointBatch(self, data: np.ndarray, ids: Optional[List[int]] = None) -> None:
        """Add a batch of data points."""
        if ids is None:
            ids = list(range(self.next_id, self.next_id + len(data)))
            self.next_id += len(data)
            
        for i, vector in enumerate(data):
            self.vectors[ids[i]] = vector
            
    def createIndex(self, params: dict, print_progress: bool = False) -> None:
        """Create the index with given parameters."""
        self.built = True
        
    def setQueryTimeParams(self, params: dict) -> None:
        """Set query time parameters."""
        pass
        
    def knnQuery(self, query: np.ndarray, k: int = 1) -> Tuple[List[int], List[float]]:
        """Perform k-nearest neighbor query."""
        if not self.vectors:
            return [], []
            
        if k <= 0:
            return [], []
            
        # Simple mock implementation: return random vectors with mock distances
        available_ids = list(self.vectors.keys())
        k = min(k, len(available_ids))
        
        # Shuffle and take first k for randomness
        random.shuffle(available_ids)
        selected_ids = available_ids[:k]
        
        # Return mock distances (just random values between 0 and 1)
        distances = [random.random() for _ in selected_ids]
        
        return selected_ids, distances
        
    def knnQueryBatch(self, queries: np.ndarray, k: int = 1, num_threads: int = 1) -> List[Tuple[List[int], List[float]]]:
        """Perform batch k-nearest neighbor queries."""
        results = []
        for query in queries:
            results.append(self.knnQuery(query, k))
        return results
        
    def saveIndex(self, filename: str, save_data: bool = True) -> None:
        """Save index to file (mock - does nothing)."""
        pass
        
    def loadIndex(self, filename: str) -> None:
        """Load index from file (mock - does nothing)."""
        pass

def init(space: str = 'cosinesimil', method: str = 'hnsw', data_type: str = None) -> MockIndex:
    """Initialize and return a mock index."""
    index = MockIndex(space, method)
    index.init(space, method)
    return index