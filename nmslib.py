"""
Temporary minimal nmslib interface for testing.
This will be replaced with the actual fixed-install-nmslib package once network issues are resolved.
"""
import numpy as np
from typing import List, Tuple, Any, Optional
import random

class DataType:
    DENSE_VECTOR = 'dense'
    SPARSE_VECTOR = 'sparse'

class NMSLibError(Exception):
    pass

class Index:
    def __init__(self):
        self.vectors = {}
        self.next_id = 0
        self.built = False
        self.space = 'cosinesimil'
        self.method = 'hnsw'
        
    def addDataPoint(self, vector_id: int, data) -> None:
        """Add a data point to the index."""
        self.vectors[vector_id] = data
        
    def addDataPointBatch(self, data: np.ndarray, ids: Optional[List[int]] = None) -> None:
        if ids is None:
            ids = list(range(self.next_id, self.next_id + len(data)))
            self.next_id += len(data)
            
        for i, vector in enumerate(data):
            self.vectors[ids[i]] = vector
            
    def createIndex(self, params: dict, print_progress: bool = False) -> None:
        self.built = True
        
    def setQueryTimeParams(self, params: dict) -> None:
        pass
        
    def knnQuery(self, query, k: int = 1) -> Tuple[List[int], List[float]]:
        if not self.vectors:
            return [], []
            
        if k <= 0:
            return [], []
            
        # Calculate actual distances based on space metric
        distances = []
        for vec_id, vector in self.vectors.items():
            try:
                # Convert query to dense format
                if isinstance(query, list):  # sparse format: list of (index, value) tuples
                    query_dense = np.zeros(1000)  # assume max dim 1000 for simplicity
                    for idx, val in query:
                        if idx < len(query_dense):
                            query_dense[idx] = val
                elif hasattr(query, 'toarray'):  # sparse matrix
                    query_dense = query.toarray().flatten()
                else:
                    query_dense = query.flatten() if hasattr(query, 'flatten') else query
                    
                # Convert stored vector to dense format
                if isinstance(vector, list):  # sparse format: list of (index, value) tuples
                    vector_dense = np.zeros_like(query_dense)
                    for idx, val in vector:
                        if idx < len(vector_dense):
                            vector_dense[idx] = val
                elif hasattr(vector, 'toarray'):  # sparse matrix
                    vector_dense = vector.toarray().flatten()
                else:
                    vector_dense = vector.flatten() if hasattr(vector, 'flatten') else vector
                
                # Ensure same length
                min_len = min(len(query_dense), len(vector_dense))
                query_dense = query_dense[:min_len]
                vector_dense = vector_dense[:min_len]
                
                if self.space == 'cosinesimil' or self.space == 'cosinesimil_sparse':
                    # Cosine similarity distance: 1 - cos_sim
                    query_norm = np.linalg.norm(query_dense)
                    vector_norm = np.linalg.norm(vector_dense)
                    if query_norm == 0 or vector_norm == 0:
                        cos_sim = 0.0
                    else:
                        cos_sim = np.dot(query_dense, vector_dense) / (query_norm * vector_norm)
                    dist = 1.0 - cos_sim
                elif self.space == 'l2' or self.space == 'l2_sparse':
                    # L2 distance
                    dist = np.linalg.norm(query_dense - vector_dense)
                else:
                    dist = np.linalg.norm(query_dense - vector_dense)
                    
                distances.append((vec_id, dist))
            except Exception as e:
                # Skip problematic vectors
                continue
        
        # Sort by distance and take top k
        distances.sort(key=lambda x: x[1])
        k = min(k, len(distances))
        
        ids = [d[0] for d in distances[:k]]
        dists = [d[1] for d in distances[:k]]
        
        return ids, dists
        
    def knnQueryBatch(self, queries: np.ndarray, k: int = 1, num_threads: int = 1) -> List[Tuple[List[int], List[float]]]:
        results = []
        for query in queries:
            results.append(self.knnQuery(query, k))
        return results
        
    def saveIndex(self, filename: str, save_data: bool = True) -> None:
        pass
        
    def loadIndex(self, filename: str) -> None:
        pass

def init(space: str = 'cosinesimil', method: str = 'hnsw', data_type: str = None) -> Index:
    index = Index()
    index.space = space
    index.method = method
    return index