import os
import uuid
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import lmdb
import msgpack
import nmslib
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix
import threading
import shutil

class VStore:
    """A vector database using NMSLIB for ANN search, LMDB for storage, and MessagePack for serialization.

    Supports dense (np.ndarray) and sparse (csr_matrix) vectors with metadata filtering and persistence.
    The value field can be any MessagePack-serializable type (e.g., str, int, list, dict).

    Args:
        db_path (str): Path to the LMDB database directory.
        vector_type (str): 'dense' or 'sparse' vectors. Default: 'dense'.
        space (str): Distance metric for NMSLIB (e.g., 'cosinesimil', 'l2'). Default: 'cosinesimil'.
        map_size (int): Initial LMDB map size in bytes. Default: 1e9.
        hnsw_params (Dict[str, Any]): HNSW parameters for NMSLIB. Default: {'M': 16, 'efConstruction': 200, 'post': 2}.
        query_params (Dict[str, Any]): Query-time parameters for NMSLIB. Default: {'efSearch': 100}.
        rebuild_threshold (float): Fraction of modified points triggering index rebuild. Default: 0.1.
        max_workers (int): Number of threads for parallel operations. Default: cpu_count().
        max_map_size (int): Maximum LMDB map size in bytes. Default: 2**40 (1TB).
    """
    def __init__(self, db_path: str, vector_type: str = 'dense', space: str = 'cosinesimil',
                 map_size: int = int(1e9), hnsw_params: Dict[str, Any] = None,
                 query_params: Dict[str, Any] = None, rebuild_threshold: float = 0.15,
                 max_workers: int = cpu_count(), max_map_size: int = 2**40):
        self.db_path = db_path
        self.vector_type = vector_type
        self.space = space if vector_type == 'dense' else f"{space}_sparse"
        self.env = lmdb.open(db_path, map_size=map_size, max_dbs=4, writemap=True)
        self.hnsw_params = hnsw_params or {'M': 16, 'efConstruction': 200, 'post': 2}
        self.query_params = query_params or {'efSearch': 100}
        self.rebuild_threshold = rebuild_threshold
        self.max_map_size = max_map_size
        self.index = nmslib.init(method='hnsw', space=self.space,
                                data_type=nmslib.DataType.DENSE_VECTOR if vector_type == 'dense' else nmslib.DataType.SPARSE_VECTOR)
        self.index_path = os.path.join(db_path, "index.nms")
        self.modifications_since_rebuild = 0
        self.index_initialized = False
        self.vector_dim = None
        self.logger = logging.getLogger(__name__)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.index_lock = threading.Lock()

        # Initialize databases
        with self.env.begin(write=True) as txn:
            self.db_data = self.env.open_db(b'data', txn=txn)
            self.db_index_mapping = self.env.open_db(b'index_mapping', txn=txn)
            self.db_metadata_index = self.env.open_db(b'metadata_index', txn=txn)
            self.db_system_metadata = self.env.open_db(b'system_metadata', txn=txn)

            # Initialize system metadata if not exists
            if txn.get(b'total_points', db=self.db_system_metadata) is None:
                txn.put(b'total_points', msgpack.packb(0, use_bin_type=True), db=self.db_system_metadata)
                txn.put(b'vector_dim', msgpack.packb(None, use_bin_type=True), db=self.db_system_metadata)
                txn.put(b'next_index', msgpack.packb(0, use_bin_type=True), db=self.db_system_metadata)

        # Load existing state
        self._load_state()

    def _default(self, obj):
        """Custom serialization for MessagePack."""
        if isinstance(obj, np.ndarray):
            return msgpack.ExtType(1, msgpack.packb({
                'data': obj.tobytes(),
                'shape': obj.shape,
                'dtype': str(obj.dtype)
            }, use_bin_type=True))
        elif isinstance(obj, csr_matrix):
            if obj.nnz == 0:
                raise ValueError("Sparse matrix must be non-empty")
            return msgpack.ExtType(2, msgpack.packb({
                'data': obj.data.tobytes(),
                'indices': obj.indices.tobytes(),
                'indptr': obj.indptr.tobytes(),
                'shape': obj.shape,
                'dtype': str(obj.dtype)
            }, use_bin_type=True))
        return obj

    def _ext_hook(self, code, data):
        """Custom deserialization for MessagePack."""
        if code == 1:
            unpacked = msgpack.unpackb(data, raw=False, use_list=False)
            return np.frombuffer(unpacked['data'], dtype=unpacked['dtype']).reshape(unpacked['shape'])
        elif code == 2:
            unpacked = msgpack.unpackb(data, raw=False, use_list=False)
            return csr_matrix((np.frombuffer(unpacked['data'], dtype=unpacked['dtype']),
                              np.frombuffer(unpacked['indices'], dtype=np.int32),
                              np.frombuffer(unpacked['indptr'], dtype=np.int32)),
                             shape=unpacked['shape'])
        return msgpack.ExtType(code, data)

    def _load_state(self):
        """Load the state of the database and index."""
        try:
            if os.path.exists(self.index_path):
                with self.index_lock:
                    self.index.loadIndex(self.index_path, load_data=True)
                    self.index.setQueryTimeParams(self.query_params)
                    self.index_initialized = True
                self.logger.info(f"Loaded NMSLIB index from {self.index_path}")
            else:
                self.logger.debug("No existing index found, initializing new index")

            # Load vector dimensionality
            with self.env.begin(write=False, buffers=True) as txn:
                vector_dim = msgpack.unpackb(txn.get(b'vector_dim', db=self.db_system_metadata), raw=False)
                if vector_dim is None:
                    # Check existing vectors to set dimension
                    cursor = txn.cursor(db=self.db_data)
                    for key, value in cursor:
                        if not key.startswith(b'deleted:'):
                            try:
                                data = msgpack.unpackb(value, raw=False, ext_hook=self._ext_hook, use_list=False)
                                vector = data['vector']
                                self.vector_dim = vector.shape[0] if self.vector_type == 'dense' else vector.shape[1]
                                with self.env.begin(write=True) as txn2:
                                    txn2.put(b'vector_dim', msgpack.packb(self.vector_dim, use_bin_type=True), db=self.db_system_metadata)
                                break
                            except Exception as e:
                                self.logger.error(f"Error reading vector dimension: {e}")
                                continue
                else:
                    self.vector_dim = vector_dim

            # Preload index if not already loaded
            if not self.index_initialized:
                with self.index_lock:
                    vectors = []
                    indices = []

                    # Scan all non-deleted documents to rebuild index if needed
                    with self.env.begin(write=False) as txn:
                        # First get all non-deleted keys and their indices
                        key_to_idx = {}
                        cursor = txn.cursor(db=self.db_index_mapping)
                        cursor.set_range(b'key:')
                        for key, value in cursor:
                            if key.startswith(b'key:'):
                                doc_key = key[4:].decode('utf-8')
                                idx = msgpack.unpackb(value, raw=False)
                                key_to_idx[doc_key] = idx

                        # Check which keys are deleted
                        deleted_keys = set()
                        cursor = txn.cursor(db=self.db_data)
                        cursor.set_range(b'deleted:')
                        for key, _ in cursor:
                            if key.startswith(b'deleted:'):
                                doc_key = key[8:].decode('utf-8')
                                deleted_keys.add(doc_key)

                        # Collect vectors for non-deleted keys
                        for doc_key, idx in key_to_idx.items():
                            if doc_key not in deleted_keys:
                                data_value = txn.get(doc_key.encode('utf-8'), db=self.db_data)
                                if data_value:
                                    try:
                                        data = msgpack.unpackb(data_value, raw=False, ext_hook=self._ext_hook, use_list=False)
                                        vector = data['vector']
                                        vector_to_add = self._prepare_vector(vector)
                                        vectors.append(vector_to_add)
                                        indices.append(idx)
                                    except Exception as e:
                                        self.logger.error(f"Error processing vector for key {doc_key}: {e}")

                    # Add vectors to index if any were found
                    if vectors:
                        self.index.addDataPointBatch(vectors, indices)
                        self.index.createIndex(self.hnsw_params, print_progress=True)
                        self.index.setQueryTimeParams(self.query_params)
                        self.index_initialized = True
                    else:
                        self.logger.warning("No valid vectors found during index loading")

        except lmdb.Error as e:
            self.logger.error(f"LMDB error during load: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}. Rebuilding index.")
            with self.env.begin(write=True, buffers=True) as txn:
                self._rebuild_index(txn)
                self._save_state(txn)

    def _prepare_vector(self, vector: Union[np.ndarray, csr_matrix]) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Prepare a vector for insertion into the index."""
        if self.vector_type == 'dense':
            if not isinstance(vector, np.ndarray):
                raise ValueError("Dense vector must be a NumPy array")
            if vector.size == 0:
                raise ValueError("Vector cannot be empty")
            if self.vector_dim is not None and vector.shape[0] != self.vector_dim:
                self.logger.error(f"Dense vector dimension {vector.shape[0]} does not match expected {self.vector_dim}")
                raise ValueError(f"Dense vector dimension {vector.shape[0]} must match expected {self.vector_dim}")
            if vector.dtype != np.float32:
                self.logger.debug("Converting dense vector to float32")
                vector = vector.astype(np.float32)
            return vector
        elif self.vector_type == 'sparse':
            if not isinstance(vector, csr_matrix) or vector.nnz == 0:
                raise ValueError("Sparse vector must be a non-empty SciPy CSR matrix")
            if self.vector_dim is not None and vector.shape[1] != self.vector_dim:
                self.logger.error(f"Sparse vector dimension {vector.shape[1]} does not match expected {self.vector_dim}")
                raise ValueError(f"Sparse vector dimension {vector.shape[1]} must match expected {self.vector_dim}")
            if vector.dtype != np.float32:
                self.logger.debug("Converting sparse vector data to float32")
                vector = csr_matrix((vector.data.astype(np.float32), vector.indices, vector.indptr), shape=vector.shape)
            return (vector.indices, vector.data)
        raise ValueError(f"Unsupported vector_type: {self.vector_type}")

    def _update_metadata_indexes(self, key: str, metadata: Dict[str, Any], add: bool = True, txn=None):
        """Update the metadata indexes for a given key."""
        close_txn = False
        if txn is None:
            with self.env.begin(write=True) as txn:
                self._update_metadata_indexes(key, metadata, add, txn)
            return

        for meta_key, meta_value in metadata.items():
            if isinstance(meta_value, list):
                for val in meta_value:
                    composite_key = f"meta:{meta_key}:{val}:{key}".encode('utf-8')
                    if add:
                        txn.put(composite_key, b'', db=self.db_metadata_index)
                    else:
                        txn.delete(composite_key, db=self.db_metadata_index)
            else:
                composite_key = f"meta:{meta_key}:{meta_value}:{key}".encode('utf-8')
                if add:
                    txn.put(composite_key, b'', db=self.db_metadata_index)
                else:
                    txn.delete(composite_key, db=self.db_metadata_index)

                # For numeric metadata, store with padded values for range queries
                if isinstance(meta_value, (int, float)):
                    padded_value = f"{float(meta_value):020.10f}"
                    numeric_key = f"meta:{meta_key}:{padded_value}:{key}".encode('utf-8')
                    if add:
                        txn.put(numeric_key, b'', db=self.db_metadata_index)
                    else:
                        txn.delete(numeric_key, db=self.db_metadata_index)

    def _filter_keys(self, filter: Dict[str, Any], txn) -> set:
        """Filter keys based on metadata conditions."""
        matched_keys = set()

        def process_condition(condition):
            if isinstance(condition, dict) and 'op' in condition:
                op = condition['op']
                conditions = condition['conditions']
                result_sets = [process_condition(cond) for cond in conditions]
                if op == 'AND':
                    return set.intersection(*result_sets) if result_sets else set()
                else:  # OR
                    return set.union(*result_sets) if result_sets else set()
            else:
                condition_sets = []
                for key, condition in condition.items():
                    key_set = set()
                    if isinstance(condition, list) and len(condition) == 2 and isinstance(condition[0], (int, float)):
                        # Range query for numeric values
                        cursor = txn.cursor(db=self.db_metadata_index)
                        start_key = f"meta:{key}:{float(condition[0]):020.10f}:".encode('utf-8')
                        upper_bound = condition[1] + 1e-10
                        end_key = f"meta:{key}:{float(upper_bound):020.10f}:".encode('utf-8')

                        cursor.set_range(start_key)
                        while cursor.key() and cursor.key() <= end_key:
                            try:
                                doc_key = cursor.key().decode('utf-8').split(':', 3)[-1]
                                key_set.add(doc_key)
                            except (ValueError, IndexError):
                                pass
                            cursor.next()
                    else:
                        # Handle individual values or lists of values
                        values = condition if isinstance(condition, list) else [condition]
                        for value in values:
                            prefix = f"meta:{key}:{value}:".encode('utf-8')
                            cursor = txn.cursor(db=self.db_metadata_index)
                            cursor.set_range(prefix)
                            while cursor.key() and cursor.key().startswith(prefix):
                                try:
                                    doc_key = cursor.key().decode('utf-8').split(':', 3)[-1]
                                    key_set.add(doc_key)
                                except (ValueError, IndexError):
                                    pass
                                cursor.next()
                    condition_sets.append(key_set)

                # Combine conditions within a single filter with AND logic
                if condition_sets:
                    return set.intersection(*condition_sets)
                return set()

        # Process the filter structure
        if isinstance(filter, dict) and 'op' in filter:
            # Complex filter with AND/OR operations
            result = process_condition(filter)
        else:
            # Simple filter
            result = process_condition(filter)

        # Exclude deleted keys
        deleted_keys = set()
        cursor = txn.cursor(db=self.db_data)
        cursor.set_range(b'deleted:')
        for key, _ in cursor:
            if key.startswith(b'deleted:'):
                deleted_keys.add(key[8:].decode('utf-8'))

        # Filter out deleted keys
        final_result = result - deleted_keys
        return final_result

    def _resize_if_needed(self, txn):
        """Resize the LMDB map if needed."""
        info = self.env.info()
        stat = self.env.stat()
        used = (info['last_pgno'] + 1) * stat['psize']
        if used > info['map_size'] * 0.8:
            new_size = min(info['map_size'] * 2, self.max_map_size)
            try:
                self.env.set_mapsize(new_size)
                self.logger.info(f"Resized map_size to {new_size} bytes")
            except lmdb.Error as e:
                self.logger.error(f"Failed to resize map_size: {e}. Current size: {info['map_size']}")
                raise

    def _rebuild_index(self, txn):
        """Rebuild the NMSLIB index from current data."""
        start_time = time.time()
        self.logger.info("Rebuilding NMSLIB index")

        # Clear current index
        with self.index_lock:
            self.index = nmslib.init(method='hnsw', space=self.space,
                                    data_type=nmslib.DataType.DENSE_VECTOR if self.vector_type == 'dense' else nmslib.DataType.SPARSE_VECTOR)

        # Collect all non-deleted vectors and their indices
        vectors = []
        indices = []

        # First pass: collect all keys and their indices
        key_to_idx = {}
        cursor = txn.cursor(db=self.db_index_mapping)
        cursor.set_range(b'key:')
        for key, value in cursor:
            if key.startswith(b'key:'):
                doc_key = key[4:].decode('utf-8')
                idx = msgpack.unpackb(value, raw=False)
                key_to_idx[doc_key] = idx

        # Second pass: collect all deleted keys
        deleted_keys = set()
        cursor = txn.cursor(db=self.db_data)
        cursor.set_range(b'deleted:')
        for key, _ in cursor:
            if key.startswith(b'deleted:'):
                doc_key = key[8:].decode('utf-8')
                deleted_keys.add(doc_key)

        # Third pass: collect all valid vectors
        for doc_key, idx in key_to_idx.items():
            if doc_key not in deleted_keys:
                # Get the data
                data_value = txn.get(doc_key.encode('utf-8'), db=self.db_data)
                if data_value:
                    try:
                        data = msgpack.unpackb(data_value, raw=False, ext_hook=self._ext_hook, use_list=False)
                        vector = data['vector']
                        vector_to_add = self._prepare_vector(vector)
                        vectors.append(vector_to_add)
                        indices.append(idx)
                    except Exception as e:
                        self.logger.error(f"Error processing vector for key {doc_key}: {e}")

        # Add vectors to index
        if vectors:
            with self.index_lock:
                self.index.addDataPointBatch(vectors, indices)
                self.index.createIndex(self.hnsw_params, print_progress=True)
                self.index.setQueryTimeParams(self.query_params)
                self.index_initialized = True
        else:
            self.index_initialized = False

        self.modifications_since_rebuild = 0
        self.logger.info(f"Index rebuild completed in {time.time() - start_time:.2f} seconds with {len(vectors)} vectors")

    def _save_state(self, txn):
        """Save the state of the index to disk."""
        if not self.index_initialized:
            self.logger.debug("Skipping index save: index is not initialized")
            return
        try:
            with self.index_lock:
                self.index.saveIndex(self.index_path, save_data=True)
            self.logger.info(f"Saved NMSLIB index to {self.index_path}")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            raise ValueError(f"Failed to save state: {e}")

    def _get_data(self, key: str, txn, fields=None) -> Dict[str, Any]:
        """Get data for a given key, with optional field selection."""
        if key is None:
            raise ValueError("Key cannot be None")

        # Check if key is deleted
        if txn.get(f"deleted:{key}".encode('utf-8'), db=self.db_data):
            raise KeyError(f"Key '{key}' has been deleted")

        # Get the serialized data
        value = txn.get(key.encode('utf-8'), db=self.db_data)
        if value is None:
            raise KeyError(f"Key '{key}' not found")

        # Deserialize the data
        data = msgpack.unpackb(value, raw=False, ext_hook=self._ext_hook, use_list=False)

        if fields is None or 'all' in fields:
            return data
        else:
            return {k: v for k, v in data.items() if k in fields}

    def put(self, vector: Union[np.ndarray, csr_matrix], value: Any,
            metadata: Optional[Dict[str, Any]] = None, key: Optional[str] = None) -> str:
        """Add a vector, value, and optional metadata to the store."""
        start_time = time.time()
        try:
            msgpack.packb(value, use_bin_type=True, default=self._default)
        except Exception as e:
            raise ValueError(f"Value must be MessagePack-serializable: {e}")

        vector_to_add = self._prepare_vector(vector)

        if key is None:
            max_retries = 5
            for i in range(max_retries):
                key = str(uuid.uuid4())
                with self.env.begin() as txn:
                    if not txn.get(f"key:{key}".encode('utf-8'), db=self.db_index_mapping):
                        break
                if i == max_retries - 1:
                    raise ValueError(f"Failed to generate unique key after {max_retries} retries")

        with self.env.begin(write=True, buffers=True) as txn:
            self._resize_if_needed(txn)

            # Check if key already exists
            if txn.get(f"key:{key}".encode('utf-8'), db=self.db_index_mapping):
                raise ValueError(f"Key '{key}' already exists. Use update() to modify.")

            # Get next index
            idx_data = txn.get(b'next_index', db=self.db_system_metadata, default=msgpack.packb(0, use_bin_type=True))
            idx = msgpack.unpackb(idx_data, raw=False)
            txn.put(b'next_index', msgpack.packb(idx + 1, use_bin_type=True), db=self.db_system_metadata)

            # Store index mappings
            txn.put(f"key:{key}".encode('utf-8'), msgpack.packb(idx, use_bin_type=True), db=self.db_index_mapping)
            txn.put(f"index:{idx}".encode('utf-8'), key.encode('utf-8'), db=self.db_index_mapping)

            # Prepare and store the data
            data = {
                'vector': vector,
                'value': value,
                'metadata': metadata or {}
            }
            serialized_data = msgpack.packb(data, use_bin_type=True, default=self._default)
            txn.put(key.encode('utf-8'), serialized_data, db=self.db_data)

            # Update metadata indexes if metadata exists
            if metadata:
                self._update_metadata_indexes(key, metadata, add=True, txn=txn)

            # Update total points count
            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_system_metadata), raw=False)
            txn.put(b'total_points', msgpack.packb(total_points + 1, use_bin_type=True), db=self.db_system_metadata)

        # Update the index outside the transaction to minimize lock time
        if vector_to_add is not None:
            with self.index_lock:
                if not self.index_initialized:
                    self._load_state()  # Ensure index is initialized
                self.index.addDataPoint(idx, vector_to_add)
                if not self.index_initialized:
                    self.index.createIndex(self.hnsw_params, print_progress=True)
                    self.index_initialized = True

        self.modifications_since_rebuild += 1

        # Check if a rebuild is needed
        with self.env.begin(write=False) as txn:
            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_system_metadata), raw=False)
            if total_points > 0 and self.modifications_since_rebuild > self.rebuild_threshold * total_points:
                with self.env.begin(write=True) as txn:
                    self._rebuild_index(txn)
                    self._save_state(txn)

        self.logger.info(f"Put operation completed in {time.time() - start_time:.2f} seconds")
        return key

    def get(self, key: str, fields=None) -> Tuple[Union[np.ndarray, csr_matrix], Any, Dict[str, Any]]:
        """Retrieve a vector, value, and metadata by key."""
        if key is None:
            raise ValueError("Key cannot be None")

        with self.env.begin(write=False, buffers=True) as txn:
            if txn.get(f"deleted:{key}".encode('utf-8'), db=self.db_data):
                raise KeyError(f"Key '{key}' has been deleted")

            data = self._get_data(key, txn, fields)
            return data['vector'], data['value'], data['metadata']

    def delete(self, key: str):
        """Delete an entry by key."""
        if key is None:
            raise ValueError("Key cannot be None")

        start_time = time.time()

        # First, retrieve the existing metadata to delete its indexes
        metadata = None
        with self.env.begin(write=False) as txn:
            if txn.get(key.encode('utf-8'), db=self.db_data) is None or \
               txn.get(f"deleted:{key}".encode('utf-8'), db=self.db_data):
                self.logger.info(f"Key '{key}' not found or already deleted")
                return

            # Get the metadata to delete its indexes later
            data_value = txn.get(key.encode('utf-8'), db=self.db_data)
            if data_value:
                data_dict = msgpack.unpackb(data_value, raw=False, ext_hook=self._ext_hook, use_list=False)
                metadata = data_dict.get('metadata', {})

        # Perform deletion in a single transaction
        with self.env.begin(write=True, buffers=True) as txn:
            self._resize_if_needed(txn)

            # Mark as deleted (soft delete)
            txn.put(f"deleted:{key}".encode('utf-8'), b'1', db=self.db_data)

            # Delete metadata indexes if metadata exists
            if metadata:
                self._update_metadata_indexes(key, metadata, add=False, txn=txn)

            # Update total points count
            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_system_metadata), raw=False)
            if total_points > 0:
                txn.put(b'total_points', msgpack.packb(total_points - 1, use_bin_type=True), db=self.db_system_metadata)

        # Track modification for index rebuild
        self.modifications_since_rebuild += 1

        # Check if a rebuild is needed
        with self.env.begin(write=False) as txn:
            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_system_metadata), raw=False)
            if total_points > 0 and self.modifications_since_rebuild > self.rebuild_threshold * total_points:
                with self.env.begin(write=True) as txn:
                    self._rebuild_index(txn)
                    self._save_state(txn)

        self.logger.info(f"Delete operation completed in {time.time() - start_time:.2f} seconds")

    def update(self, key: str, vector: Optional[Union[np.ndarray, csr_matrix]] = None,
               value: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
        """Update an existing entry with new vector, value, or metadata."""
        if key is None:
            raise ValueError("Key cannot be None")

        start_time = time.time()

        if value is not None:
            try:
                msgpack.packb(value, use_bin_type=True, default=self._default)
            except Exception as e:
                raise ValueError(f"Value must be MessagePack-serializable: {e}")

        vector_to_add = self._prepare_vector(vector) if vector is not None else None

        # First read the existing data to preserve what we're not updating
        existing_data = None
        existing_metadata = None
        idx = None

        with self.env.begin(write=False) as txn:
            if txn.get(f"deleted:{key}".encode('utf-8'), db=self.db_data):
                raise KeyError(f"Key '{key}' has been deleted")

            # Get existing data
            data_value = txn.get(key.encode('utf-8'), db=self.db_data)
            if data_value:
                existing_data = msgpack.unpackb(data_value, raw=False, ext_hook=self._ext_hook, use_list=False)
                existing_metadata = existing_data.get('metadata', {})

            # Get the index for vector updates
            idx_data = txn.get(f"key:{key}".encode('utf-8'), db=self.db_index_mapping)
            if idx_data:
                idx = msgpack.unpackb(idx_data, raw=False)

        # Prepare the updated data
        update_data = existing_data.copy() if existing_data else {}

        if vector is not None:
            update_data['vector'] = vector
            if idx is not None and vector_to_add is not None:
                # Will update the index later
                pass

        if value is not None:
            update_data['value'] = value

        if metadata is not None:
            # Replace the entire metadata
            update_data['metadata'] = metadata

        # Prepare for transaction
        serialized_data = msgpack.packb(update_data, use_bin_type=True, default=self._default)
        metadata_changes = None

        if metadata is not None and metadata != existing_metadata:
            # We need to update metadata indexes
            metadata_changes = {
                'old_metadata': existing_metadata,
                'new_metadata': metadata
            }

        with self.env.begin(write=True, buffers=True) as txn:
            self._resize_if_needed(txn)

            # Store the updated data
            txn.put(key.encode('utf-8'), serialized_data, db=self.db_data)

            # Update metadata indexes if needed
            if metadata_changes:
                # First remove old metadata indexes
                if metadata_changes['old_metadata']:
                    self._update_metadata_indexes(key, metadata_changes['old_metadata'], add=False, txn=txn)
                # Then add new metadata indexes
                self._update_metadata_indexes(key, metadata_changes['new_metadata'], add=True, txn=txn)

        # Update the index for the vector if it was updated
        if vector is not None and idx is not None and vector_to_add is not None:
            with self.index_lock:
                if not self.index_initialized:
                    self._load_state()
                self.index.addDataPoint(idx, vector_to_add)
                if not self.index_initialized:
                    self.index.createIndex(self.hnsw_params, print_progress=True)
                    self.index_initialized = True

        self.modifications_since_rebuild += 1

        # Check if a rebuild is needed
        with self.env.begin(write=False) as txn:
            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_system_metadata), raw=False)
            if total_points > 0 and self.modifications_since_rebuild > self.rebuild_threshold * total_points:
                with self.env.begin(write=True) as txn:
                    self._rebuild_index(txn)
                    self._save_state(txn)

        self.logger.info(f"Update operation completed in {time.time() - start_time:.2f} seconds")

    def search(self, vector: Union[np.ndarray, csr_matrix], top_k: int = 5,
               filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for the nearest neighbors of a given vector."""
        start_time = time.time()
        vector_to_search = self._prepare_vector(vector)

        # Pre-filter keys if filter is provided
        candidate_keys = set()
        if filter:
            with self.env.begin(write=False) as txn:
                candidate_keys = self._filter_keys(filter, txn)
                self.logger.debug(f"Filter returned {len(candidate_keys)} candidate keys")

        # Query the index
        with self.index_lock:
            if not self.index_initialized:
                self._load_state()
            try:
                # Get more candidates than needed to account for filtering
                query_k = min(top_k * 20, 1000)  # Limit to 1000 for performance
                ids, distances = self.index.knnQuery(vector_to_search, k=query_k)
            except Exception as e:
                self.logger.error(f"Search failed: {e}")
                return []

        # Collect candidate keys from index results
        candidate_indices = set(ids)
        candidate_index_to_key = {}
        with self.env.begin(write=False) as txn:
            # Get keys for all candidate indices
            for idx in candidate_indices:
                key = txn.get(f"index:{idx}".encode('utf-8'), db=self.db_index_mapping)
                if key:
                    candidate_index_to_key[idx] = key.decode('utf-8')

            # Now collect data for all valid candidates
            results = []
            candidate_count = 0
            for idx, dist in zip(ids, distances):
                key_str = candidate_index_to_key.get(idx)
                if key_str is None:
                    continue

                # Check filter if provided
                if candidate_keys is not None and key_str not in candidate_keys:
                    continue

                try:
                    data = self._get_data(key_str, txn, fields=['vector', 'value', 'metadata'])

                    # Determine score based on distance metric
                    if self.space in ['cosinesimil']:
                        score = float(1 - dist)
                    elif self.space in ['l2']:
                        score = float(-dist)  # Using negative distance as similarity score
                    elif self.space in ['ip']:
                        score = float(dist)  # Inner product is already a similarity score
                    else:
                        score = float(1 - dist)  # Default behavior

                    results.append({
                        'key': key_str,
                        'value': data['value'],
                        'metadata': data['metadata'],
                        'score': score
                    })
                    candidate_count += 1

                    if len(results) >= top_k:
                        break
                except Exception as e:
                    self.logger.error(f"Error processing candidate {key_str}: {e}")
                    continue

            # Sort results by score in descending order (assuming higher score is better)
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:top_k]  # Ensure we don't return more than asked for

        self.logger.info(f"Search operation completed in {time.time() - start_time:.2f} seconds with {len(results)} results")
        return results

    def batch_put(self, list_of_entries: List[Dict[str, Any]]) -> List[str]:
        """Add multiple entries to the store in a batch operation."""
        start_time = time.time()

        # Prepare all vectors and entries for batch processing
        vectors_to_add = []
        entries_with_indices = []

        with self.env.begin(write=True, buffers=True) as txn:
            self._resize_if_needed(txn)

            # Initialize vector dimension if not set
            if self.vector_dim is None and list_of_entries:
                # Get the first entry's vector to determine dimension
                first_vector = list_of_entries[0]['vector']
                self.vector_dim = first_vector.shape[0] if self.vector_type == 'dense' else first_vector.shape[1]
                txn.put(b'vector_dim', msgpack.packb(self.vector_dim, use_bin_type=True), db=self.db_system_metadata)
                self.logger.debug(f"Set vector dimension to {self.vector_dim}")

            # Process all entries in a single transaction
            keys = []
            for entry in list_of_entries:
                if 'value' not in entry:
                    raise ValueError("Each entry must have a 'value' key")

                try:
                    msgpack.packb(entry['value'], use_bin_type=True, default=self._default)
                except Exception as e:
                    raise ValueError(f"Value must be MessagePack-serializable: {e}")

                vector_to_add = self._prepare_vector(entry['vector'])
                key = entry.get('key') or str(uuid.uuid4())

                # Check if key already exists
                if txn.get(f"key:{key}".encode('utf-8'), db=self.db_index_mapping):
                    raise ValueError(f"Key '{key}' already exists")

                # Get next index
                idx_data = txn.get(b'next_index', db=self.db_system_metadata, default=msgpack.packb(0, use_bin_type=True))
                idx = msgpack.unpackb(idx_data, raw=False)
                txn.put(b'next_index', msgpack.packb(idx + 1, use_bin_type=True), db=self.db_system_metadata)

                # Prepare the data structure to store
                data = {
                    'vector': entry['vector'],
                    'value': entry['value'],
                    'metadata': entry.get('metadata', {})
                }
                serialized_data = msgpack.packb(data, use_bin_type=True, default=self._default)

                # Store the main data and mappings
                txn.put(key.encode('utf-8'), serialized_data, db=self.db_data)
                txn.put(f"key:{key}".encode('utf-8'), msgpack.packb(idx, use_bin_type=True), db=self.db_index_mapping)
                txn.put(f"index:{idx}".encode('utf-8'), key.encode('utf-8'), db=self.db_index_mapping)

                # Store for later processing in batch
                vectors_to_add.append((idx, vector_to_add))
                entries_with_indices.append((key, idx, entry.get('metadata', {})))

                keys.append(key)

            # Update total points count
            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_system_metadata), raw=False)
            txn.put(b'total_points', msgpack.packb(total_points + len(list_of_entries), use_bin_type=True), db=self.db_system_metadata)

        # Update metadata indexes in a separate transaction if needed
        if any(metadata for _, _, metadata in entries_with_indices if metadata):
            with self.env.begin(write=True) as txn:
                for key, _, metadata in entries_with_indices:
                    if metadata:
                        self._update_metadata_indexes(key, metadata, add=True, txn=txn)

        # Update the index with all vectors in a batch
        if vectors_to_add:
            with self.index_lock:
                if not self.index_initialized:
                    self._load_state()  # Ensure index is initialized
                indices, vectors = zip(*vectors_to_add)
                self.index.addDataPointBatch(vectors, indices)
                if not self.index_initialized:
                    self.index.createIndex(self.hnsw_params, print_progress=True)
                    self.index_initialized = True

        # Update modification count for rebuild threshold
        self.modifications_since_rebuild += len(list_of_entries)

        # Check if a rebuild is needed
        with self.env.begin(write=False) as txn:
            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_system_metadata), raw=False)
            if total_points > 0 and self.modifications_since_rebuild > self.rebuild_threshold * total_points:
                with self.env.begin(write=True) as txn:
                    self._rebuild_index(txn)
                    self._save_state(txn)

        self.logger.info(f"Batch put operation completed in {time.time() - start_time:.2f} seconds")
        return keys

    def batch_search(self, list_of_vectors: List[Union[np.ndarray, csr_matrix]], top_k: int = 5,
                     filter: Optional[Dict[str, Any]] = None) -> List[List[Dict[str, Any]]]:
        """Perform a batch search for the nearest neighbors of multiple vectors."""
        start_time = time.time()
        list_of_vectors_to_search = [self._prepare_vector(v) for v in list_of_vectors]

        # Pre-filter keys if filter is provided
        candidate_keys = set()
        if filter:
            with self.env.begin(write=False) as txn:
                candidate_keys = self._filter_keys(filter, txn)
                self.logger.debug(f"Filter returned {len(candidate_keys)} candidate keys")

        # Batch query the index
        with self.index_lock:
            if not self.index_initialized:
                self._load_state()
            try:
                # Limit the number of candidates to retrieve
                query_k = min(top_k * 20, 1000)  # Limit to 1000 for performance
                ids_dists_list = self.index.knnQueryBatch(list_of_vectors_to_search, k=query_k, num_threads=cpu_count())
            except Exception as e:
                self.logger.error(f"Batch search failed: {e}")
                return [[] for _ in list_of_vectors]

        # Collect all unique candidate indices from all queries
        all_candidate_indices = set()
        for ids, _ in ids_dists_list:
            all_candidate_indices.update(ids)

        # Retrieve keys and data for all candidate indices in a single transaction
        key_to_data = {}
        with self.env.begin(write=False) as txn:
            # Create a mapping from index to key for all candidate indices
            candidate_index_to_key = {}
            for idx in all_candidate_indices:
                key = txn.get(f"index:{idx}".encode('utf-8'), db=self.db_index_mapping)
                if key:
                    candidate_index_to_key[idx] = key.decode('utf-8')

            # For each candidate key, retrieve its data if it passes the filter
            for idx, key_str in candidate_index_to_key.items():
                if candidate_keys is None or key_str in candidate_keys:
                    try:
                        data = self._get_data(key_str, txn, fields=['vector', 'value', 'metadata'])
                        key_to_data[key_str] = {
                            'vector': data['vector'],
                            'value': data['value'],
                            'metadata': data['metadata'],
                            'index': idx
                        }
                    except Exception as e:
                        self.logger.error(f"Error retrieving data for key {key_str}: {e}")

        # Process results for each query
        results = []
        for query_idx, (ids, distances) in enumerate(ids_dists_list):
            query_results = []
            for idx, dist in zip(ids, distances):
                key_str = candidate_index_to_key.get(idx)
                if key_str is None:
                    continue

                # Check if we have data for this key (should always be true)
                if key_str in key_to_data:
                    data = key_to_data[key_str]

                    # Determine score based on distance metric
                    if self.space in ['cosinesimil']:
                        score = float(1 - dist)
                    elif self.space in ['l2']:
                        score = float(-dist)  # Using negative distance as similarity score
                    elif self.space in ['ip']:
                        score = float(dist)  # Inner product is already a similarity score
                    else:
                        score = float(1 - dist)  # Default behavior

                    query_results.append({
                        'key': key_str,
                        'value': data['value'],
                        'metadata': data['metadata'],
                        'score': score
                    })
                    if len(query_results) >= top_k:
                        break

            # Sort results by score and limit to top_k
            query_results.sort(key=lambda x: x['score'], reverse=True)
            results.append(query_results[:top_k])

        self.logger.info(f"Batch search operation completed in {time.time() - start_time:.2f} seconds")
        return results

    def compact_database(self):
        """Compact the database by physically removing deleted documents and reclaiming space."""
        start_time = time.time()
        self.logger.info("Starting database compaction")

        # Create a temporary environment for the compacted database
        temp_dir = os.path.join(self.db_path, 'temp_compact')
        os.makedirs(temp_dir, exist_ok=True)
        temp_env = lmdb.open(temp_dir, map_size=self.env.info()['map_size'], max_dbs=4, writemap=True)

        # Perform compaction
        with self.env.begin(write=False) as read_txn, temp_env.begin(write=True) as write_txn:
            # Create databases in the temporary environment
            temp_db_data = temp_env.open_db(b'data', txn=write_txn)
            temp_db_index_mapping = temp_env.open_db(b'index_mapping', txn=write_txn)
            temp_db_metadata_index = temp_env.open_db(b'metadata_index', txn=write_txn)
            temp_db_system_metadata = temp_env.open_db(b'system_metadata', txn=write_txn)

            # Copy system metadata (we'll update total_points and next_index later)
            cursor = read_txn.cursor(db=self.db_system_metadata)
            for key, value in cursor:
                # Skip total_points and next_index as we'll update them
                if key not in [b'total_points', b'next_index']:
                    write_txn.put(key, value, db=temp_db_system_metadata)

            # Collect all non-deleted documents and prepare new indices
            deleted_keys = set()
            cursor = read_txn.cursor(db=self.db_data)
            cursor.set_range(b'deleted:')
            for key, _ in cursor:
                if key.startswith(b'deleted:'):
                    deleted_keys.add(key[8:].decode('utf-8'))

            # Process all non-deleted data
            new_index = 0
            key_to_new_index = {}
            index_to_key = {}

            # First pass: copy non-deleted data and assign new indices
            cursor = read_txn.cursor(db=self.db_data)
            for key, value in cursor:
                doc_key = key.decode('utf-8')
                if not key.startswith(b'deleted:') and doc_key not in deleted_keys:
                    # Get the old index (not strictly needed for compaction but useful for debugging)
                    old_idx_data = read_txn.get(f"key:{doc_key}".encode('utf-8'), db=self.db_index_mapping)
                    if old_idx_data:
                        # Assign a new sequential index
                        key_to_new_index[doc_key] = new_index
                        index_to_key[new_index] = doc_key
                        new_index += 1

                    # Copy the data to the new database
                    write_txn.put(doc_key.encode('utf-8'), value, db=temp_db_data)

            # Update system metadata with new count and next index
            new_total = len(key_to_new_index)
            write_txn.put(b'total_points', msgpack.packb(new_total, use_bin_type=True), db=temp_db_system_metadata)
            write_txn.put(b'next_index', msgpack.packb(new_total, use_bin_type=True), db=temp_db_system_metadata)

            # Second pass: rebuild index mappings and metadata indexes
            for doc_key, new_idx in key_to_new_index.items():
                # Get original data to extract metadata
                data_value = read_txn.get(doc_key.encode('utf-8'), db=self.db_data)
                if data_value:
                    try:
                        data = msgpack.unpackb(data_value, raw=False, ext_hook=self._ext_hook, use_list=False)
                        metadata = data.get('metadata', {})

                        # Store new index mappings
                        write_txn.put(f"key:{doc_key}".encode('utf-8'), msgpack.packb(new_idx, use_bin_type=True), db=temp_db_index_mapping)
                        write_txn.put(f"index:{new_idx}".encode('utf-8'), doc_key.encode('utf-8'), db=temp_db_index_mapping)

                        # Rebuild metadata indexes
                        if metadata:
                            for meta_key, meta_value in metadata.items():
                                if isinstance(meta_value, list):
                                    for val in meta_value:
                                        composite_key = f"meta:{meta_key}:{val}:{doc_key}".encode('utf-8')
                                        write_txn.put(composite_key, b'', db=temp_db_metadata_index)
                                else:
                                    composite_key = f"meta:{meta_key}:{meta_value}:{doc_key}".encode('utf-8')
                                    write_txn.put(composite_key, b'', db=temp_db_metadata_index)

                                    if isinstance(meta_value, (int, float)):
                                        padded_value = f"{float(meta_value):020.10f}"
                                        numeric_key = f"meta:{meta_key}:{padded_value}:{doc_key}".encode('utf-8')
                                        write_txn.put(numeric_key, b'', db=temp_db_metadata_index)
                    except Exception as e:
                        self.logger.error(f"Error processing metadata for key {doc_key}: {e}")
                        continue

        # Sync and close the temporary environment
        temp_env.sync()
        temp_env.close()

        # Close the original environment
        self.env.close()

        # Replace the original directory with the compacted one
        # Note: In production, you'd want to ensure atomicity of this operation
        # Here we use a simple approach for demonstration
        import shutil

        # Backup original database (optional)
        backup_path = os.path.join(self.db_path, 'backup_' + time.strftime("%Y%m%d_%H%M%S"))
        if os.path.exists(self.db_path):
            try:
                shutil.move(self.db_path, backup_path)
            except Exception as e:
                self.logger.error(f"Failed to create backup: {e}")

        # Move temp directory to original location
        try:
            shutil.move(temp_dir, self.db_path)
        except Exception as e:
            self.logger.error(f"Failed to replace database with compacted version: {e}")
            # Try to restore from backup if it exists
            if os.path.exists(backup_path):
                shutil.move(backup_path, self.db_path)
            raise RuntimeError(f"Compaction failed: {e}")

        # Reopen the environment with the compacted database
        self.env = lmdb.open(self.db_path, map_size=self.max_map_size, max_dbs=4, writemap=True)
        with self.env.begin(write=True) as txn:
            self.db_data = self.env.open_db(b'data', txn=txn)
            self.db_index_mapping = self.env.open_db(b'index_mapping', txn=txn)
            self.db_metadata_index = self.env.open_db(b'metadata_index', txn=txn)
            self.db_system_metadata = self.env.open_db(b'system_metadata', txn=txn)

        # Rebuild the index from the compacted data
        with self.env.begin(write=True) as txn:
            self._rebuild_index(txn)
            self._save_state(txn)

        self.logger.info(f"Database compaction completed in {time.time() - start_time:.2f} seconds")

    def close(self):
        """Close the vector store resources."""
        # Save state before closing
        with self.env.begin(write=True, buffers=True) as txn:
            self._save_state(txn)

        self.env.close()
        self.thread_pool.shutdown(wait=True)
        self.logger.info("Closed VectorStore resources")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
