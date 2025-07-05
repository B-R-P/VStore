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
import heapq

class VStore:
    """A vector database using NMSLIB for ANN search, LMDB for storage, and MessagePack for serialization.

    Supports dense (np.ndarray) and sparse (csr_matrix) vectors with metadata filtering and persistence.
    Optimized for read-heavy use cases with efficient query handling and minimal write overhead.

    Args:
        db_path (str): Path to the LMDB database directory.
        vector_type (str): 'dense' or 'sparse' vectors. Default: 'dense'.
        space (str): Distance metric for NMSLIB (e.g., 'cosinesimil', 'l2'). Default: 'cosinesimil'.
        map_size (int): Initial LMDB map size in bytes. Default: 1e9.
        hnsw_params (Dict[str, Any]): HNSW parameters for NMSLIB. Default: {'M': 32, 'efConstruction': 200, 'post': 2}.
        query_params (Dict[str, Any]): Query-time parameters for NMSLIB. Default: {'efSearch': 200}.
        rebuild_threshold (float): Fraction of modified points triggering index rebuild. Default: 0.5.
        max_workers (int): Number of threads for parallel operations. Default: cpu_count().
        max_map_size (int): Maximum LMDB map size in bytes. Default: 2**40 (1TB).
        max_readers (int): Maximum concurrent LMDB readers. Default: 1000.
        indexed_metadata_fields (Optional[List[str]]): Fields to index for faster filtering.
    """
    def __init__(self, db_path: str, vector_type: str = 'dense', space: str = 'cosinesimil',
                 map_size: int = int(1e9), hnsw_params: Dict[str, Any] = None,
                 query_params: Dict[str, Any] = None, rebuild_threshold: float = 0.5,
                 max_workers: int = cpu_count(), max_map_size: int = 2**40,
                 max_readers: int = 1000, indexed_metadata_fields: Optional[List[str]] = None):
        self.db_path = db_path
        self.vector_type = vector_type
        self.space = space if vector_type == 'dense' else f"{space}_sparse"
        self.env = lmdb.open(db_path, map_size=map_size, max_dbs=7, writemap=True, max_readers=max_readers)
        self.hnsw_params = hnsw_params or {'M': 32, 'efConstruction': 200, 'post': 2}
        self.query_params = query_params or {'efSearch': 200}
        self.rebuild_threshold = rebuild_threshold
        self.max_map_size = max_map_size
        self.max_readers = max_readers
        self.indexed_metadata_fields = indexed_metadata_fields
        self.index = nmslib.init(method='hnsw', space=self.space,
                                data_type=nmslib.DataType.DENSE_VECTOR if vector_type == 'dense' else nmslib.DataType.SPARSE_VECTOR)
        self.index_path = os.path.join(db_path, "index.nms")
        self.modifications_since_rebuild = 0
        self.index_initialized = False
        self.vector_dim = None
        self.logger = logging.getLogger(__name__)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.index_lock = threading.Lock()

        with self.env.begin(write=True) as txn:
            self.db_data = self.env.open_db(b'data', txn=txn)
            self.db_key_to_index = self.env.open_db(b'key_to_index', txn=txn)
            self.db_index_to_key = self.env.open_db(b'index_to_key', txn=txn)
            self.db_metadata = self.env.open_db(b'metadata', txn=txn)
            self.db_numeric_metadata = self.env.open_db(b'numeric_metadata', txn=txn)
            self.db_index_counter = self.env.open_db(b'index_counter', txn=txn)
            self.db_deleted_ids = self.env.open_db(b'deleted_ids', txn=txn)

            if txn.get(b'total_points', db=self.db_metadata) is None:
                txn.put(b'total_points', msgpack.packb(0, use_bin_type=True), db=self.db_metadata)
            if txn.get(b'vector_dim', db=self.db_metadata) is None:
                txn.put(b'vector_dim', msgpack.packb(None, use_bin_type=True), db=self.db_metadata)

        self._load_state()

    def _default(self, obj):
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
        try:
            if os.path.exists(self.index_path):
                with self.index_lock:
                    self.index.loadIndex(self.index_path, load_data=True)
                    self.index.setQueryTimeParams(self.query_params)
                    self.index_initialized = True
                self.logger.info(f"Loaded NMSLIB index from {self.index_path}")
            else:
                self.logger.debug("No existing index found, initializing new index")
            with self.env.begin(buffers=True) as txn:
                vector_dim = msgpack.unpackb(txn.get(b'vector_dim', db=self.db_metadata), raw=False)
                if vector_dim is None:
                    with txn.cursor(db=self.db_data) as cursor:
                        for _, value in cursor:
                            data = msgpack.unpackb(value, raw=False, ext_hook=self._ext_hook, use_list=False)
                            vector = data['vector']
                            self.vector_dim = vector.shape[0] if self.vector_type == 'dense' else vector.shape[1]
                            with self.env.begin(write=True) as txn2:
                                txn2.put(b'vector_dim', msgpack.packb(self.vector_dim, use_bin_type=True), db=self.db_metadata)
                            break
                else:
                    self.vector_dim = vector_dim
        except lmdb.Error as e:
            self.logger.error(f"LMDB error during load: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}. Rebuilding index.")
            with self.env.begin(write=True, buffers=True) as txn:
                self._rebuild_index(txn)
                self._save_state(txn)

    def _rebuild_index(self, txn):
        start_time = time.time()
        self.logger.info("Rebuilding NMSLIB index")

        # Safety check - if we have no valid vectors, just return
        try:
            with txn.cursor(db=self.db_key_to_index) as cursor:
                if not cursor.first():
                    self.logger.warning("No valid vectors found for index rebuilding")
                    self.index_initialized = False
                    self.modifications_since_rebuild = 0
                    return
        except Exception as e:
            self.logger.error(f"Error checking for valid vectors: {e}")
            self.index_initialized = False
            self.modifications_since_rebuild = 0
            return

        with self.index_lock:
            try:
                # Initialize a new index with safety checks
                try:
                    self.index = nmslib.init(
                        method='hnsw',
                        space=self.space,
                        data_type=nmslib.DataType.DENSE_VECTOR if self.vector_type == 'dense'
                                  else nmslib.DataType.SPARSE_VECTOR
                    )
                except Exception as e:
                    self.logger.error(f"Failed to initialize NMSLIB index: {e}")
                    self.index_initialized = False
                    self.modifications_since_rebuild = 0
                    return

                vectors = []
                indices = []
                deleted_keys = set()

                # Safely collect deleted keys
                try:
                    with txn.cursor(db=self.db_deleted_ids) as cursor:
                        for k, _ in cursor:
                            try:
                                key_str = k.decode('utf-8') if isinstance(k, bytes) else k.tobytes().decode('utf-8')
                                deleted_keys.add(key_str)
                            except Exception as e:
                                self.logger.warning(f"Could not decode deleted key: {e}")
                                continue
                    self.logger.debug(f"Found {len(deleted_keys)} deleted keys")
                except Exception as e:
                    self.logger.warning(f"Could not retrieve deleted keys: {e}")

                # Process valid vectors
                try:
                    with txn.cursor(db=self.db_key_to_index) as cursor:
                        for key, idx_data in cursor:
                            try:
                                key_str = key.decode('utf-8') if isinstance(key, bytes) else key.tobytes().decode('utf-8')

                                # Skip deleted keys
                                if key_str in deleted_keys:
                                    continue

                                # Get and validate index data
                                try:
                                    idx = msgpack.unpackb(idx_data, raw=False)
                                except Exception as e:
                                    self.logger.warning(f"Could not unpack index data for key {key_str}: {e}")
                                    continue

                                # Get and validate the vector data
                                try:
                                    data = self._get_data(key_str, txn)
                                    if not data or 'vector' not in data:
                                        self.logger.warning(f"No vector data found for key {key_str}")
                                        continue

                                    # Prepare the vector with additional validation
                                    vector = self._prepare_vector(data['vector'])

                                    # Additional validation for the vector
                                    if self.vector_type == 'dense':
                                        if not isinstance(vector, np.ndarray):
                                            raise ValueError("Expected numpy array for dense vector")
                                        if vector.size == 0:
                                            raise ValueError("Empty dense vector")
                                    elif self.vector_type == 'sparse':
                                        if not isinstance(vector, list):
                                            raise ValueError("Expected list for sparse vector")
                                        if len(vector) == 0:
                                            raise ValueError("Empty sparse vector")

                                    vectors.append(vector)
                                    indices.append(idx)
                                except Exception as e:
                                    self.logger.warning(f"Error processing vector for key {key_str}: {e}")
                                    continue

                            except Exception as e:
                                self.logger.warning(f"Error processing key {key}: {e}")
                                continue

                except Exception as e:
                    self.logger.error(f"Error processing vectors: {e}")
                    self.index_initialized = False
                    self.modifications_since_rebuild = 0
                    return

                # Only proceed if we have valid vectors
                if len(vectors) == 0:
                    self.logger.warning("No valid vectors remaining after filtering")
                    self.index_initialized = False
                    self.modifications_since_rebuild = 0
                    return

                # Add vectors to the index with additional safety checks
                try:
                    self.logger.debug(f"Adding {len(vectors)} vectors to the index")

                    # Validate vector dimensions
                    if self.vector_type == 'dense':
                        first_vector_length = len(vectors[0])
                        for vec in vectors:
                            if len(vec) != first_vector_length:
                                raise ValueError(f"Inconsistent vector dimensions. Expected {first_vector_length}, got {len(vec)}")
                    elif self.vector_type == 'sparse':
                        # For sparse vectors, we can't easily check dimensions
                        pass

                    # Add data in chunks to avoid memory issues
                    chunk_size = 1000
                    for i in range(0, len(vectors), chunk_size):
                        chunk_vectors = vectors[i:i+chunk_size]
                        chunk_indices = indices[i:i+chunk_size]
                        try:
                            self.index.addDataPointBatch(chunk_vectors, chunk_indices)
                        except Exception as e:
                            self.logger.error(f"Failed to add chunk starting at index {i}: {e}")
                            raise

                    # Create index with safety checks
                    try:
                        # Use default params if our params somehow got corrupted
                        params = self.hnsw_params or {}
                        if not isinstance(params, dict):
                            params = {'M': 32, 'efConstruction': 200, 'post': 2}

                        self.index.createIndex(params, print_progress=True)
                        self.index_initialized = True
                    except Exception as e:
                        self.logger.error(f"Failed to create index: {e}")
                        self.index_initialized = False
                        raise

                    try:
                        # Use default params if our query params got corrupted
                        query_params = self.query_params or {}
                        if not isinstance(query_params, dict):
                            query_params = {'efSearch': 200}

                        self.index.setQueryTimeParams(query_params)
                    except Exception as e:
                        self.logger.error(f"Failed to set query parameters: {e}")
                        # Don't fail completely if we can't set query params

                except Exception as e:
                    self.logger.error(f"Error adding data points to index: {e}")
                    self.index_initialized = False
                    raise

            except Exception as e:
                self.logger.error(f"Error during index rebuild: {e}")
                self.index_initialized = False
                raise

        # Reset modifications counter
        self.modifications_since_rebuild = 0
        self.logger.info(f"Index rebuild completed in {time.time() - start_time:.2f} seconds")



    def _save_state(self, txn):
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

    def _get_data(self, key: str, txn) -> Dict[str, Any]:
        if key is None:
            raise ValueError("Key cannot be None")
        value = txn.get(key.encode('utf-8'), db=self.db_data)
        if value is None:
            raise KeyError(f"Key '{key}' not found")
        return msgpack.unpackb(value, raw=False, ext_hook=self._ext_hook, use_list=False)

    def _prepare_vector(self, vector: Union[np.ndarray, csr_matrix]) -> Union[np.ndarray, List[Tuple[int, float]]]:
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
            if self.space == 'cosinesimil':
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
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
            if self.space == 'cosinesimil_sparse':
                norm = np.sqrt(np.sum(vector.data ** 2))
                if norm > 0:
                    vector = csr_matrix((vector.data / norm, vector.indices, vector.indptr), shape=vector.shape)
            # Convert to list of (index, value) pairs for NMSLIB
            return list(zip(vector.indices, vector.data))
        raise ValueError(f"Unsupported vector_type: {self.vector_type}")

    def _update_metadata(self, key: str, metadata: Dict[str, Any], add: bool = True, txn=None):
        for meta_key, meta_value in metadata.items():
            if self.indexed_metadata_fields is not None and meta_key not in self.indexed_metadata_fields:
                continue
            if isinstance(meta_value, list):
                for val in meta_value:
                    composite_key = f"{meta_key}:{val}:{key}".encode('utf-8')
                    if add:
                        txn.put(composite_key, b'', db=self.db_metadata)
                    else:
                        txn.delete(composite_key, db=self.db_metadata)
            else:
                composite_key = f"{meta_key}:{meta_value}:{key}".encode('utf-8')
                if add:
                    txn.put(composite_key, b'', db=self.db_metadata)
                else:
                    txn.delete(composite_key, db=self.db_metadata)
                if isinstance(meta_value, (int, float)):
                    padded_value = f"{float(meta_value):020.10f}"
                    numeric_key = f"{meta_key}:{padded_value}:{key}".encode('utf-8')
                    if add:
                        txn.put(numeric_key, b'', db=self.db_numeric_metadata)
                    else:
                        txn.delete(numeric_key, db=self.db_numeric_metadata)

    def _validate_filter(self, filter: Dict[str, Any]) -> None:
        if isinstance(filter, dict) and 'op' in filter:
            if filter['op'] not in ('AND', 'OR'):
                raise ValueError(f"Invalid operator: {filter['op']}")
            for cond in filter['conditions']:
                self._validate_filter(cond)
        else:
            for key, value in filter.items():
                if isinstance(value, list) and len(value) != 2:
                    raise ValueError(f"Range filter for '{key}' must be a list of [min, max]")

    def _estimate_filter_selectivity(self, filter: Optional[Dict[str, Any]], total_points: int) -> float:
        if not filter:
            return 1.0
        if isinstance(filter, dict) and 'op' in filter:
            selectivities = [self._estimate_filter_selectivity(cond, total_points) for cond in filter['conditions']]
            return min(selectivities) if filter['op'] == 'AND' else 1 - (1 - sum(selectivities) / len(selectivities))**len(selectivities)
        with self.env.begin(write=False) as txn:
            sample_size = min(total_points, 1000)
            matches = len(self._filter_keys(filter, txn))
            return matches / sample_size if sample_size > 0 else 0.1

    def _filter_keys(self, filter: Optional[Dict[str, Any]], txn) -> set:
        if not filter:
            deleted_keys = set(k.decode('utf-8') if isinstance(k, bytes) else k.tobytes().decode('utf-8') for k, _ in txn.cursor(db=self.db_deleted_ids))
            return set(k.decode('utf-8') if isinstance(k, bytes) else k.tobytes().decode('utf-8') for k, _ in txn.cursor(db=self.db_key_to_index)) - deleted_keys

        self._validate_filter(filter)
        if isinstance(filter, dict) and 'op' in filter:
            op = filter['op']
            conditions = filter['conditions']
            result_sets = []
            for cond in conditions:
                keys = self._filter_keys(cond, txn)
                self.logger.debug(f"Condition {cond} matched {len(keys)} keys")
                result_sets.append(keys)
            if not result_sets:
                self.logger.debug("No conditions provided, returning empty set")
                return set()
            return set.intersection(*result_sets) if op == 'AND' else set.union(*result_sets)

        matched_keys = set()
        for key, condition in filter.items():
            if isinstance(condition, list) and isinstance(condition[0], (int, float)):
                cursor = txn.cursor(db=self.db_numeric_metadata)
                start_key = f"{key}:{float(condition[0]):020.10f}:".encode('utf-8')
                upper_bound = condition[1] + 1e-10
                end_key = f"{key}:{float(upper_bound):020.10f}:".encode('utf-8')
                cursor.set_range(start_key)
                while cursor.key():
                    meta_key = cursor.key() if isinstance(cursor.key(), bytes) else cursor.key().tobytes()
                    if meta_key > end_key:
                        break
                    try:
                        _, _, doc_key = meta_key.decode('utf-8').split(':', 2)
                        matched_keys.add(doc_key)
                    except (ValueError, IndexError):
                        pass
                    cursor.next()
                self.logger.debug(f"Numeric filter {key}:{condition} matched {len(matched_keys)} keys")
            else:
                values = condition if isinstance(condition, list) else [condition]
                for value in values:
                    prefix = f"{key}:{value}:".encode('utf-8')
                    cursor = txn.cursor(db=self.db_metadata)
                    cursor.set_range(prefix)
                    value_keys = set()
                    while cursor.key():
                        meta_key = cursor.key() if isinstance(cursor.key(), bytes) else cursor.key().tobytes()
                        if not meta_key.startswith(prefix):
                            break
                        try:
                            doc_key = meta_key.decode('utf-8').split(':', 2)[2]
                            value_keys.add(doc_key)
                        except (ValueError, IndexError):
                            pass
                        cursor.next()
                    matched_keys.update(value_keys)
                    self.logger.debug(f"Non-numeric filter {key}:{value} matched {len(value_keys)} keys")
        deleted_keys = set(k.decode('utf-8') if isinstance(k, bytes) else k.tobytes().decode('utf-8') for k, _ in txn.cursor(db=self.db_deleted_ids))
        self.logger.debug(f"Filtered {len(matched_keys)} keys, removed {len(deleted_keys)} deleted keys")
        return matched_keys - deleted_keys

    def _resize_if_needed(self, txn):
        info = self.env.info()
        stat = self.env.stat()
        used = (info['last_pgno'] + 1) * stat['psize']
        page_size = os.sysconf('SC_PAGESIZE') if hasattr(os, 'sysconf') else 4096  # Default to 4096
        if used > info['map_size'] * 0.6:  # Lowered threshold to 60%
            new_size = min(info['map_size'] * 4, self.max_map_size)
            # Round up to nearest multiple of page_size
            new_size = ((new_size + page_size - 1) // page_size) * page_size
            try:
                self.env.set_mapsize(new_size)
                self.logger.info(f"Resized map_size to {new_size} bytes")
            except lmdb.Error as e:
                self.logger.error(f"Failed to resize map_size: {e}. Current size: {info['map_size']}")
                raise

    def put(self, vector: Union[np.ndarray, csr_matrix], value: Any,
            metadata: Optional[Dict[str, Any]] = None, key: Optional[str] = None) -> str:
        start_time = time.time()
        try:
            msgpack.packb(value, use_bin_type=True, default=self._default)
        except Exception as e:
            raise ValueError(f"Value must be MessagePack-serializable: {e}")
        vector_to_add = self._prepare_vector(vector)
        with self.env.begin(write=True, buffers=True) as txn:
            self._resize_if_needed(txn)
            if self.vector_dim is None:
                self.vector_dim = vector.shape[0] if self.vector_type == 'dense' else vector.shape[1]
                txn.put(b'vector_dim', msgpack.packb(self.vector_dim, use_bin_type=True), db=self.db_metadata)
                self.logger.debug(f"Set vector dimension to {self.vector_dim}")

            if key is None:
                max_retries = 5
                for i in range(max_retries):
                    key = str(uuid.uuid4())
                    if not txn.get(key.encode('utf-8'), db=self.db_key_to_index):
                        break
                    if i == max_retries - 1:
                        self.logger.warning("Failed to generate unique key after retries")
                        raise ValueError(f"Failed to generate unique key after {max_retries} retries")

            if txn.get(key.encode('utf-8'), db=self.db_key_to_index):
                raise ValueError(f"Key '{key}' already exists. Use update() to modify.")

            idx_data = txn.get(b'next_index', db=self.db_index_counter, default=msgpack.packb(0, use_bin_type=True))
            idx = msgpack.unpackb(idx_data, raw=False)
            txn.put(b'next_index', msgpack.packb(idx + 1, use_bin_type=True), db=self.db_index_counter)

            with self.index_lock:
                self.index.addDataPoint(idx, vector_to_add)
                self.index_initialized = True

            data = {'vector': vector, 'value': value, 'metadata': metadata or {}}
            serialized_data = msgpack.packb(data, use_bin_type=True, default=self._default)
            txn.put(key.encode('utf-8'), serialized_data, db=self.db_data)
            txn.put(key.encode('utf-8'), msgpack.packb(idx, use_bin_type=True), db=self.db_key_to_index)
            txn.put(str(idx).encode('utf-8'), key.encode('utf-8'), db=self.db_index_to_key)

            if metadata:
                self._update_metadata(key, metadata, add=True, txn=txn)

            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_metadata), raw=False)
            txn.put(b'total_points', msgpack.packb(total_points + 1, use_bin_type=True), db=self.db_metadata)

        self.modifications_since_rebuild += 1
        with self.env.begin(write=True, buffers=True) as txn:
            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_metadata), raw=False)
            if total_points > 0 and self.modifications_since_rebuild > self.rebuild_threshold * total_points:
                self._rebuild_index(txn)
                self._save_state(txn)
        self.logger.info(f"Put operation completed in {time.time() - start_time:.2f} seconds")
        return key

    def get(self, key: str) -> Tuple[Union[np.ndarray, csr_matrix], Any, Dict[str, Any]]:
        if key is None:
            raise ValueError("Key cannot be None")
        with self.env.begin(write=False, buffers=True) as txn:
            if txn.get(key.encode('utf-8'), db=self.db_deleted_ids):
                raise KeyError(f"Key '{key}' has been deleted")
            data = self._get_data(key, txn)
            return data['vector'], data['value'], data['metadata']

    def delete(self, key: str):
        if key is None:
            raise ValueError("Key cannot be None")
        start_time = time.time()
        with self.env.begin(write=True, buffers=True) as txn:
            self._resize_if_needed(txn)
            if not txn.get(key.encode('utf-8'), db=self.db_key_to_index) or \
              txn.get(key.encode('utf-8'), db=self.db_deleted_ids):
                self.logger.info(f"Key '{key}' not found or already deleted")
                return

            data = self._get_data(key, txn)
            metadata = data['metadata']
            self._update_metadata(key, metadata, add=False, txn=txn)

            txn.delete(key.encode('utf-8'), db=self.db_data)
            idx = msgpack.unpackb(txn.get(key.encode('utf-8'), db=self.db_key_to_index), raw=False)
            txn.delete(key.encode('utf-8'), db=self.db_key_to_index)
            txn.delete(str(idx).encode('utf-8'), db=self.db_index_to_key)
            txn.put(key.encode('utf-8'), b'', db=self.db_deleted_ids)

            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_metadata), raw=False)
            if total_points > 0:
                txn.put(b'total_points', msgpack.packb(total_points - 1, use_bin_type=True), db=self.db_metadata)

            self.modifications_since_rebuild += 1  # Increment before threshold check

            if total_points > 0 and self.modifications_since_rebuild > self.rebuild_threshold * total_points:
                self._rebuild_index(txn)
                self._save_state(txn)

        self.logger.info(f"Delete operation completed in {time.time() - start_time:.2f} seconds")
  
    def update(self, key: str, vector: Optional[Union[np.ndarray, csr_matrix]] = None,
               value: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
        if key is None:
            raise ValueError("Key cannot be None")
        start_time = time.time()
        if value is not None:
            try:
                msgpack.packb(value, use_bin_type=True, default=self._default)
            except Exception as e:
                raise ValueError(f"Value must be MessagePack-serializable: {e}")
        vector_to_add = self._prepare_vector(vector) if vector is not None else None

        with self.env.begin(write=True, buffers=True) as txn:
            self._resize_if_needed(txn)
            if txn.get(key.encode('utf-8'), db=self.db_deleted_ids):
                raise KeyError(f"Key '{key}' has been deleted")
            data = self._get_data(key, txn)
            if vector is not None:
                data['vector'] = vector
                idx = msgpack.unpackb(txn.get(key.encode('utf-8'), db=self.db_key_to_index), raw=False)
                with self.index_lock:
                    self.index.addDataPoint(idx, vector_to_add)
                    self.index_initialized = True
            if value is not None:
                data['value'] = value
            if metadata is not None:
                self._update_metadata(key, data['metadata'], add=False, txn=txn)
                data['metadata'] = metadata
                self._update_metadata(key, metadata, add=True, txn=txn)

            serialized_data = msgpack.packb(data, use_bin_type=True, default=self._default)
            txn.put(key.encode('utf-8'), serialized_data, db=self.db_data)

        self.modifications_since_rebuild += 1
        with self.env.begin(write=True, buffers=True) as txn:
            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_metadata), raw=False)
            if total_points > 0 and self.modifications_since_rebuild > self.rebuild_threshold * total_points:
                self._rebuild_index(txn)
                self._save_state(txn)
        self.logger.info(f"Update operation completed in {time.time() - start_time:.2f} seconds")

    def search(self, vector: Union[np.ndarray, csr_matrix], top_k: int = 5,
               filter: Optional[Dict[str, Any]] = None, sort_descending: bool = True) -> List[Dict[str, Any]]:
        start_time = time.time()
        vector_to_search = self._prepare_vector(vector)

        if self.vector_dim is None:
            self.logger.info("No vector dimension set; returning empty results")
            return []

        candidate_keys = None
        candidate_indices = None

        with self.env.begin(write=True, buffers=True) as txn:
            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_metadata), raw=False)
            if total_points == 0 or not self.index_initialized:
                return []

            if total_points > 0 and self.modifications_since_rebuild > self.rebuild_threshold * total_points:
                self._rebuild_index(txn)
                self._save_state(txn)

            if filter:
                candidate_keys = self._filter_keys(filter, txn)
                if not candidate_keys:
                    return []
                candidate_indices = set()
                for key in candidate_keys:
                    idx_data = txn.get(key.encode('utf-8'), db=self.db_key_to_index)
                    if idx_data:
                        idx = msgpack.unpackb(idx_data, raw=False)
                        candidate_indices.add(idx)

            selectivity = self._estimate_filter_selectivity(filter, total_points)
            query_k = min(max(int(top_k / max(selectivity, 0.01)), top_k * 10), total_points, 10000)
            max_candidates = top_k * 100
            candidates = []
            seen_indices = set()
            counter = 0  # Unique counter for tiebreaking

            while query_k <= max_candidates:
                try:
                    ids, similarities = self.index.knnQuery(vector_to_search, k=query_k)
                except nmslib.NMSLibError as e:
                    self.logger.error(f"NMSLIB query failed: {e}")
                    raise ValueError(f"Search failed due to NMSLIB error: {e}")
                except Exception as e:
                    self.logger.error(f"Unexpected error in search: {e}")
                    return []

                new_results = 0
                for idx, similarity in zip(ids, similarities):
                    if idx in seen_indices:
                        continue
                    if candidate_indices is not None and idx not in candidate_indices:
                        continue

                    key = txn.get(str(idx).encode('utf-8'), db=self.db_index_to_key)
                    if key is None:
                        continue
                    key_str = key.tobytes().decode('utf-8')
                    if candidate_keys is not None and key_str not in candidate_keys:
                        continue

                    data = self._get_data(key_str, txn)
                    heapq.heappush(candidates, (-similarity if sort_descending else similarity, counter, {
                        'key': key_str,
                        'value': data['value'],
                        'metadata': data['metadata'],
                        'score': similarity
                    }))
                    seen_indices.add(idx)
                    new_results += 1
                    counter += 1
                    if len(candidates) > top_k:
                        heapq.heappop(candidates)

                if new_results == 0:
                    break
                query_k = min(int(query_k * 1.5), max_candidates)

            candidates = [item[2] for item in sorted(candidates, reverse=sort_descending)]

        self.logger.info(f"Search operation completed in {time.time() - start_time:.2f} seconds with {len(candidates)} results")
        return candidates

    def batch_put(self, list_of_entries: List[Dict[str, Any]]) -> List[str]:
        start_time = time.time()
        vectors = []
        indices = []
        keys = []
        with self.env.begin(write=True, buffers=True) as txn:
            self._resize_if_needed(txn)
            if self.vector_dim is None and list_of_entries:
                self.vector_dim = list_of_entries[0]['vector'].shape[0] if self.vector_type == 'dense' else list_of_entries[0]['vector'].shape[1]
                txn.put(b'vector_dim', msgpack.packb(self.vector_dim, use_bin_type=True), db=self.db_metadata)
                self.logger.debug(f"Set vector dimension to {self.vector_dim}")
            for entry in list_of_entries:
                if 'value' not in entry:
                    raise ValueError("Each entry must have a 'value' key")
                try:
                    msgpack.packb(entry['value'], use_bin_type=True, default=self._default)
                except Exception as e:
                    raise ValueError(f"Value must be MessagePack-serializable: {e}")
                vector_to_add = self._prepare_vector(entry['vector'])

                key = entry.get('key') or str(uuid.uuid4())
                if txn.get(key.encode('utf-8'), db=self.db_key_to_index):
                    raise ValueError(f"Key '{key}' already exists")

                idx_data = txn.get(b'next_index', db=self.db_index_counter, default=msgpack.packb(0, use_bin_type=True))
                idx = msgpack.unpackb(idx_data, raw=False)
                txn.put(b'next_index', msgpack.packb(idx + 1, use_bin_type=True), db=self.db_index_counter)

                vectors.append(vector_to_add)
                indices.append(idx)
                keys.append(key)

                data = {
                    'vector': entry['vector'],
                    'value': entry['value'],
                    'metadata': entry.get('metadata', {})
                }
                serialized_data = msgpack.packb(data, use_bin_type=True, default=self._default)
                txn.put(key.encode('utf-8'), serialized_data, db=self.db_data)
                txn.put(key.encode('utf-8'), msgpack.packb(idx, use_bin_type=True), db=self.db_key_to_index)
                txn.put(str(idx).encode('utf-8'), key.encode('utf-8'), db=self.db_index_to_key)
                if 'metadata' in entry:
                    self._update_metadata(key, entry['metadata'], add=True, txn=txn)
            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_metadata), raw=False)
            txn.put(b'total_points', msgpack.packb(total_points + len(list_of_entries), use_bin_type=True), db=self.db_metadata)

        with self.index_lock:
            if vectors:
                self.index.addDataPointBatch(vectors, indices)
                self.index.createIndex(self.hnsw_params, print_progress=True)
                self.index_initialized = True

        self.modifications_since_rebuild += len(list_of_entries)
        with self.env.begin(write=True, buffers=True) as txn:
            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_metadata), raw=False)
            if total_points > 0 and self.modifications_since_rebuild > self.rebuild_threshold * total_points:
                self._rebuild_index(txn)
                self._save_state(txn)
        self.logger.info(f"Batch put operation completed in {time.time() - start_time:.2f} seconds")
        return keys

    def batch_get(self, list_of_keys: List[str]) -> List[Tuple[Union[np.ndarray, csr_matrix], Any, Dict[str, Any]]]:
        with self.env.begin(write=False, buffers=True) as txn:
            results = []
            for key in list_of_keys:
                if key is None:
                    raise ValueError("Key cannot be None")
                if txn.get(key.encode('utf-8'), db=self.db_deleted_ids):
                    raise KeyError(f"Key '{key}' has been deleted")
                data = self._get_data(key, txn)
                results.append((data['vector'], data['value'], data['metadata']))
            return results

    def batch_search(self, list_of_vectors: List[Union[np.ndarray, csr_matrix]], top_k: int = 5,
                     filter: Optional[Dict[str, Any]] = None, sort_descending: bool = True) -> List[List[Dict[str, Any]]]:
        start_time = time.time()
        list_of_vectors_to_search = [self._prepare_vector(v) for v in list_of_vectors]

        if self.vector_dim is None:
            self.logger.info("No vector dimension set; returning empty results")
            return [[] for _ in list_of_vectors]

        with self.env.begin(write=True, buffers=True) as txn:
            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_metadata), raw=False)
            if total_points == 0 or not self.index_initialized:
                return [[] for _ in list_of_vectors]

            # Check for index rebuild
            if total_points > 0 and self.modifications_since_rebuild > self.rebuild_threshold * total_points:
                self._rebuild_index(txn)
                self._save_state(txn)

            candidate_keys = self._filter_keys(filter, txn) if filter else None
            candidate_indices = None
            if candidate_keys:
                if not candidate_keys:
                    return [[] for _ in list_of_vectors]
                candidate_indices = set()
                for key in candidate_keys:
                    idx_data = txn.get(key.encode('utf-8'), db=self.db_key_to_index)
                    if idx_data:
                        idx = msgpack.unpackb(idx_data, raw=False)
                        candidate_indices.add(idx)

            selectivity = self._estimate_filter_selectivity(filter, total_points)
            query_k = min(max(int(top_k / max(selectivity, 0.01)), top_k * 10), total_points, 10000)
            max_candidates = top_k * 100

            results_all = [[] for _ in list_of_vectors]
            seen_indices_all = [set() for _ in list_of_vectors]
            counters = [0 for _ in list_of_vectors]  # Unique counter for each query

            while query_k <= max_candidates:
                try:
                    ids_similarities = self.index.knnQueryBatch(
                        list_of_vectors_to_search, k=query_k, num_threads=cpu_count()
                    )
                except nmslib.NMSLibError as e:
                    self.logger.error(f"NMSLIB batch query failed: {e}")
                    raise ValueError(f"Batch search failed due to NMSLIB error: {e}")
                except Exception as e:
                    self.logger.error(f"Unexpected error in batch search: {e}")
                    return [[] for _ in list_of_vectors]

                any_new_results = False

                for i, (ids, similarities) in enumerate(ids_similarities):
                    results = results_all[i]
                    seen_indices = seen_indices_all[i]
                    counter = counters[i]
                    for idx, similarity in zip(ids, similarities):
                        if idx in seen_indices:
                            continue
                        if candidate_indices is not None and idx not in candidate_indices:
                            continue

                        key = txn.get(str(idx).encode('utf-8'), db=self.db_index_to_key)
                        if key is None:
                            continue
                        key_str = key.tobytes().decode('utf-8')
                        if candidate_keys is not None and key_str not in candidate_keys:
                            continue

                        data = self._get_data(key_str, txn)
                        heapq.heappush(results, (-similarity if sort_descending else similarity, counter, {
                            'key': key_str,
                            'value': data['value'],
                            'metadata': data['metadata'],
                            'score': similarity
                        }))
                        seen_indices.add(idx)
                        any_new_results = True
                        counter += 1
                        if len(results) > top_k:
                            heapq.heappop(results)
                    counters[i] = counter

                if not any_new_results:
                    break
                query_k = min(int(query_k * 1.5), max_candidates)

            # Final sort and truncate for each query
            for i, results in enumerate(results_all):
                results_all[i] = [item[2] for item in sorted(results, reverse=sort_descending)]

            self.logger.info(f"Batch search operation completed in {time.time() - start_time:.2f} seconds with {len(results_all)} queries")
            return results_all

    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        with self.env.begin(write=False, buffers=True) as txn:
            if filter:
                return len(self._filter_keys(filter, txn))
            key_to_index_stats = txn.stat(db=self.db_key_to_index)
            deleted_ids_stats = txn.stat(db=self.db_deleted_ids)
            return key_to_index_stats['entries'] - deleted_ids_stats['entries']

    def clear(self):
        with self.env.begin(write=True, buffers=True) as txn:
            txn.drop(self.db_data, delete=False)
            txn.drop(self.db_key_to_index, delete=False)
            txn.drop(self.db_index_to_key, delete=False)
            txn.drop(self.db_metadata, delete=False)
            txn.drop(self.db_numeric_metadata, delete=False)
            txn.drop(self.db_index_counter, delete=False)
            txn.drop(self.db_deleted_ids, delete=False)
            txn.put(b'total_points', msgpack.packb(0, use_bin_type=True), db=self.db_metadata)
            txn.put(b'vector_dim', msgpack.packb(None, use_bin_type=True), db=self.db_metadata)
        with self.index_lock:
            self.index = nmslib.init(method='hnsw', space=self.space,
                                     data_type=nmslib.DataType.DENSE_VECTOR if self.vector_type == 'dense' else nmslib.DataType.SPARSE_VECTOR)
            self.index_initialized = False
            self.vector_dim = None
        self.modifications_since_rebuild = 0
        with self.env.begin(write=True, buffers=True) as txn:
            self._save_state(txn)
        self.logger.info("Cleared all data from store")

    def get_by_metadata(self, filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        with self.env.begin(write=False, buffers=True) as txn:
            keys = self._filter_keys(filter, txn)
            return [{'key': key, **self._get_data(key, txn)} for key in keys]

    def compact_index(self):
        with self.env.begin(write=True, buffers=True) as txn:
            valid_keys = set(k.tobytes().decode('utf-8') for k, _ in txn.cursor(db=self.db_data))
            with txn.cursor(db=self.db_deleted_ids) as cursor:
                for key, _ in cursor:
                    if key.tobytes().decode('utf-8') not in valid_keys:
                        cursor.delete()
            self._rebuild_index(txn)
            self._save_state(txn)
        self.logger.info("Compacted index and cleaned up deleted documents")

    def validate_indices(self):
        with self.env.begin(write=False, buffers=True) as txn:
            data_stats = txn.stat(db=self.db_data)
            key_to_index_stats = txn.stat(db=self.db_key_to_index)
            index_to_key_stats = txn.stat(db=self.db_index_to_key)
            deleted_stats = txn.stat(db=self.db_deleted_ids)
            data_count = data_stats['entries']
            key_to_index_count = key_to_index_stats['entries']
            index_to_key_count = index_to_key_stats['entries']
            deleted_count = deleted_stats['entries']
            if data_count != key_to_index_count or data_count != index_to_key_count + deleted_count:
                raise ValueError(f"Inconsistent counts: data={data_count}, key_to_index={key_to_index_count}, index_to_key={index_to_key_count}, deleted={deleted_count}")
            self.logger.info("Indices are consistent based on counts")

    def close(self):
        with self.env.begin(write=True, buffers=True) as txn:
            self._save_state(txn)
        self.env.close()
        self.thread_pool.shutdown(wait=True)
        self.logger.info("Closed VectorStore resources")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
