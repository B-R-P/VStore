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
                 query_params: Dict[str, Any] = None, rebuild_threshold: float = 0.1,
                 max_workers: int = cpu_count(), max_map_size: int = 2**40):
        self.db_path = db_path
        self.vector_type = vector_type
        self.space = space if vector_type == 'dense' else f"{space}_sparse"
        self.env = lmdb.open(db_path, map_size=map_size, max_dbs=7, writemap=True)
        self.hnsw_params = hnsw_params or {'M': 16, 'efConstruction': 200, 'post': 2}
        self.query_params = query_params or {'efSearch': 100}
        self.rebuild_threshold = rebuild_threshold
        self.max_map_size = max_map_size
        self.index = nmslib.init(method='hnsw', space=self.space,
                                 data_type=nmslib.DataType.DENSE_VECTOR if vector_type == 'dense' else nmslib.DataType.SPARSE_VECTOR)
        self.index_path = os.path.join(db_path, "index.nms")
        self.modifications_since_rebuild = 0
        self.index_initialized = False
        self.vector_dim = None  # To store expected vector dimensionality
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
            # Load vector dimensionality
            with self.env.begin(buffers=True) as txn:
                vector_dim = msgpack.unpackb(txn.get(b'vector_dim', db=self.db_metadata), raw=False)
                if vector_dim is None:
                    # Check existing vectors to set dimension
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
        with self.index_lock:
            self.index = nmslib.init(method='hnsw', space=self.space,
                                     data_type=nmslib.DataType.DENSE_VECTOR if self.vector_type == 'dense' else nmslib.DataType.SPARSE_VECTOR)
            vectors = []
            indices = []
            deleted_keys = set(k.decode('utf-8') if isinstance(k, bytes) else k.tobytes().decode('utf-8') for k, _ in txn.cursor(db=self.db_deleted_ids))
            with txn.cursor(db=self.db_key_to_index) as cursor:
                for key, idx_data in cursor:
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key.tobytes().decode('utf-8')
                    if key_str in deleted_keys:
                        continue
                    idx = msgpack.unpackb(idx_data, raw=False)
                    data = self._get_data(key_str, txn)
                    vectors.append(self._prepare_vector(data['vector']))
                    indices.append(idx)
            if vectors:
                self.index.addDataPointBatch(vectors, indices)
                self.index.createIndex(self.hnsw_params, print_progress=True)
                self.index_initialized = True
            else:
                self.index_initialized = False
            self.index.setQueryTimeParams(self.query_params)
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

    def _prepare_vector(self, vector: Union[np.ndarray, csr_matrix]) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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

    def _update_metadata(self, key: str, metadata: Dict[str, Any], add: bool = True, txn=None):
        for meta_key, meta_value in metadata.items():
            if isinstance(meta_value, list):
                for val in meta_value:
                    composite_key = f"{meta_key}:{val}:{key}".encode('utf-8')
                    if add:
                        txn.put(composite_key, b'', dk=self.db_metadata)
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
            self.logger.debug(f"{'Added' if add else 'Removed'} metadata {meta_key}:{meta_value} for key {key}")

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
                # Add a small epsilon to the upper bound to include it in the range
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
        if used > info['map_size'] * 0.8:
            new_size = min(info['map_size'] * 2, self.max_map_size)
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
        # Set vector dimension if not already set
        with self.env.begin(write=True, buffers=True) as txn:
            if self.vector_dim is None:
                self.vector_dim = vector.shape[0] if self.vector_type == 'dense' else vector.shape[1]
                txn.put(b'vector_dim', msgpack.packb(self.vector_dim, use_bin_type=True), db=self.db_metadata)
                self.logger.debug(f"Set vector dimension to {self.vector_dim}")

        if key is None:
            max_retries = 5
            for i in range(max_retries):
                key = str(uuid.uuid4())
                with self.env.begin() as txn:
                    if not txn.get(key.encode('utf-8'), db=self.db_key_to_index):
                        break
                if i == max_retries - 1:
                    self.logger.warning("Failed to generate unique key after retries")
                    raise ValueError(f"Failed to generate unique key after {max_retries} retries")

        with self.env.begin(write=True, buffers=True) as txn:
            self._resize_if_needed(txn)
            if txn.get(key.encode('utf-8'), db=self.db_key_to_index):
                raise ValueError(f"Key '{key}' already exists. Use update() to modify.")

            idx_data = txn.get(b'next_index', db=self.db_index_counter, default=msgpack.packb(0, use_bin_type=True))
            idx = msgpack.unpackb(idx_data, raw=False)
            txn.put(b'next_index', msgpack.packb(idx + 1, use_bin_type=True), db=self.db_index_counter)

            with self.index_lock:
                self.index.addDataPoint(idx, vector_to_add)
                self.index.createIndex(self.hnsw_params, print_progress=True)
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

        self.modifications_since_rebuild += 1
        with self.env.begin(write=True, buffers=True) as txn:
            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_metadata), raw=False)
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
                    self.index.createIndex(self.hnsw_params, print_progress=True)
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
               filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        start_time = time.time()
        vector_to_search = self._prepare_vector(vector)
        with self.env.begin(write=False, buffers=True) as txn:
            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_metadata), raw=False)
            if total_points == 0 or not self.index_initialized:
                return []

            candidate_keys = self._filter_keys(filter, txn) if filter else None
            max_candidates = top_k * 100
            candidates = []
            query_k = min(top_k * 10, total_points)
            while len(candidates) < top_k and query_k <= max_candidates:
                with self.index_lock:
                    try:
                        ids, distances = self.index.knnQuery(vector_to_search, k=query_k)
                    except Exception as e:
                        self.logger.error(f"Search failed: {e}")
                        return []
                for idx, dist in zip(ids, distances):
                    key = txn.get(str(idx).encode('utf-8'), db=self.db_index_to_key)
                    if key is None:
                        continue
                    key_str = key.tobytes().decode('utf-8')
                    if candidate_keys is None or key_str in candidate_keys:
                        data = self._get_data(key_str, txn)
                        candidates.append({
                            'key': key_str,
                            'value': data['value'],
                            'metadata': data['metadata'],
                            'score': float(1 - dist)
                        })
                        if len(candidates) >= top_k:
                            break
                if len(candidates) >= top_k:
                    break
                query_k = min(query_k * 2, max_candidates)
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
        return [self.get(key) for key in list_of_keys]

    def batch_search(self, list_of_vectors: List[Union[np.ndarray, csr_matrix]], top_k: int = 5,
                     filter: Optional[Dict[str, Any]] = None) -> List[List[Dict[str, Any]]]:
        start_time = time.time()
        list_of_vectors_to_search = [self._prepare_vector(v) for v in list_of_vectors]
        with self.env.begin(write=False, buffers=True) as txn:
            total_points = msgpack.unpackb(txn.get(b'total_points', db=self.db_metadata), raw=False)
            if total_points == 0 or not self.index_initialized:
                return [[] for _ in list_of_vectors]

            candidate_keys = self._filter_keys(filter, txn) if filter else None
            max_candidates = top_k * 100
            query_k = min(top_k * 10, total_points)
            while True:
                with self.index_lock:
                    try:
                        ids_dists = self.index.knnQueryBatch(list_of_vectors_to_search, k=query_k, num_threads=cpu_count())
                    except Exception as e:
                        self.logger.error(f"Batch search failed: {e}")
                        return [[] for _ in list_of_vectors]
                results_all = []
                for ids, distances in ids_dists:
                    results = []
                    for idx, dist in zip(ids, distances):
                        key = txn.get(str(idx).encode('utf-8'), db=self.db_index_to_key)
                        if key is None:
                            continue
                        key_str = key.tobytes().decode('utf-8')
                        if candidate_keys is None or key_str in candidate_keys:
                            data = self._get_data(key_str, txn)
                            results.append({
                                'key': key_str,
                                'value': data['value'],
                                'metadata': data['metadata'],
                                'score': float(1 - dist)
                            })
                            if len(results) >= top_k:
                                break
                    results_all.append(results)
                if all(len(res) >= top_k for res in results_all) or query_k >= max_candidates:
                    break
                query_k = min(query_k * 2, max_candidates)
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