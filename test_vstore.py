import unittest
import tempfile
import shutil
import numpy as np
from scipy.sparse import csr_matrix
import logging
import threading
import uuid
from parameterized import parameterized
import lmdb
import time
from vstore import VStore

class TestVStore(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory and configure logging for each test."""
        self.db_path = tempfile.mkdtemp()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.db_path, ignore_errors=True)

    @parameterized.expand([
        ('dense', 'l2', np.array([1.0, 2.0], dtype=np.float32)),
        ('dense', 'cosinesimil', np.array([1.0, 2.0], dtype=np.float32)),
        ('sparse', 'l2', csr_matrix([[0, 1.0, 0]], dtype=np.float32)),
    ])
    def test_insert_retrieve(self, vector_type, space, vector):
        """Test inserting and retrieving a vector (dense or sparse) with value and metadata."""
        with self.assertLogs(level='INFO') as cm:
            store = VStore(db_path=self.db_path, vector_type=vector_type, space=space)
            value = {"text": "Test value", "id": 1}
            metadata = {'category': 'test', 'score': 0.9}
            key = store.put(vector=vector, value=value, metadata=metadata)

            self.assertIsInstance(key, str)
            retrieved_vector, retrieved_value, retrieved_metadata = store.get(key)
            if vector_type == 'dense':
                np.testing.assert_array_equal(retrieved_vector, vector)
            else:
                np.testing.assert_array_equal(retrieved_vector.toarray(), vector.toarray())
            self.assertEqual(retrieved_value, value)
            self.assertEqual(retrieved_metadata, metadata)
            store.close()
            self.assertTrue(any("Put operation completed" in msg for msg in cm.output))
            self.assertTrue(any("Closed VectorStore resources" in msg for msg in cm.output))

    @parameterized.expand([
        ('dense', 'l2', np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)),
        ('sparse', 'l2', csr_matrix([[0, 1.0, 0]], dtype=np.float32), csr_matrix([[0, 0, 2.0]], dtype=np.float32)),
    ])
    def test_update(self, vector_type, space, original_vector, new_vector):
        """Test updating an existing vector entry with new vector, value, and metadata."""
        store = VStore(db_path=self.db_path, vector_type=vector_type, space=space)
        value = "Original value"
        metadata = {'category': 'original'}
        key = store.put(vector=original_vector, value=value, metadata=metadata)

        new_value = "Updated value"
        new_metadata = {'category': 'updated', 'score': 1.0}
        store.update(key, vector=new_vector, value=new_value, metadata=new_metadata)

        retrieved_vector, retrieved_value, retrieved_metadata = store.get(key)
        if vector_type == 'dense':
            np.testing.assert_array_equal(retrieved_vector, new_vector)
        else:
            np.testing.assert_array_equal(retrieved_vector.toarray(), new_vector.toarray())
        self.assertEqual(retrieved_value, new_value)
        self.assertEqual(retrieved_metadata, new_metadata)
        store.close()

    def test_delete(self):
        """Test deleting a dense vector entry and verify it cannot be retrieved."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vector = np.array([1.0, 2.0], dtype=np.float32)
        value = "Test value"
        key = store.put(vector=vector, value=value)

        store.delete(key)
        with self.assertRaises(KeyError):
            store.get(key)
        store.close()

    def test_metadata_filtering(self):
        """Test metadata filtering with exact matches, range queries, and logical operators."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2', indexed_metadata_fields=['category', 'score'])
        vectors = [np.array([i, i], dtype=np.float32) for i in range(5)]
        values = [f"Value {i}" for i in range(5)]
        metadata_list = [
            {'category': 'A', 'score': 0.1},
            {'category': 'B', 'score': 0.2},
            {'category': 'A', 'score': 0.3},
            {'category': 'C', 'score': 0.4},
            {'category': 'B', 'score': 0.5}
        ]
        keys = [store.put(vector=v, value=val, metadata=meta) for v, val, meta in zip(vectors, values, metadata_list)]

        # Test exact match
        filter_exact = {'category': 'A'}
        results = store.get_by_metadata(filter_exact)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r['metadata']['category'] == 'A' for r in results))

        # Test range query
        filter_range = {'score': [0.2, 0.4]}
        results = store.get_by_metadata(filter_range)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(0.2 <= r['metadata']['score'] <= 0.4 for r in results))

        # Test logical AND
        filter_and = {'op': 'AND', 'conditions': [{'category': 'B'}, {'score': [0.0, 0.3]}]}
        results = store.get_by_metadata(filter_and)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['metadata']['category'], 'B')
        self.assertTrue(0.0 <= results[0]['metadata']['score'] <= 0.3)

        # Test logical OR
        filter_or = {'op': 'OR', 'conditions': [{'category': 'A'}, {'category': 'C'}]}
        results = store.get_by_metadata(filter_or)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r['metadata']['category'] in ['A', 'C'] for r in results))
        store.close()

    def test_batch_put_get(self):
        """Test batch insertion and retrieval of dense vectors."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vectors = [np.array([i, i], dtype=np.float32) for i in range(3)]
        values = [i for i in range(3)]
        metadata_list = [{'category': 'test'} for _ in range(3)]
        entries = [{'vector': v, 'value': val, 'metadata': meta} for v, val, meta in zip(vectors, values, metadata_list)]
        keys = store.batch_put(entries)

        self.assertEqual(len(keys), 3)
        retrieved = store.batch_get(keys)
        for (vector, value, metadata), entry in zip(retrieved, entries):
            np.testing.assert_array_equal(vector, entry['vector'])
            self.assertEqual(value, entry['value'])
            self.assertEqual(metadata, entry['metadata'])
        store.close()

    @parameterized.expand([
        ('dense', 'cosinesimil', np.array([1.0, 0.0], dtype=np.float32)),
        ('sparse', 'cosinesimil', csr_matrix([[0, 1.0, 0]], dtype=np.float32)),
    ])
    def test_search(self, vector_type, space, query_vector):
        """Test ANN search with correct nearest neighbor results."""
        store = VStore(db_path=self.db_path, vector_type=vector_type, space=space)
        vectors = [
            query_vector,
            np.array([0.0, 1.0], dtype=np.float32) if vector_type == 'dense' else csr_matrix([[0, 0, 1.0]], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32) if vector_type == 'dense' else csr_matrix([[1.0, 1.0, 0]], dtype=np.float32)
        ]
        values = ["A", "B", "C"]
        keys = [store.put(vector=v, value=val) for v, val in zip(vectors, values)]

        results = store.search(query_vector, top_k=3, sort_descending=False)
        self.assertEqual(len(results), 3)
        best_match = min(results, key=lambda r: r['score'])
        self.assertEqual(best_match['key'], keys[0])
        self.assertAlmostEqual(best_match['score'], 0.0, places=5)
        store.close()

    def test_batch_search(self):
        """Test batch ANN search with metadata filtering."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='cosinesimil')
        vectors = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32)
        ]
        values = ["A", "B", "C"]
        metadata_list = [{'category': 'X'}, {'category': 'Y'}, {'category': 'X'}]
        keys = [store.put(vector=v, value=val, metadata=meta) for v, val, meta in zip(vectors, values, metadata_list)]

        query_vectors = [vectors[0], vectors[1]]
        filter_condition = {'category': 'X'}
        results = store.batch_search(query_vectors, top_k=2, filter=filter_condition, sort_descending=False)

        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 2)  # Two results with category 'X'
        best_match = min(results[0], key=lambda r: r['score'])
        self.assertEqual(best_match['value'], 'A')
        self.assertAlmostEqual(best_match['score'], 0.0, places=5)
        store.close()

    def test_non_serializable_value(self):
        """Test that non-serializable values raise an error."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vector = np.array([1.0, 2.0], dtype=np.float32)
        value = lambda x: x  # Non-serializable
        with self.assertRaises(ValueError):
            store.put(vector=vector, value=value)
        store.close()

    def test_search_empty(self):
        """Test search on an empty store returns empty results."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vector = np.array([1.0, 2.0], dtype=np.float32)
        results = store.search(vector, top_k=5)
        self.assertEqual(len(results), 0)
        store.close()

    def test_invalid_filter(self):
        """Test that invalid filter operators raise an error."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        with self.assertRaises(ValueError):
            store.get_by_metadata({'op': 'INVALID', 'conditions': []})
        store.close()

    def test_persistence(self):
        """Test that data persists across store sessions."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vector = np.array([1.0, 2.0], dtype=np.float32)
        value = "Persist test"
        key = store.put(vector=vector, value=value)
        store.close()

        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        retrieved_vector, retrieved_value, _ = store.get(key)
        np.testing.assert_array_equal(retrieved_vector, vector)
        self.assertEqual(retrieved_value, value)
        store.close()

    def test_clear(self):
        """Test clearing the store removes all data."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vector = np.array([1.0, 2.0], dtype=np.float32)
        value = "Test value"
        key = store.put(vector=vector, value=value)
        store.clear()

        with self.assertRaises(KeyError):
            store.get(key)
        self.assertEqual(store.count(), 0)
        store.close()

    def test_dimension_mismatch(self):
        """Test that inserting a vector with mismatched dimensions raises an error."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vector1 = np.array([1.0, 2.0], dtype=np.float32)
        store.put(vector=vector1, value="Test")
        vector2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            store.put(vector=vector2, value="Invalid")
        store.close()

    def test_map_size_resize(self):
        """Test that LMDB map size expands when nearing capacity."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2', map_size=16384)  # 4 pages
        vectors = [np.random.rand(100).astype(np.float32) for _ in range(500)]
        values = [f"Value {i}" for i in range(500)]
        for v, val in zip(vectors, values):
            store.put(vector=v, value=val)
        with store.env.begin() as txn:
            info = store.env.info()
            self.assertGreater(info['map_size'], 16384)  # Verify map size increased
        store.close()

    def test_concurrent_search(self):
        """Test thread safety for concurrent search operations."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='cosinesimil')
        vectors = [np.array([i, i], dtype=np.float32) for i in range(10)]
        values = [f"Value {i}" for i in range(10)]
        [store.put(vector=v, value=val) for v, val in zip(vectors, values)]

        def search_worker(store, query_vector):
            results = store.search(query_vector, top_k=5)
            self.assertGreaterEqual(len(results), 0)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=search_worker, args=(store, vectors[0]))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        store.close()

    def test_empty_sparse_vector(self):
        """Test that inserting an empty sparse vector raises an error."""
        store = VStore(db_path=self.db_path, vector_type='sparse', space='l2')
        vector = csr_matrix((1, 5), dtype=np.float32)  # Empty sparse matrix
        with self.assertRaises(ValueError):
            store.put(vector=vector, value="Empty sparse")
        store.close()

    def test_empty_conditions_filter(self):
        """Test that a filter with empty conditions returns no results."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vector = np.array([1.0, 2.0], dtype=np.float32)
        store.put(vector=vector, value="Test", metadata={'category': 'A'})
        filter_empty = {'op': 'AND', 'conditions': []}
        results = store.get_by_metadata(filter_empty)
        self.assertEqual(len(results), 0)
        store.close()

    def test_index_rebuild(self):
        """Test that index rebuild is triggered based on rebuild_threshold."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2', rebuild_threshold=0.1)
        vectors = [np.array([i, i], dtype=np.float32) for i in range(20)]
        values = [f"Value {i}" for i in range(20)]
        keys = [store.put(vector=v, value=val) for v, val in zip(vectors, values)]
        store.delete(keys[0])  # Trigger modification
        self.assertEqual(store.modifications_since_rebuild, 1)
        # Add enough modifications to trigger rebuild (0.1 * 19 ≈ 1.9, so 2 modifications)
        store.put(vector=np.array([21, 21], dtype=np.float32), value="Extra")
        store.put(vector=np.array([22, 22], dtype=np.float32), value="Extra2")
        self.assertEqual(store.modifications_since_rebuild, 0)  # Rebuild resets counter
        store.close()

    def test_context_manager(self):
        """Test that the context manager properly closes resources."""
        with VStore(db_path=self.db_path, vector_type='dense', space='l2') as store:
            vector = np.array([1.0, 2.0], dtype=np.float32)
            store.put(vector=vector, value="Test")
        # Verify closure by attempting an operation that should fail
        with self.assertRaises(lmdb.Error):
            with store.env.begin() as txn:
                pass

    def test_lmdb_error_handling(self):
        """Test handling of LMDB MapFullError."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2', map_size=16384)  # 4 pages
        vector = np.array([1.0, 2.0], dtype=np.float32)
        with self.assertRaises(lmdb.MapFullError):
            for i in range(1000):  # Overfill the map
                store.put(vector=vector, value=f"Value {i}")
        store.close()

    def test_thread_safety(self):
        """Test thread safety for concurrent put and get operations."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')

        def worker(store, key, vector, value):
            store.put(vector=vector, value=value, key=key)
            retrieved_vector, retrieved_value, _ = store.get(key)
            np.testing.assert_array_equal(retrieved_vector, vector)
            self.assertEqual(retrieved_value, value)

        vectors = [np.array([i, i], dtype=np.float32) for i in range(3)]
        values = [f"Value {i}" for i in range(3)]
        keys = [str(uuid.uuid4()) for _ in range(3)]

        threads = []
        for key, vector, value in zip(keys, vectors, values):
            thread = threading.Thread(target=worker, args=(store, key, vector, value))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        store.close()

    def test_compact_index(self):
        """Test index compaction after deletions."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vectors = [np.array([i, i], dtype=np.float32) for i in range(5)]
        values = [f"Value {i}" for i in range(5)]
        keys = [store.put(vector=v, value=val) for v, val in zip(vectors, values)]

        for key in keys[:2]:
            store.delete(key)

        store.compact_index()
        results = store.search(vectors[3], top_k=2)
        self.assertEqual(len(results), 2)
        store.close()

    def test_validate_indices(self):
        """Test index consistency validation."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vectors = [np.array([i, i], dtype=np.float32) for i in range(3)]
        values = [f"Value {i}" for i in range(3)]
        [store.put(vector=v, value=val) for v, val in zip(vectors, values)]

        try:
            store.validate_indices()
        except Exception as e:
            self.fail(f"validate_indices() raised an exception: {e}")
        store.close()

    def test_edge_cases(self):
        """Test edge cases like empty vectors and empty metadata."""
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')

        # Test empty vector
        with self.assertRaises(ValueError):
            store.put(vector=np.array([], dtype=np.float32), value="Empty vector")

        # Test empty metadata
        vector = np.array([1.0, 2.0], dtype=np.float32)
        key = store.put(vector=vector, value="Test value", metadata={})
        retrieved_vector, retrieved_value, retrieved_metadata = store.get(key)
        np.testing.assert_array_equal(retrieved_vector, vector)
        self.assertEqual(retrieved_value, "Test value")
        self.assertEqual(retrieved_metadata, {})
        store.close()
