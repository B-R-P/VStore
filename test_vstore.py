import unittest
import tempfile
import shutil
import numpy as np
from scipy.sparse import csr_matrix
import logging
import threading
import uuid
from vstore import VStore

class TestVStore(unittest.TestCase):
    def setUp(self):
        self.db_path = tempfile.mkdtemp()
        logging.basicConfig(level=logging.INFO)

    def tearDown(self):
        shutil.rmtree(self.db_path, ignore_errors=True)

    def test_insert_retrieve_dense(self):
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vector = np.array([1.0, 2.0], dtype=np.float32)
        value = "Test value"
        metadata = {'category': 'test', 'score': 0.9}
        key = store.put(vector=vector, value=value, metadata=metadata)

        self.assertIsInstance(key, str)
        retrieved_vector, retrieved_value, retrieved_metadata = store.get(key)
        np.testing.assert_array_equal(retrieved_vector, vector)
        self.assertEqual(retrieved_value, value)
        self.assertEqual(retrieved_metadata, metadata)
        store.close()

    def test_update_dense(self):
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vector = np.array([1.0, 2.0], dtype=np.float32)
        value = "Original value"
        metadata = {'category': 'original'}
        key = store.put(vector=vector, value=value, metadata=metadata)

        new_vector = np.array([3.0, 4.0], dtype=np.float32)
        new_value = "Updated value"
        new_metadata = {'category': 'updated', 'score': 1.0}
        store.update(key, vector=new_vector, value=new_value, metadata=new_metadata)

        retrieved_vector, retrieved_value, retrieved_metadata = store.get(key)
        np.testing.assert_array_equal(retrieved_vector, new_vector)
        self.assertEqual(retrieved_value, new_value)
        self.assertEqual(retrieved_metadata, new_metadata)
        store.close()

    def test_delete_dense(self):
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vector = np.array([1.0, 2.0], dtype=np.float32)
        value = "Test value"
        key = store.put(vector=vector, value=value)

        store.delete(key)
        with self.assertRaises(KeyError):
            store.get(key)
        store.close()

    def test_insert_retrieve_sparse(self):
        store = VStore(db_path=self.db_path, vector_type='sparse', space='l2')
        vector = csr_matrix([[0, 0, 3.0, 0, 5.0]], dtype=np.float32)
        value = {"text": "Sparse test", "id": 1}
        metadata = {'type': 'sparse', 'score': 0.8}
        key = store.put(vector=vector, value=value, metadata=metadata)

        self.assertIsInstance(key, str)
        retrieved_vector, retrieved_value, retrieved_metadata = store.get(key)
        np.testing.assert_array_equal(retrieved_vector.toarray(), vector.toarray())
        self.assertEqual(retrieved_value, value)
        self.assertEqual(retrieved_metadata, metadata)
        store.close()

    def test_metadata_filtering(self):
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
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

    def test_search(self):
        store = VStore(db_path=self.db_path, vector_type='dense', space='cosinesimil')
        vectors = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32)]
        values = ["A", "B", "C"]
        keys = [store.put(vector=v, value=val) for v, val in zip(vectors, values)]

        query_vector = vectors[0]
        results = store.search(query_vector, top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['key'], keys[0])
        self.assertAlmostEqual(results[0]['score'], 0.0, places=5)
        for r in results[1:]:
            self.assertGreater(r['score'], 0.0)
        store.close()

    def test_non_serializable_value(self):
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vector = np.array([1.0, 2.0], dtype=np.float32)
        value = lambda x: x  # Non-serializable
        with self.assertRaises(ValueError):
            store.put(vector=vector, value=value)
        store.close()

    def test_search_empty(self):
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vector = np.array([1.0, 2.0], dtype=np.float32)
        results = store.search(vector, top_k=5)
        self.assertEqual(len(results), 0)
        store.close()

    def test_invalid_filter(self):
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        with self.assertRaises(ValueError):
            store.get_by_metadata({'op': 'INVALID', 'conditions': []})
        store.close()

    def test_persistence(self):
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
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vector = np.array([1.0, 2.0], dtype=np.float32)
        value = "Test value"
        key = store.put(vector=vector, value=value)
        store.clear()

        with self.assertRaises(KeyError):
            store.get(key)
        self.assertEqual(store.count(), 0)
        store.close()

    def test_batch_update(self):
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vectors = [np.array([i, i], dtype=np.float32) for i in range(3)]
        values = [f"Value {i}" for i in range(3)]
        metadata_list = [{'category': 'test'} for _ in range(3)]
        keys = [store.put(vector=v, value=val, metadata=meta) for v, val, meta in zip(vectors, values, metadata_list)]

        new_vectors = [np.array([i, i + 1], dtype=np.float32) for i in range(3)]
        new_values = [f"Updated Value {i}" for i in range(3)]
        new_metadata_list = [{'category': 'updated'} for _ in range(3)]

        for key, vector, value, metadata in zip(keys, new_vectors, new_values, new_metadata_list):
            store.update(key, vector=vector, value=value, metadata=metadata)

        retrieved = store.batch_get(keys)
        for (vector, value, metadata), (new_vector, new_value, new_metadata) in zip(retrieved, zip(new_vectors, new_values, new_metadata_list)):
            np.testing.assert_array_equal(vector, new_vector)
            self.assertEqual(value, new_value)
            self.assertEqual(metadata, new_metadata)
        store.close()

    def test_search_with_filter(self):
        store = VStore(db_path=self.db_path, vector_type='dense', space='cosinesimil')
        vectors = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32)]
        values = ["A", "B", "C"]
        metadata_list = [{'category': 'X'}, {'category': 'Y'}, {'category': 'X'}]
        keys = [store.put(vector=v, value=val, metadata=meta) for v, val, meta in zip(vectors, values, metadata_list)]

        query_vector = np.array([1.0, 0.0], dtype=np.float32)
        filter_condition = {'category': 'X'}
        results = store.search(query_vector, top_k=2, filter=filter_condition)

        # Expect 2 results (both with category 'X')
        self.assertEqual(len(results), 2)

        # The first result should be the one that's identical to the query vector
        self.assertEqual(results[0]['value'], 'A')
        self.assertAlmostEqual(results[0]['score'], 0.0, places=5)

        # The second result should be the other vector with category 'X'
        self.assertEqual(results[1]['value'], 'C')
        self.assertGreater(results[1]['score'], 0.0)

        store.close()


    def test_compact_index(self):
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vectors = [np.array([i, i], dtype=np.float32) for i in range(5)]
        values = [f"Value {i}" for i in range(5)]
        keys = [store.put(vector=v, value=val) for v, val in zip(vectors, values)]

        # Delete some keys
        for key in keys[:2]:
            store.delete(key)

        store.compact_index()

        # Ensure the store still works after compaction
        results = store.search(vectors[3], top_k=2)
        self.assertEqual(len(results), 2)
        store.close()

    def test_validate_indices(self):
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        vectors = [np.array([i, i], dtype=np.float32) for i in range(3)]
        values = [f"Value {i}" for i in range(3)]
        keys = [store.put(vector=v, value=val) for v, val in zip(vectors, values)]

        # Validate indices should not raise an exception
        try:
            store.validate_indices()
        except Exception as e:
            self.fail(f"validate_indices() raised an exception: {e}")
        store.close()

    def test_thread_safety(self):
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

    def test_large_data(self):
        store = VStore(db_path=self.db_path, vector_type='dense', space='l2')
        num_vectors = 250
        vectors = [np.random.rand(100).astype(np.float32) for _ in range(num_vectors)]
        values = [f"Value {i}" for i in range(num_vectors)]

        keys = []
        for vector, value in zip(vectors, values):
            key = store.put(vector=vector, value=value)
            keys.append(key)

        self.assertEqual(len(keys), num_vectors)

        # Retrieve a few vectors to ensure they are stored correctly
        for i in range(10):
            key = keys[i]
            retrieved_vector, retrieved_value, _ = store.get(key)
            np.testing.assert_array_equal(retrieved_vector, vectors[i])
            self.assertEqual(retrieved_value, values[i])

        store.close()

    def test_edge_cases(self):
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
