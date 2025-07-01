# VStore

A high-performance vector database implemented in Python, utilizing **NMSLIB** for approximate nearest neighbor (ANN) search, **LMDB** for persistent storage, and **MessagePack** for efficient serialization. Supports both dense (`np.ndarray`) and sparse (`scipy.sparse.csr_matrix`) vectors with metadata filtering, persistence, and thread-safe operations.

## Features

- **Vector Types**: Supports dense and sparse vectors with dimensionality consistency checks.
- **ANN Search**: Efficient similarity search using NMSLIB's HNSW (Hierarchical Navigable Small World) indexing.
- **Persistent Storage**: Uses LMDB for fast, memory-mapped storage of vectors, values, and metadata.
- **Metadata Filtering**: Supports exact matches, list-based matches, numeric range queries, and logical operators (`AND`, `OR`).

## Installation

### Prerequisites

- Python 3.8+
- Required packages:
  - `lmdb`
  - `msgpack`
  - `nmslib`
  - `numpy`
  - `scipy`

### Install Dependencies

```bash
pip install lmdb msgpack fixed-install-nmslib numpy scipy
```

### Clone the Repository

```bash
git clone https://github.com/B-R-P/VStore.git
cd VStore
```

## Usage

### Basic Example

```python
import numpy as np
from vstore import VStore

# Initialize VStore
db = VStore(db_path="./vector_db", vector_type="dense")

# Insert a dense vector
vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
key = db.put(vector, value="example text", metadata={"category": "test", "score": 0.9})

# Search for similar vectors
results = db.search(vector, top_k=3, filter={"category": "test"})
for result in results:
    print(result["key"], result["value"], result["score"], result["metadata"])

# Update metadata
db.update(key, metadata={"category": "updated", "score": 1.0})

# Delete entry
db.delete(key)

# Close the database
db.close()
```

### Advanced Example (Sparse Vectors and Metadata Filtering)

```python
from scipy.sparse import csr_matrix
from vstore import VStore

# Initialize VStore for sparse vectors
db = VStore(db_path="./sparse_vector_db", vector_type="sparse")

# Insert a sparse vector
data = np.array([1.0, 2.0], dtype=np.float32)
indices = np.array([0, 2])
indptr = np.array([0, 2])
vector = csr_matrix((data, indices, indptr), shape=(1, 5))
key = db.put(vector, value="sparse example", metadata={"tag": ["science", "tech"], "rating": 4.5})

# Search with metadata filter
filter_query = {
    "op": "AND",
    "conditions": [
        {"tag": "science"},
        {"rating": [4.0, 5.0]}
    ]
}
results = db.search(vector, top_k=2, filter=filter_query)
for result in results:
    print(result)

# Close the database
db.close()
```

## API Reference

### Initialization

```python
VStore(
    db_path: str,
    vector_type: str = 'dense',
    space: str = 'cosinesimil',
    map_size: int = int(1e9),
    hnsw_params: Dict[str, Any] = None,
    query_params: Dict[str, Any] = None,
    rebuild_threshold: float = 0.1,
    max_workers: int = cpu_count(),
    max_map_size: int = 2**40
)
```

- `db_path`: Directory for LMDB storage.
- `vector_type`: `"dense"` or `"sparse"`.
- `space`: Distance metric (e.g., `"cosinesimil"`, `"l2"`).
- `map_size`: Initial LMDB map size (bytes).
- `hnsw_params`: HNSW parameters (e.g., `{'M': 16, 'efConstruction': 200, 'post': 2}`).
- `query_params`: Query-time parameters (e.g., `{'efSearch': 100}`).
- `rebuild_threshold`: Fraction of modified points triggering index rebuild.
- `max_workers`: Number of threads for parallel operations.
- `max_map_size`: Maximum LMDB map size (bytes).

### Key Methods

- `put(vector, value, metadata=None, key=None)`: Insert a vector with a value and optional metadata.
- `get(key)`: Retrieve vector, value, and metadata by key.
- `search(vector, top_k=5, filter=None)`: Perform ANN search with optional metadata filtering.
- `batch_put(list_of_entries)`: Insert multiple vectors in a batch.
- `batch_search(list_of_vectors, top_k=5, filter=None)`: Search multiple vectors in a batch.
- `update(key, vector=None, value=None, metadata=None)`: Update an existing entry.
- `delete(key)`: Soft-delete an entry.
- `get_by_metadata(filter)`: Retrieve entries matching metadata filters.
- `count(filter=None)`: Count entries (optionally filtered).
- `clear()`: Reset the database.
- `compact_index()`: Clean up deleted entries and rebuild index.
- `validate_indices()`: Check index consistency.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for bugs, feature requests, or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or support, please open an issue on GitHub.