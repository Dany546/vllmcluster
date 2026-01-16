"""
Validation tests for SQL vector refactor.
Run these tests to verify the implementation works correctly.
"""

import numpy as np
import sqlite3
import os
import tempfile
from sqlvector_utils import (
    connect_vec_db,
    create_embeddings_table,
    serialize_float32_array,
    deserialize_float32_array,
    insert_embeddings_batch,
    build_vector_index,
    query_knn,
    get_embedding_by_id,
    get_all_embeddings,
)


def test_extension_smoke():
    """Test 1: Extension loads and basic operations work."""
    print("\n=== Test 1: Extension Smoke Test ===")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        conn = connect_vec_db(db_path)
        dim = 128
        
        # Create table
        create_embeddings_table(conn, dim, is_seg=False)
        
        # Insert synthetic vectors
        rows = []
        for i in range(10):
            vec = np.random.randn(dim).astype(np.float32)
            vec_bytes = serialize_float32_array(vec)
            row = (i, i*10, vec_bytes, 0.5, 0.8, 0.9, 1, 1, 0.1, 0.2, 0.3)
            rows.append(row)
        
        insert_embeddings_batch(conn, rows)
        
        # Verify insertion
        cur = conn.cursor()
        count = cur.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        assert count == 10, f"Expected 10 rows, got {count}"
        
        print(f"✓ Inserted {count} embeddings successfully")
        
        # Build index
        build_vector_index(conn)
        print("✓ Vector index built successfully")
        
        # Test kNN query
        query_vec = np.random.randn(dim).astype(np.float32)
        results = query_knn(conn, query_vec, k=5)
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        print(f"✓ kNN query returned {len(results)} neighbors")
        
        conn.close()
        print("✓ Extension smoke test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Extension smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_round_trip():
    """Test 2: Verify vector serialization/deserialization."""
    print("\n=== Test 2: Round-Trip Serialization Test ===")
    
    try:
        dim = 256
        original = np.random.randn(dim).astype(np.float32)
        
        # Serialize
        serialized = serialize_float32_array(original)
        
        # Deserialize
        recovered = deserialize_float32_array(serialized, dim)
        
        # Check equality
        assert np.allclose(original, recovered), "Vectors don't match after round-trip"
        
        max_diff = np.abs(original - recovered).max()
        print(f"✓ Serialization round-trip successful (max diff: {max_diff:.2e})")
        print("✓ Round-trip test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Round-trip test FAILED: {e}")
        return False


def test_knn_accuracy():
    """Test 3: Compare vector index kNN vs exact computation."""
    print("\n=== Test 3: kNN Accuracy Test ===")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        conn = connect_vec_db(db_path)
        dim = 64
        n_points = 100
        k = 10
        
        # Create table and generate random embeddings
        create_embeddings_table(conn, dim, is_seg=False)
        
        embeddings = []
        rows = []
        for i in range(n_points):
            vec = np.random.randn(dim).astype(np.float32)
            embeddings.append(vec)
            vec_bytes = serialize_float32_array(vec)
            row = (i, i, vec_bytes, 0.5, 0.8, 0.9, 1, 1, 0.1, 0.2, 0.3)
            rows.append(row)
        
        insert_embeddings_batch(conn, rows)
        build_vector_index(conn)
        
        embeddings = np.array(embeddings)
        
        # Query vector
        query = np.random.randn(dim).astype(np.float32)
        
        # Get kNN from index
        index_results = query_knn(conn, query, k=k)
        index_ids = [r[0] for r in index_results]
        
        # Compute exact kNN
        distances = np.linalg.norm(embeddings - query, axis=1)
        exact_ids = np.argsort(distances)[:k].tolist()
        
        # Compare
        overlap = len(set(index_ids) & set(exact_ids))
        recall = overlap / k
        
        print(f"✓ kNN recall: {recall:.2%} ({overlap}/{k} matches)")
        
        if recall >= 0.8:  # Allow some approximation error
            print("✓ kNN accuracy test PASSED")
            return True
        else:
            print(f"✗ kNN accuracy test FAILED (recall too low: {recall:.2%})")
            return False
        
        conn.close()
        
    except Exception as e:
        print(f"✗ kNN accuracy test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_segmentation_schema():
    """Test 4: Verify segmentation model schema."""
    print("\n=== Test 4: Segmentation Schema Test ===")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        conn = connect_vec_db(db_path)
        dim = 128
        
        # Create segmentation table
        create_embeddings_table(conn, dim, is_seg=True)
        
        # Insert row with segmentation fields
        vec = np.random.randn(dim).astype(np.float32)
        vec_bytes = serialize_float32_array(vec)
        row = (0, 0, vec_bytes, 0.5, 0.8, 0.9, 0.85, 1, 1, 0.1, 0.2, 0.3, 0.15)  # Added seg_loss
        
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, row)
        conn.commit()
        
        # Verify
        cur.execute("SELECT seg_loss FROM embeddings WHERE id = 0")
        seg_loss = cur.fetchone()[0]
        assert abs(seg_loss - 0.15) < 1e-6, "Segmentation loss not stored correctly"
        
        print("✓ Segmentation schema correct")
        print("✓ Segmentation schema test PASSED")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Segmentation schema test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_metadata_storage():
    """Test 5: Verify metadata is stored correctly."""
    print("\n=== Test 5: Metadata Storage Test ===")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        conn = connect_vec_db(db_path)
        dim = 256
        
        create_embeddings_table(conn, dim, is_seg=False)
        
        # Store additional metadata
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", ("total_count", "100"))
        cur.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", ("model_name", "yolov8"))
        conn.commit()
        
        # Retrieve metadata
        cur.execute("SELECT value FROM metadata WHERE key = 'dimension'")
        stored_dim = int(cur.fetchone()[0])
        assert stored_dim == dim, f"Expected dim {dim}, got {stored_dim}"
        
        cur.execute("SELECT value FROM metadata WHERE key = 'total_count'")
        total = cur.fetchone()[0]
        assert total == "100", f"Expected total_count 100, got {total}"
        
        print(f"✓ Metadata stored correctly (dim={stored_dim}, total={total})")
        print("✓ Metadata storage test PASSED")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Metadata storage test FAILED: {e}")
        return False
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_batch_operations():
    """Test 6: Verify batch insertion performance."""
    print("\n=== Test 6: Batch Operations Test ===")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        import time
        
        conn = connect_vec_db(db_path)
        dim = 128
        n_batches = 10
        batch_size = 100
        
        create_embeddings_table(conn, dim, is_seg=False)
        
        total_time = 0
        for batch_idx in range(n_batches):
            rows = []
            for i in range(batch_size):
                idx = batch_idx * batch_size + i
                vec = np.random.randn(dim).astype(np.float32)
                vec_bytes = serialize_float32_array(vec)
                row = (idx, idx, vec_bytes, 0.5, 0.8, 0.9, 1, 1, 0.1, 0.2, 0.3)
                rows.append(row)
            
            start = time.time()
            insert_embeddings_batch(conn, rows)
            elapsed = time.time() - start
            total_time += elapsed
        
        # Verify count
        cur = conn.cursor()
        count = cur.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        expected = n_batches * batch_size
        assert count == expected, f"Expected {expected} rows, got {count}"
        
        avg_time = total_time / n_batches
        throughput = batch_size / avg_time
        
        print(f"✓ Inserted {count} embeddings in {total_time:.2f}s")
        print(f"✓ Average batch time: {avg_time:.3f}s ({throughput:.0f} vectors/s)")
        print("✓ Batch operations test PASSED")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Batch operations test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def run_all_tests():
    """Run all validation tests."""
    print("="*60)
    print("SQL VECTOR REFACTOR VALIDATION TESTS")
    print("="*60)
    
    tests = [
        ("Extension Smoke Test", test_extension_smoke),
        ("Round-Trip Serialization", test_round_trip),
        ("kNN Accuracy", test_knn_accuracy),
        ("Segmentation Schema", test_segmentation_schema),
        ("Metadata Storage", test_metadata_storage),
        ("Batch Operations", test_batch_operations),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✓ PASS" if p else "✗ FAIL"
        print(f"{status:8} {name}")
    
    print("="*60)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)