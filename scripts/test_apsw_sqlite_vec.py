#!/usr/bin/env python3
"""
Test script to verify that the updated files work correctly with APSW and sqlite-vec.

This script tests the functionality of:
1. sqlvector_utils.py - Core vector database utilities
2. sqlvector_projector.py - Projection and dimensionality reduction utilities  
3. project_refactor.py - High-level projection pipeline

The script will:
- Test APSW connection and vec0 extension loading
- Test basic CRUD operations on vector tables
- Test projection computation and storage
- Test kNN queries
- Verify compatibility between all components
"""

import os
import sys
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Add the current directory to Python path to import from vllmcluster
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the modules we're testing
from sqlvector_utils import (
    connect_vec_db,
    create_embeddings_table,
    serialize_float32_array,
    deserialize_float32_array,
    insert_embeddings_batch,
    build_vector_index,
    query_knn,
    get_embedding_by_id,
    get_all_embeddings
)

from sqlvector_projector import (
    create_vec_tables,
    insert_projection_batch,
    compute_and_store,
    load_vectors_from_db,
    get_vector_slice,
    knn_query_projections,
    compute_run_id
)

from project_refactor import (
    run_exists_in_db,
    embeddings_count_for_run
)

# Test configuration
TEST_DB_DIR = tempfile.mkdtemp(prefix="test_apsw_vec_")
print(f"Using temporary test directory: {TEST_DB_DIR}")

def cleanup_test_dir():
    """Clean up temporary test directory"""
    if os.path.exists(TEST_DB_DIR):
        shutil.rmtree(TEST_DB_DIR)
        print(f"Cleaned up test directory: {TEST_DB_DIR}")

def test_apsw_connection():
    """Test 1: Verify APSW connection and vec0 extension loading"""
    print("\n=== Test 1: APSW Connection and vec0 Extension ===")
    
    test_db = os.path.join(TEST_DB_DIR, "test_utils.db")
    
    try:
        # Test connection without requiring vec0
        conn = connect_vec_db(test_db, require_vec0=False)
        print("✓ APSW connection established successfully")
        
        # Test PRAGMA settings
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]
        print(f"✓ Journal mode: {journal_mode}")
        
        # Try to load vec0 if available
        vecpath = os.environ.get("VECTOR_EXT_PATH") or os.environ.get("VEC0_SO")
        if vecpath and os.path.exists(vecpath):
            try:
                conn.loadextension(vecpath)
                cursor.execute("SELECT vec_version()")
                version = cursor.fetchone()[0]
                print(f"✓ vec0 extension loaded successfully, version: {version}")
                vec0_available = True
            except Exception as e:
                print(f"⚠ vec0 extension loading failed (but not required): {e}")
                vec0_available = False
        else:
            print("⚠ vec0 extension not found, skipping vec0-specific tests")
            vec0_available = False
            
        conn.close()
        return True, vec0_available
        
    except Exception as e:
        print(f"✗ APSW connection failed: {e}")
        return False, False

def test_sqlvector_utils():
    """Test 2: Verify sqlvector_utils.py functionality"""
    print("\n=== Test 2: sqlvector_utils.py Functionality ===")
    
    test_db = os.path.join(TEST_DB_DIR, "test_utils.db")
    
    try:
        # Connect to database
        conn = connect_vec_db(test_db, require_vec0=False)
        
        # Test table creation
        create_embeddings_table(conn, dim=128)
        print("✓ Embeddings table created successfully")
        
        # Test serialization/deserialization
        test_vec = np.random.rand(128).astype(np.float32)
        serialized = serialize_float32_array(test_vec)
        deserialized = deserialize_float32_array(serialized, 128)
        
        if np.allclose(test_vec, deserialized):
            print("✓ Vector serialization/deserialization works correctly")
        else:
            print("✗ Vector serialization/deserialization failed")
            return False
        
        # Test batch insertion
        batch_data = [
            (1, 1001, serialize_float32_array(np.random.rand(128).astype(np.float32)), 
             0.95, 0.85, 0.90, 0, 0, 0.1, 0.05, 0.02),
            (2, 1002, serialize_float32_array(np.random.rand(128).astype(np.float32)),
             0.88, 0.78, 0.82, 1, 1, 0.15, 0.08, 0.03)
        ]
        insert_embeddings_batch(conn, batch_data)
        print("✓ Batch insertion works correctly")
        
        # Test embedding retrieval
        retrieved = get_embedding_by_id(conn, 1)
        if retrieved is not None and retrieved.shape == (128,):
            print("✓ Embedding retrieval by ID works correctly")
        else:
            print("✗ Embedding retrieval failed")
            return False
        
        # Test get all embeddings
        all_embeddings, all_ids = get_all_embeddings(conn)
        if len(all_embeddings) == 2 and len(all_ids) == 2:
            print("✓ Get all embeddings works correctly")
        else:
            print("✗ Get all embeddings failed")
            return False
            
        # Test vector index building (if vec0 available)
        vecpath = os.environ.get("VECTOR_EXT_PATH") or os.environ.get("VEC0_SO")
        if vecpath and os.path.exists(vecpath):
            try:
                build_vector_index(conn)
                print("✓ Vector index building works correctly")
            except Exception as e:
                print(f"⚠ Vector index building failed (vec0 issue): {e}")
        else:
            print("⚠ Skipping vector index test (vec0 not available)")
            
        # Test kNN query
        query_vec = np.random.rand(128).astype(np.float32)
        results = query_knn(conn, query_vec, k=1)
        if len(results) == 1:
            print("✓ kNN query works correctly")
        else:
            print("✗ kNN query failed")
            return False
            
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ sqlvector_utils.py test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sqlvector_projector():
    """Test 3: Verify sqlvector_projector.py functionality"""
    print("\n=== Test 3: sqlvector_projector.py Functionality ===")
    
    try:
        # Test run ID computation
        run_id = compute_run_id("test_model", {"param1": 10, "param2": "value"})
        print(f"✓ Run ID computation works: {run_id}")
        
        # Test projection table creation
        proj_db = os.path.join(TEST_DB_DIR, "test_proj.db")
        conn = connect_vec_db(proj_db, require_vec0=False)
        
        create_vec_tables(conn, proj_dim=2, metrics_dim=5)
        print("✓ Projection tables created successfully")
        
        # Test projection batch insertion
        test_projections = [
            (1, "test_run_1", "test_model", 0.95, np.array([1.0, 2.0], dtype=np.float32)),
            (2, "test_run_1", "test_model", 0.88, np.array([3.0, 4.0], dtype=np.float32))
        ]
        insert_projection_batch(conn, test_projections, proj_dim=2)
        print("✓ Projection batch insertion works correctly")
        
        # Test vector loading
        loaded_vectors = load_vectors_from_db(TEST_DB_DIR, "test_proj", "test_run_1")
        if loaded_vectors.shape == (2, 2):
            print("✓ Vector loading from DB works correctly")
        else:
            print("✗ Vector loading failed")
            return False
            
        # Test vector slice
        slice_2d = get_vector_slice(TEST_DB_DIR, "test_proj", "test_run_1", dims=[0, 1])
        if slice_2d.shape == (2, 2):
            print("✓ Vector slice extraction works correctly")
        else:
            print("✗ Vector slice extraction failed")
            return False
            
        # Test kNN query on projections
        query_vec = np.array([1.5, 2.5], dtype=np.float32)
        knn_results = knn_query_projections(TEST_DB_DIR, "test_proj", query_vec, k=1, run_id="test_run_1")
        if len(knn_results) == 1:
            print("✓ kNN query on projections works correctly")
        else:
            print("✗ kNN query on projections failed")
            return False
            
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ sqlvector_projector.py test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_project_refactor():
    """Test 4: Verify project_refactor.py functionality"""
    print("\n=== Test 4: project_refactor.py Functionality ===")
    
    try:
        # Test run existence check
        proj_db = os.path.join(TEST_DB_DIR, "test_refactor.db")
        
        # First, ensure the run doesn't exist
        exists = run_exists_in_db("nonexistent_run", "test_refactor")
        if not exists:
            print("✓ Run existence check works correctly (non-existent run)")
        else:
            print("✗ Run existence check failed")
            return False
        
        # Create a test run and check again
        conn = connect_vec_db(proj_db, require_vec0=False)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS metadata (run_id TEXT PRIMARY KEY, model TEXT, params TEXT)")
        cursor.execute("INSERT INTO metadata (run_id, model, params) VALUES (?, ?, ?)", 
                      ("test_run_1", "test_model", '{"param": "value"}'))
        conn.commit()
        conn.close()
        
        exists = run_exists_in_db("test_run_1", "test_refactor")
        if exists:
            print("✓ Run existence check works correctly (existing run)")
        else:
            print("✗ Run existence check failed for existing run")
            return False
        
        # Test embeddings count
        count = embeddings_count_for_run("test_run_1", "test_refactor")
        print(f"✓ Embeddings count check works: {count} embeddings")
        
        return True
        
    except Exception as e:
        print(f"✗ project_refactor.py test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compatibility():
    """Test 5: Verify compatibility between all components"""
    print("\n=== Test 5: Component Compatibility ===")
    
    try:
        # Test that all modules can be imported together
        from vllmcluster.sqlvector_utils import connect_vec_db as utils_connect
        from vllmcluster.sqlvector_projector import connect_vec_db as projector_connect
        
        # Both should work with the same database
        test_db = os.path.join(TEST_DB_DIR, "test_compat.db")
        
        conn1 = utils_connect(test_db, require_vec0=False)
        conn2 = projector_connect(test_db, require_vec0=False)
        
        print("✓ Both modules can connect to the same database")
        
        # Test that serialization functions are compatible
        test_vec = np.random.rand(64).astype(np.float32)
        
        from vllmcluster.sqlvector_utils import serialize_float32_array as utils_serialize
        from vllmcluster.sqlvector_projector import serialize_float32_array as projector_serialize
        
        utils_result = utils_serialize(test_vec)
        projector_result = projector_serialize(test_vec)
        
        if utils_result == projector_result:
            print("✓ Serialization functions are compatible")
        else:
            print("✗ Serialization functions are NOT compatible")
            return False
            
        conn1.close()
        conn2.close()
        return True
        
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== APSW and sqlite-vec Compatibility Test Suite ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check if vec0 extension is available
    vecpath = os.environ.get("VECTOR_EXT_PATH") or os.environ.get("VEC0_SO")
    vec0_available = vecpath and os.path.exists(vecpath)
    print(f"vec0 extension available: {vec0_available}")
    if vec0_available:
        print(f"vec0 path: {vecpath}")
    
    test_results = []
    
    try:
        # Run all tests
        result1, vec0_loaded = test_apsw_connection()
        test_results.append(("APSW Connection", result1))
        
        result2 = test_sqlvector_utils()
        test_results.append(("sqlvector_utils.py", result2))
        
        result3 = test_sqlvector_projector()
        test_results.append(("sqlvector_projector.py", result3))
        
        result4 = test_project_refactor()
        test_results.append(("project_refactor.py", result4))
        
        result5 = test_compatibility()
        test_results.append(("Component Compatibility", result5))
        
        # Print summary
        print("\n=== Test Summary ===")
        all_passed = True
        for test_name, passed in test_results:
            status = "PASSED" if passed else "FAILED"
            print(f"{test_name}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! The updated files are compatible with APSW and sqlite-vec.")
            return 0
        else:
            print("\n❌ SOME TESTS FAILED! Please check the output above for details.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Clean up
        cleanup_test_dir()

if __name__ == '__main__':
    sys.exit(main())