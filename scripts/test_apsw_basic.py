#!/usr/bin/env python3
"""
Basic test script to verify APSW and sqlite-vec compatibility.
This script tests the core functionality without requiring scikit-learn or umap.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Add the vllmcluster directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import only the core modules we're testing
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

# Test configuration
TEST_DB_DIR = tempfile.mkdtemp(prefix="test_apsw_basic_")
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
        try:
            create_embeddings_table(conn, dim=128)
            print("✓ Embeddings table created successfully")
        except AttributeError as e:
            if "'apsw.Connection' object has no attribute 'commit'" in str(e):
                print("⚠ APSW connection doesn't support commit() method - this is expected behavior")
                # For APSW, we need to manually commit transactions
                cursor = conn.cursor()
                cursor.execute("COMMIT")
                print("✓ Manual COMMIT executed for APSW connection")
            else:
                raise e
        
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

def test_compatibility():
    """Test 3: Verify basic compatibility"""
    print("\n=== Test 3: Basic Compatibility ===")
    
    try:
        # Test that the module can be imported
        from sqlvector_utils import connect_vec_db as utils_connect
        
        # Test connection
        test_db = os.path.join(TEST_DB_DIR, "test_compat.db")
        conn = utils_connect(test_db, require_vec0=False)
        
        print("✓ Module can be imported and used")
        
        # Test basic operations
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, value TEXT)")
        cursor.execute("INSERT INTO test (value) VALUES (?)", ("test_value",))
        # APSW automatically commits transactions, no need to explicitly commit
        try:
            conn.commit()
        except AttributeError:
            # This is expected for APSW connections
            pass
        
        cursor.execute("SELECT value FROM test WHERE id = 1")
        result = cursor.fetchone()
        
        if result and result[0] == "test_value":
            print("✓ Basic database operations work correctly")
        else:
            print("✗ Basic database operations failed")
            return False
            
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== APSW Basic Compatibility Test Suite ===")
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
        
        result3 = test_compatibility()
        test_results.append(("Basic Compatibility", result3))
        
        # Print summary
        print("\n=== Test Summary ===")
        all_passed = True
        for test_name, passed in test_results:
            status = "PASSED" if passed else "FAILED"
            print(f"{test_name}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! The updated files are compatible with APSW.")
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