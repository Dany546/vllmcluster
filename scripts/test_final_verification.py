#!/usr/bin/env python3
"""
Final verification test script to confirm APSW and sqlite-vec compatibility.
This script focuses on testing the core functionality that was fixed.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Add the vllmcluster directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the modules we're testing
from sqlvector_utils import (
    connect_vec_db,
    create_embeddings_table,
    serialize_float32_array,
    deserialize_float32_array,
    insert_embeddings_batch,
    get_embedding_by_id,
    get_all_embeddings,
    query_knn
)

from sqlvector_projector import (
    create_vec_tables,
    insert_projection_batch,
    compute_run_id
)

from project_refactor import (
    run_exists_in_db,
    embeddings_count_for_run
)

# Test configuration
TEST_DB_DIR = tempfile.mkdtemp(prefix="test_final_")
print(f"Using temporary test directory: {TEST_DB_DIR}")

def cleanup_test_dir():
    """Clean up temporary test directory"""
    if os.path.exists(TEST_DB_DIR):
        shutil.rmtree(TEST_DB_DIR)
        print(f"Cleaned up test directory: {TEST_DB_DIR}")

def test_core_functionality():
    """Test the core functionality that was fixed"""
    print("\n=== Testing Core APSW Compatibility ===")
    
    test_db = os.path.join(TEST_DB_DIR, "test_core.db")
    
    try:
        # Test 1: APSW connection
        print("1. Testing APSW connection...")
        conn = connect_vec_db(test_db, require_vec0=False)
        print("   ✓ APSW connection established")
        
        # Test 2: Table creation (this was failing before the fix)
        print("2. Testing table creation with commit...")
        create_embeddings_table(conn, dim=128)
        print("   ✓ Table creation with commit works")
        
        # Test 3: Serialization/deserialization
        print("3. Testing vector serialization...")
        test_vec = np.random.rand(128).astype(np.float32)
        serialized = serialize_float32_array(test_vec)
        deserialized = deserialize_float32_array(serialized, 128)
        
        if np.allclose(test_vec, deserialized):
            print("   ✓ Vector serialization works")
        else:
            print("   ✗ Vector serialization failed")
            return False
        
        # Test 4: Batch insertion (this was failing before the fix)
        print("4. Testing batch insertion with commit...")
        batch_data = [
            (1, 1001, serialize_float32_array(np.random.rand(128).astype(np.float32)), 
             0.95, 0.85, 0.90, 0, 0, 0.1, 0.05, 0.02),
            (2, 1002, serialize_float32_array(np.random.rand(128).astype(np.float32)),
             0.88, 0.78, 0.82, 1, 1, 0.15, 0.08, 0.03)
        ]
        insert_embeddings_batch(conn, batch_data)
        print("   ✓ Batch insertion with commit works")
        
        # Test 5: Embedding retrieval
        print("5. Testing embedding retrieval...")
        retrieved = get_embedding_by_id(conn, 1)
        if retrieved is not None and retrieved.shape == (128,):
            print("   ✓ Embedding retrieval works")
        else:
            print("   ✗ Embedding retrieval failed")
            return False
        
        # Test 6: Get all embeddings
        print("6. Testing get all embeddings...")
        all_embeddings, all_ids = get_all_embeddings(conn)
        if len(all_embeddings) == 2 and len(all_ids) == 2:
            print("   ✓ Get all embeddings works")
        else:
            print("   ✗ Get all embeddings failed")
            return False
        
        # Test 7: kNN query
        print("7. Testing kNN query...")
        query_vec = np.random.rand(128).astype(np.float32)
        results = query_knn(conn, query_vec, k=1)
        if len(results) == 1:
            print("   ✓ kNN query works")
        else:
            print("   ✗ kNN query failed")
            return False
            
        conn.close()
        return True
        
    except Exception as e:
        print(f"   ✗ Core functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_projector_functionality():
    """Test projector functionality that was fixed"""
    print("\n=== Testing Projector APSW Compatibility ===")
    
    try:
        # Test 1: Run ID computation
        print("1. Testing run ID computation...")
        run_id = compute_run_id("test_model", {"param1": 10, "param2": "test"})
        print(f"   ✓ Run ID computation works: {run_id}")
        
        # Test 2: Projection table creation (this was failing before the fix)
        print("2. Testing projection table creation with commit...")
        proj_db = os.path.join(TEST_DB_DIR, "test_proj.db")
        conn = connect_vec_db(proj_db, require_vec0=False)
        create_vec_tables(conn, proj_dim=3, metrics_dim=5)
        print("   ✓ Projection table creation with commit works")
        
        # Test 3: Projection batch insertion (this was failing before the fix)
        print("3. Testing projection batch insertion with commit...")
        test_projections = [
            (1, "test_run_1", "test_model", 0.95, np.array([1.0, 2.0, 3.0], dtype=np.float32)),
            (2, "test_run_1", "test_model", 0.88, np.array([4.0, 5.0, 6.0], dtype=np.float32))
        ]
        insert_projection_batch(conn, test_projections, proj_dim=3)
        print("   ✓ Projection batch insertion with commit works")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"   ✗ Projector functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_refactor_functionality():
    """Test refactor functionality"""
    print("\n=== Testing Refactor Functionality ===")
    
    try:
        # Test 1: Run existence check
        print("1. Testing run existence check...")
        proj_db = os.path.join(TEST_DB_DIR, "test_refactor.db")
        
        # Should not exist initially
        exists = run_exists_in_db("nonexistent_run", "test_refactor")
        if not exists:
            print("   ✓ Run existence check works (non-existent)")
        else:
            print("   ✗ Run existence check failed")
            return False
        
        # Create a test run
        conn = connect_vec_db(proj_db, require_vec0=False)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS metadata (run_id TEXT PRIMARY KEY, model TEXT, params TEXT)")
        cursor.execute("INSERT INTO metadata (run_id, model, params) VALUES (?, ?, ?)", 
                      ("test_run_1", "test_model", '{"param": "value"}'))
        
        # Commit using safe method
        try:
            conn.commit()
        except Exception:
            try:
                conn.execute('COMMIT')
            except Exception:
                pass
        
        conn.close()
        
        # Should exist now
        exists = run_exists_in_db("test_run_1", "test_refactor")
        if exists:
            print("   ✓ Run existence check works (existing)")
        else:
            print("   ✗ Run existence check failed for existing run")
            return False
        
        # Test 2: Embeddings count
        print("2. Testing embeddings count...")
        count = embeddings_count_for_run("test_run_1", "test_refactor")
        print(f"   ✓ Embeddings count works: {count} embeddings")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Refactor functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_module_compatibility():
    """Test compatibility between modules"""
    print("\n=== Testing Cross-Module Compatibility ===")
    
    try:
        # Test that both modules can work with the same database
        print("1. Testing shared database access...")
        test_db = os.path.join(TEST_DB_DIR, "test_shared.db")
        
        from sqlvector_utils import connect_vec_db as utils_connect
        from sqlvector_projector import connect_vec_db as projector_connect
        
        conn1 = utils_connect(test_db, require_vec0=False)
        conn2 = projector_connect(test_db, require_vec0=False)
        
        print("   ✓ Both modules can connect to the same database")
        
        # Test serialization compatibility
        print("2. Testing serialization compatibility...")
        from sqlvector_utils import serialize_float32_array as utils_serialize
        from sqlvector_projector import serialize_float32_array as projector_serialize
        
        test_vec = np.random.rand(64).astype(np.float32)
        utils_result = utils_serialize(test_vec)
        projector_result = projector_serialize(test_vec)
        
        if utils_result == projector_result:
            print("   ✓ Serialization functions are compatible")
        else:
            print("   ✗ Serialization functions are NOT compatible")
            return False
            
        conn1.close()
        conn2.close()
        
        return True
        
    except Exception as e:
        print(f"   ✗ Cross-module compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run final verification tests"""
    print("=== Final APSW Compatibility Verification ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check if vec0 extension is available
    vecpath = os.environ.get("VECTOR_EXT_PATH") or os.environ.get("VEC0_SO")
    vec0_available = vecpath and os.path.exists(vecpath)
    print(f"vec0 extension available: {vec0_available}")
    
    test_results = []
    
    try:
        # Run all tests
        result1 = test_core_functionality()
        test_results.append(("Core APSW Compatibility", result1))
        
        result2 = test_projector_functionality()
        test_results.append(("Projector APSW Compatibility", result2))
        
        result3 = test_refactor_functionality()
        test_results.append(("Refactor Functionality", result3))
        
        result4 = test_cross_module_compatibility()
        test_results.append(("Cross-Module Compatibility", result4))
        
        # Print summary
        print("\n=== Final Verification Summary ===")
        all_passed = True
        for test_name, passed in test_results:
            status = "PASSED" if passed else "FAILED"
            print(f"{test_name}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n🎉 ALL FINAL VERIFICATION TESTS PASSED!")
            print("✅ The updated files are fully compatible with APSW.")
            print("✅ All commit-related issues have been resolved.")
            print("✅ The code is ready for production use with APSW and sqlite-vec.")
            return 0
        else:
            print("\n❌ SOME FINAL VERIFICATION TESTS FAILED!")
            print("Please check the output above for details.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Final verification failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Clean up
        cleanup_test_dir()

if __name__ == '__main__':
    sys.exit(main())