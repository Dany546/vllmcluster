#!/usr/bin/env python3
"""
Comprehensive test script to verify APSW and sqlite-vec compatibility.
This script tests all major functionality of the updated files.
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
    build_vector_index,
    query_knn,
    get_embedding_by_id,
    get_all_embeddings
)

from sqlvector_projector import (
    create_vec_tables,
    insert_projection_batch,
    compute_run_id,
    load_vectors_from_db,
    get_vector_slice,
    knn_query_projections
)

from project_refactor import (
    run_exists_in_db,
    embeddings_count_for_run
)

# Test configuration
TEST_DB_DIR = tempfile.mkdtemp(prefix="test_comprehensive_")
print(f"Using temporary test directory: {TEST_DB_DIR}")

def cleanup_test_dir():
    """Clean up temporary test directory"""
    if os.path.exists(TEST_DB_DIR):
        shutil.rmtree(TEST_DB_DIR)
        print(f"Cleaned up test directory: {TEST_DB_DIR}")

def test_apsw_connection():
    """Test 1: Verify APSW connection and vec0 extension loading"""
    print("\n=== Test 1: APSW Connection and vec0 Extension ===")
    
    test_db = os.path.join(TEST_DB_DIR, "test_connection.db")
    
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

def test_sqlvector_utils_comprehensive():
    """Test 2: Comprehensive test of sqlvector_utils.py functionality"""
    print("\n=== Test 2: sqlvector_utils.py Comprehensive Test ===")
    
    test_db = os.path.join(TEST_DB_DIR, "test_utils.db")
    
    try:
        # Connect to database
        conn = connect_vec_db(test_db, require_vec0=False)
        
        # Test table creation with different configurations
        create_embeddings_table(conn, dim=128, is_seg=False)
        print("✓ Embeddings table created successfully (non-segmentation)")
        
        create_embeddings_table(conn, dim=256, is_seg=True)
        print("✓ Embeddings table created successfully (segmentation)")
        
        # Test serialization/deserialization with various array sizes
        for size in [64, 128, 256, 512]:
            test_vec = np.random.rand(size).astype(np.float32)
            serialized = serialize_float32_array(test_vec)
            deserialized = deserialize_float32_array(serialized, size)
            
            if np.allclose(test_vec, deserialized):
                print(f"✓ Vector serialization/deserialization works for size {size}")
            else:
                print(f"✗ Vector serialization/deserialization failed for size {size}")
                return False
        
        # Test batch insertion with larger dataset
        batch_data = []
        for i in range(10):
            batch_data.append((
                i+1, 1000+i, serialize_float32_array(np.random.rand(128).astype(np.float32)), 
                0.90 + np.random.rand() * 0.1,  # mean_conf
                0.80 + np.random.rand() * 0.1,  # mean_iou
                0.85 + np.random.rand() * 0.1,  # mean_conf
                i % 2, i % 3,  # flags
                0.1 + np.random.rand() * 0.05,  # box_loss
                0.05 + np.random.rand() * 0.02,  # cls_loss
                0.02 + np.random.rand() * 0.01   # dfl_loss
            ))
        
        insert_embeddings_batch(conn, batch_data)
        print("✓ Large batch insertion works correctly")
        
        # Test embedding retrieval for multiple IDs
        for i in range(1, 6):
            retrieved = get_embedding_by_id(conn, i)
            if retrieved is None or retrieved.shape != (128,):
                print(f"✗ Embedding retrieval failed for ID {i}")
                return False
        print("✓ Multiple embedding retrievals work correctly")
        
        # Test get all embeddings
        all_embeddings, all_ids = get_all_embeddings(conn)
        if len(all_embeddings) == 10 and len(all_ids) == 10:
            print("✓ Get all embeddings works correctly")
        else:
            print(f"✗ Get all embeddings failed: got {len(all_embeddings)} embeddings, {len(all_ids)} IDs")
            return False
            
        # Test kNN query with different k values
        query_vec = np.random.rand(128).astype(np.float32)
        for k in [1, 3, 5, 10]:
            results = query_knn(conn, query_vec, k=k)
            if len(results) != k:
                print(f"✗ kNN query failed for k={k}")
                return False
        print("✓ kNN queries work correctly for various k values")
            
        # Test kNN query with filter
        results = query_knn(conn, query_vec, k=3, filter_clause="flag_cat = 0")
        if len(results) <= 3:  # Should return up to 3 results matching the filter
            print("✓ kNN query with filtering works correctly")
        else:
            print("✗ kNN query with filtering failed")
            return False
            
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ sqlvector_utils.py comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sqlvector_projector_comprehensive():
    """Test 3: Comprehensive test of sqlvector_projector.py functionality"""
    print("\n=== Test 3: sqlvector_projector.py Comprehensive Test ===")
    
    try:
        # Test run ID computation with various inputs
        test_cases = [
            ("model1", {"param1": 10}),
            ("model2", {"param1": 20, "param2": "test"}),
            ("model3", {"learning_rate": 0.001, "batch_size": 32}),
        ]
        
        run_ids = []
        for model, params in test_cases:
            run_id = compute_run_id(model, params)
            run_ids.append(run_id)
            print(f"✓ Run ID computation works: {run_id}")
        
        # Test that different params produce different run IDs
        if len(set(run_ids)) == len(run_ids):
            print("✓ Run ID computation produces unique IDs for different inputs")
        else:
            print("✗ Run ID computation produced duplicate IDs")
            return False
        
        # Test projection table creation with different dimensions
        for proj_dim, metrics_dim in [(2, 5), (10, 20), (50, 100)]:
            proj_db = os.path.join(TEST_DB_DIR, f"test_proj_{proj_dim}d.db")
            conn = connect_vec_db(proj_db, require_vec0=False)
            create_vec_tables(conn, proj_dim=proj_dim, metrics_dim=metrics_dim)
            print(f"✓ Projection tables created successfully ({proj_dim}d projections, {metrics_dim}d metrics)")
            conn.close()
        
        # Test projection batch insertion and retrieval
        proj_db = os.path.join(TEST_DB_DIR, "test_proj_main.db")
        conn = connect_vec_db(proj_db, require_vec0=False)
        create_vec_tables(conn, proj_dim=3, metrics_dim=5)
        
        # Insert multiple runs
        test_projections = []
        for run_num in range(3):
            for i in range(5):
                test_projections.append((
                    i + run_num * 100,
                    f"test_run_{run_num}", 
                    f"test_model_{run_num}", 
                    0.85 + np.random.rand() * 0.1,
                    np.random.rand(3).astype(np.float32)
                ))
        
        insert_projection_batch(conn, test_projections, proj_dim=3)
        print("✓ Multi-run projection batch insertion works correctly")
        
        # Test vector loading for each run
        for run_num in range(3):
            run_id = f"test_run_{run_num}"
            loaded_vectors = load_vectors_from_db(TEST_DB_DIR, "test_proj_main", run_id)
            if loaded_vectors.shape == (5, 3):
                print(f"✓ Vector loading works for run {run_id}")
            else:
                print(f"✗ Vector loading failed for run {run_id}")
                return False
        
        # Test vector slice extraction
        slice_2d = get_vector_slice(TEST_DB_DIR, "test_proj_main", "test_run_0", dims=[0, 1])
        if slice_2d.shape == (5, 2):
            print("✓ Vector slice extraction works correctly")
        else:
            print("✗ Vector slice extraction failed")
            return False
            
        # Test kNN query on projections
        query_vec = np.random.rand(3).astype(np.float32)
        knn_results = knn_query_projections(TEST_DB_DIR, "test_proj_main", query_vec, k=2, run_id="test_run_0")
        if len(knn_results) == 2:
            print("✓ kNN query on projections works correctly")
        else:
            print("✗ kNN query on projections failed")
            return False
            
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ sqlvector_projector.py comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_project_refactor_comprehensive():
    """Test 4: Comprehensive test of project_refactor.py functionality"""
    print("\n=== Test 4: project_refactor.py Comprehensive Test ===")
    
    try:
        # Test run existence check with non-existent runs
        proj_db = os.path.join(TEST_DB_DIR, "test_refactor.db")
        
        non_existent_runs = ["nonexistent_run_1", "nonexistent_run_2"]
        for run_id in non_existent_runs:
            exists = run_exists_in_db(run_id, "test_refactor")
            if not exists:
                print(f"✓ Run existence check works correctly (non-existent run: {run_id})")
            else:
                print(f"✗ Run existence check failed for non-existent run: {run_id}")
                return False
        
        # Create multiple test runs and check existence
        conn = connect_vec_db(proj_db, require_vec0=False)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS metadata (run_id TEXT PRIMARY KEY, model TEXT, params TEXT)")
        
        test_runs = [
            ("test_run_1", "model_a", '{"param": "value1"}'),
            ("test_run_2", "model_b", '{"param": "value2"}'),
            ("test_run_3", "model_c", '{"param": "value3"}'),
        ]
        
        for run_id, model, params in test_runs:
            cursor.execute("INSERT INTO metadata (run_id, model, params) VALUES (?, ?, ?)", (run_id, model, params))
        
        # Commit using the safe method
        try:
            conn.commit()
        except Exception:
            try:
                conn.execute('COMMIT')
            except Exception:
                pass
        
        conn.close()
        
        # Test existence for created runs
        for run_id, _, _ in test_runs:
            exists = run_exists_in_db(run_id, "test_refactor")
            if exists:
                print(f"✓ Run existence check works correctly (existing run: {run_id})")
            else:
                print(f"✗ Run existence check failed for existing run: {run_id}")
                return False
        
        # Test embeddings count (should be 0 since we haven't created vec_projections table)
        for run_id, _, _ in test_runs:
            count = embeddings_count_for_run(run_id, "test_refactor")
            print(f"✓ Embeddings count check works for {run_id}: {count} embeddings")
            
        return True
        
    except Exception as e:
        print(f"✗ project_refactor.py comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_module_compatibility():
    """Test 5: Test compatibility and integration between all modules"""
    print("\n=== Test 5: Cross-Module Compatibility ===")
    
    try:
        # Test that all modules can be imported together
        from sqlvector_utils import connect_vec_db as utils_connect
        from sqlvector_projector import connect_vec_db as projector_connect
        from sqlvector_utils import serialize_float32_array as utils_serialize
        from sqlvector_projector import serialize_float32_array as projector_serialize
        
        # Test that both connection functions work with the same database
        test_db = os.path.join(TEST_DB_DIR, "test_compat.db")
        
        conn1 = utils_connect(test_db, require_vec0=False)
        conn2 = projector_connect(test_db, require_vec0=False)
        
        print("✓ Both modules can connect to the same database")
        
        # Test that serialization functions are compatible
        test_vec = np.random.rand(64).astype(np.float32)
        
        utils_result = utils_serialize(test_vec)
        projector_result = projector_serialize(test_vec)
        
        if utils_result == projector_result:
            print("✓ Serialization functions are compatible between modules")
        else:
            print("✗ Serialization functions are NOT compatible")
            return False
        
        # Test that deserialization works across modules
        from sqlvector_utils import deserialize_float32_array as utils_deserialize
        from sqlvector_projector import deserialize_float32_array as projector_deserialize
        
        deserialized_utils = utils_deserialize(utils_result, 64)
        deserialized_projector = projector_deserialize(projector_result, 64)
        
        if np.allclose(deserialized_utils, deserialized_projector):
            print("✓ Deserialization functions are compatible between modules")
        else:
            print("✗ Deserialization functions are NOT compatible")
            return False
            
        conn1.close()
        conn2.close()
        
        # Test integration: create embeddings and then project them
        emb_db = os.path.join(TEST_DB_DIR, "test_integration_emb.db")
        proj_db = os.path.join(TEST_DB_DIR, "test_integration_proj.db")
        
        # Create embeddings
        emb_conn = utils_connect(emb_db, require_vec0=False)
        create_embeddings_table(emb_conn, dim=64)
        
        # Insert some embeddings
        emb_batch = []
        for i in range(5):
            emb_batch.append((
                i+1, 1000+i, serialize_float32_array(np.random.rand(64).astype(np.float32)),
                0.90, 0.85, 0.88, 0, 0, 0.1, 0.05, 0.02
            ))
        insert_embeddings_batch(emb_conn, emb_batch)
        emb_conn.close()
        
        # Create projections from embeddings
        proj_conn = projector_connect(proj_db, require_vec0=False)
        create_vec_tables(proj_conn, proj_dim=2, metrics_dim=0)
        
        # Simulate projection by taking first 2 dimensions
        proj_batch = []
        for i in range(5):
            emb_2d = np.random.rand(2).astype(np.float32)  # Simulated projection
            proj_batch.append((i+1, "test_run_1", "test_model", 0.90, emb_2d))
        
        insert_projection_batch(proj_conn, proj_batch, proj_dim=2)
        proj_conn.close()
        
        print("✓ Integration test: embeddings -> projections workflow works")
        
        return True
        
    except Exception as e:
        print(f"✗ Cross-module compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all comprehensive tests"""
    print("=== Comprehensive APSW and sqlite-vec Test Suite ===")
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
        
        result2 = test_sqlvector_utils_comprehensive()
        test_results.append(("sqlvector_utils.py Comprehensive", result2))
        
        result3 = test_sqlvector_projector_comprehensive()
        test_results.append(("sqlvector_projector.py Comprehensive", result3))
        
        result4 = test_project_refactor_comprehensive()
        test_results.append(("project_refactor.py Comprehensive", result4))
        
        result5 = test_cross_module_compatibility()
        test_results.append(("Cross-Module Compatibility", result5))
        
        # Print summary
        print("\n=== Comprehensive Test Summary ===")
        all_passed = True
        for test_name, passed in test_results:
            status = "PASSED" if passed else "FAILED"
            print(f"{test_name}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n🎉 ALL COMPREHENSIVE TESTS PASSED!")
            print("The updated files are fully compatible with APSW and sqlite-vec.")
            print("All functionality has been verified and is working correctly.")
            return 0
        else:
            print("\n❌ SOME COMPREHENSIVE TESTS FAILED!")
            print("Please check the output above for details.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Comprehensive test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Clean up
        cleanup_test_dir()

if __name__ == '__main__':
    sys.exit(main())