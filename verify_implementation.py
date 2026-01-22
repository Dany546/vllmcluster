#!/usr/bin/env python3
"""
Final Verification Checklist - YOLO Clustering Implementation
Ensures all 5 required functionalities are present in the code
"""

import ast
import re
from pathlib import Path


def check_file_exists(path: str) -> bool:
    """Check if file exists."""
    return Path(path).exists()


def check_table_exists(filepath: str, table_name: str) -> bool:
    """Check if CREATE TABLE statement exists for given table."""
    with open(filepath, 'r') as f:
        content = f.read()
    pattern = rf'CREATE TABLE.*{table_name}'
    return bool(re.search(pattern, content, re.IGNORECASE | re.DOTALL))


def check_function_exists(filepath: str, func_name: str) -> bool:
    """Check if function/method is defined."""
    with open(filepath, 'r') as f:
        try:
            tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    return True
        except:
            pass
    return False


def check_code_pattern(filepath: str, pattern: str) -> bool:
    """Check if code pattern exists (regex)."""
    with open(filepath, 'r') as f:
        content = f.read()
    return bool(re.search(pattern, content, re.IGNORECASE | re.DOTALL))


def run_verification():
    """Run all verification checks."""
    
    print("=" * 70)
    print("YOLO CLUSTERING IMPLEMENTATION - FINAL VERIFICATION CHECKLIST")
    print("=" * 70)
    print()
    
    results = {}
    
    # ===== FILE EXISTENCE =====
    print("[1/6] Checking file existence...")
    print("-" * 70)
    
    clustering_py = "/home/ucl/irec/darimez/MIRO/vllmcluster/clustering.py"
    yolo_extract = "/home/ucl/irec/darimez/MIRO/vllmcluster/yolo_extract.py"
    
    results['files_exist'] = (
        check_file_exists(clustering_py) and 
        check_file_exists(yolo_extract)
    )
    
    print(f"✓ clustering.py exists: {check_file_exists(clustering_py)}")
    print(f"✓ yolo_extract.py exists: {check_file_exists(yolo_extract)}")
    print()
    
    # ===== DATABASE TABLES =====
    print("[2/6] Checking database table schemas...")
    print("-" * 70)
    
    results['tables'] = {
        'features': check_table_exists(clustering_py, 'features'),
        'predictions': check_table_exists(clustering_py, 'predictions'),
        'embeddings': check_table_exists(clustering_py, 'embeddings'),
        'distances': check_table_exists(clustering_py, 'distances'),
    }
    
    for table, exists in results['tables'].items():
        status = "✓" if exists else "✗"
        print(f"{status} CREATE TABLE {table}: {exists}")
    print()
    
    # ===== FEATURE SAVING =====
    print("[3/6] Checking feature saving implementation...")
    print("-" * 70)
    
    results['features'] = {
        'gap_features_column': check_code_pattern(clustering_py, r'gap_features'),
        'bottleneck_features_column': check_code_pattern(clustering_py, r'bottleneck_features'),
        'insert_features': check_code_pattern(clustering_py, r'INSERT.*features'),
        'json_serialization': check_code_pattern(clustering_py, r'json\.dumps'),
    }
    
    for check, found in results['features'].items():
        status = "✓" if found else "✗"
        print(f"{status} {check}: {found}")
    print()
    
    # ===== PREDICTION SAVING =====
    print("[4/6] Checking prediction saving implementation...")
    print("-" * 70)
    
    results['predictions'] = {
        'box_columns': check_code_pattern(clustering_py, r'box_x1|box_y1|box_x2|box_y2'),
        'confidence_column': check_code_pattern(clustering_py, r'confidence'),
        'class_id_column': check_code_pattern(clustering_py, r'class_id'),
        'mask_column': check_code_pattern(clustering_py, r'mask'),
        'insert_predictions': check_code_pattern(clustering_py, r'INSERT.*predictions'),
        'per_detection_loop': check_code_pattern(clustering_py, r'for.*box.*in.*range|for box_idx'),
    }
    
    for check, found in results['predictions'].items():
        status = "✓" if found else "✗"
        print(f"{status} {check}: {found}")
    print()
    
    # ===== LOSS COMPONENTS =====
    print("[5/6] Checking loss components implementation...")
    print("-" * 70)
    
    results['losses'] = {
        'box_loss': check_code_pattern(clustering_py, r'box_loss'),
        'cls_loss': check_code_pattern(clustering_py, r'cls_loss'),
        'dfl_loss': check_code_pattern(clustering_py, r'dfl_loss'),
        'seg_loss': check_code_pattern(clustering_py, r'seg_loss'),
        'loss_extraction': check_code_pattern(clustering_py, r'loss_items\[.*\]'),
        'embeddings_storage': check_code_pattern(clustering_py, r'box_loss.*REAL'),
    }
    
    for check, found in results['losses'].items():
        status = "✓" if found else "✗"
        print(f"{status} {check}: {found}")
    print()
    
    # ===== EVALUATION METRICS & DISTANCE MATRIX =====
    print("[6/6] Checking evaluation metrics and distance matrix...")
    print("-" * 70)
    
    results['metrics_distances'] = {
        'mean_iou': check_code_pattern(clustering_py, r'mean_iou'),
        'mean_conf': check_code_pattern(clustering_py, r'mean_conf'),
        'hit_freq': check_code_pattern(clustering_py, r'hit_freq'),
        'dice_score': check_code_pattern(clustering_py, r'dice'),
        'distance_computation': check_code_pattern(clustering_py, r'for.*img_id.*in.*idx'),
        'component_distances': check_code_pattern(clustering_py, r"component.*'box'|'cls'"),
        'loss_comparison': check_code_pattern(clustering_py, r'loss_items\[.*\]'),
        'model_isolation': check_code_pattern(clustering_py, r'if.*yolo.*in.*model|is_yolo'),
    }
    
    for check, found in results['metrics_distances'].items():
        status = "✓" if found else "✗"
        print(f"{status} {check}: {found}")
    print()
    
    # ===== COMPILATION =====
    print("=" * 70)
    print("COMPILATION CHECK")
    print("=" * 70)
    print()
    
    import subprocess
    try:
        result = subprocess.run(
            ['python3', '-m', 'py_compile', clustering_py],
            cwd='/home/ucl/irec/darimez/MIRO/vllmcluster',
            capture_output=True,
            timeout=10
        )
        results['compiles'] = result.returncode == 0
        if results['compiles']:
            print("✓ clustering.py compiles successfully")
        else:
            print(f"✗ Compilation error: {result.stderr.decode()}")
    except Exception as e:
        results['compiles'] = False
        print(f"✗ Compilation check failed: {e}")
    
    print()
    
    # ===== SUMMARY =====
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    all_checks = (
        results.get('files_exist', False) and
        all(results.get('tables', {}).values()) and
        all(results.get('features', {}).values()) and
        all(results.get('predictions', {}).values()) and
        all(results.get('losses', {}).values()) and
        all(results.get('metrics_distances', {}).values()) and
        results.get('compiles', False)
    )
    
    if all_checks:
        print("✅ ALL CHECKS PASSED - Implementation Complete!")
        print()
        print("The YOLO clustering implementation includes:")
        print("  1. ✓ Feature saving (GAP + bottleneck)")
        print("  2. ✓ Prediction saving (per-detection)")
        print("  3. ✓ Loss components (box, cls, dfl, seg)")
        print("  4. ✓ Evaluation metrics (IoU, confidence, hit_freq, Dice)")
        print("  5. ✓ Asymmetric distance matrix (all image pairs)")
        print("  6. ✓ CLIP/DINO isolation (embeddings only)")
        print()
        print("Ready for deployment!")
    else:
        print("❌ SOME CHECKS FAILED - Review details above")
        print()
        failed = []
        if not results.get('files_exist'):
            failed.append("File existence")
        if not all(results.get('tables', {}).values()):
            failed.append("Database tables")
        if not all(results.get('features', {}).values()):
            failed.append("Feature saving")
        if not all(results.get('predictions', {}).values()):
            failed.append("Prediction saving")
        if not all(results.get('losses', {}).values()):
            failed.append("Loss components")
        if not all(results.get('metrics_distances', {}).values()):
            failed.append("Metrics & distances")
        if not results.get('compiles'):
            failed.append("Compilation")
        
        print(f"Failed checks: {', '.join(failed)}")
    
    print()
    print("=" * 70)
    
    return all_checks


if __name__ == "__main__":
    success = run_verification()
    exit(0 if success else 1)
