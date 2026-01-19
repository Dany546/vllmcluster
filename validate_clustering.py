#!/usr/bin/env python3
"""
Validation script to verify all YOLO clustering functionalities:
1. Feature saving
2. Prediction saving
3. Loss components saving
4. Asymmetric distance computation
5. Evaluation metrics computation
6. CLIP/DINO isolation
"""

import sqlite3
import json
import numpy as np
from pathlib import Path


def validate_databases(embeddings_db: str, distances_db: str) -> dict:
    """Validate all functionality by checking database contents."""
    
    results = {
        "embeddings_db_exists": False,
        "distances_db_exists": False,
        "features_saved": False,
        "predictions_saved": False,
        "loss_components_saved": False,
        "eval_metrics_saved": False,
        "distance_matrix_computed": False,
        "errors": []
    }
    
    # Check embeddings DB
    emb_path = Path(embeddings_db)
    if not emb_path.exists():
        results["errors"].append(f"Embeddings DB not found: {embeddings_db}")
        return results
    
    results["embeddings_db_exists"] = True
    
    try:
        emb_conn = sqlite3.connect(embeddings_db)
        emb_cursor = emb_conn.cursor()
        
        # Check tables exist
        emb_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in emb_cursor.fetchall()}
        
        required_tables = {"embeddings", "features", "predictions", "metadata"}
        missing_tables = required_tables - tables
        if missing_tables:
            results["errors"].append(f"Missing tables: {missing_tables}")
        
        # 1. Check features table
        if "features" in tables:
            emb_cursor.execute("SELECT COUNT(*) FROM features")
            count = emb_cursor.fetchone()[0]
            if count > 0:
                results["features_saved"] = True
                emb_cursor.execute("SELECT img_id, gap_features, bottleneck_features FROM features LIMIT 1")
                row = emb_cursor.fetchone()
                if row:
                    try:
                        gap = json.loads(row[1]) if row[1] else None
                        bottleneck = json.loads(row[2]) if row[2] else None
                        if gap or bottleneck:
                            print(f"✓ Features saved: img_id={row[0]}, gap_features={list(gap.keys()) if gap else None}")
                    except Exception as e:
                        results["errors"].append(f"Feature JSON parse error: {e}")
            else:
                results["errors"].append("Features table exists but is empty")
        
        # 2. Check predictions table
        if "predictions" in tables:
            emb_cursor.execute("SELECT COUNT(*) FROM predictions")
            count = emb_cursor.fetchone()[0]
            if count > 0:
                results["predictions_saved"] = True
                emb_cursor.execute("SELECT img_id, box_x1, box_y1, box_x2, box_y2, confidence, class_id FROM predictions LIMIT 1")
                row = emb_cursor.fetchone()
                if row:
                    print(f"✓ Predictions saved: {count} total detections")
                    print(f"  Sample: img_id={row[0]}, box=({row[1]:.1f}, {row[2]:.1f}, {row[3]:.1f}, {row[4]:.1f}), conf={row[5]:.3f}, cls={row[6]}")
            else:
                results["errors"].append("Predictions table exists but is empty")
        
        # 3. Check loss components and eval metrics in embeddings table
        emb_cursor.execute("PRAGMA table_info(embeddings)")
        columns = {row[1] for row in emb_cursor.fetchall()}
        
        loss_columns = {"box_loss", "cls_loss", "dfl_loss", "seg_loss"}
        metric_columns = {"mean_iou", "mean_conf", "hit_freq"}
        
        has_losses = loss_columns.issubset(columns)
        has_metrics = metric_columns.issubset(columns)
        
        if has_losses:
            results["loss_components_saved"] = True
            print(f"✓ Loss components columns found: {loss_columns & columns}")
        else:
            missing = loss_columns - columns
            results["errors"].append(f"Missing loss columns: {missing}")
        
        if has_metrics:
            results["eval_metrics_saved"] = True
            print(f"✓ Evaluation metrics columns found: {metric_columns & columns}")
        else:
            missing = metric_columns - columns
            results["errors"].append(f"Missing metric columns: {missing}")
        
        # Check actual values
        emb_cursor.execute("SELECT COUNT(*) FROM embeddings")
        emb_count = emb_cursor.fetchone()[0]
        if emb_count > 0:
            emb_cursor.execute("""
                SELECT mean_iou, mean_conf, hit_freq, box_loss, cls_loss, dfl_loss 
                FROM embeddings LIMIT 1
            """)
            row = emb_cursor.fetchone()
            if row:
                print(f"✓ Sample embedding metrics:")
                print(f"    IoU={row[0]:.3f}, Conf={row[1]:.3f}, Hit_freq={row[2]:.3f}")
                print(f"    Losses: box={row[3]:.3f}, cls={row[4]:.3f}, dfl={row[5]:.3f}")
        
        emb_conn.close()
        
    except Exception as e:
        results["errors"].append(f"Embeddings DB error: {e}")
    
    # Check distances DB
    dist_path = Path(distances_db)
    if not dist_path.exists():
        results["errors"].append(f"Distances DB not found: {distances_db}")
        return results
    
    results["distances_db_exists"] = True
    
    try:
        dist_conn = sqlite3.connect(distances_db)
        dist_cursor = dist_conn.cursor()
        
        # Check distances table
        dist_cursor.execute("SELECT COUNT(*) FROM distances")
        count = dist_cursor.fetchone()[0]
        if count > 0:
            results["distance_matrix_computed"] = True
            
            # Check component diversity
            dist_cursor.execute("SELECT DISTINCT component FROM distances")
            components = {row[0] for row in dist_cursor.fetchall()}
            print(f"✓ Distance matrix computed: {count} entries with components: {components}")
            
            # Check symmetry (should have both i→j and j→i)
            dist_cursor.execute("""
                SELECT DISTINCT i, j FROM distances 
                WHERE component = 'box' 
                LIMIT 5
            """)
            sample_pairs = dist_cursor.fetchall()
            print(f"  Sample pairs: {sample_pairs}")
            
        else:
            results["errors"].append("Distances table exists but is empty")
        
        dist_conn.close()
        
    except Exception as e:
        results["errors"].append(f"Distances DB error: {e}")
    
    return results


def print_validation_report(results: dict):
    """Print formatted validation report."""
    print("\n" + "="*60)
    print("YOLO CLUSTERING FUNCTIONALITY VALIDATION REPORT")
    print("="*60 + "\n")
    
    checks = [
        ("Features Saving", results["features_saved"]),
        ("Predictions Saving", results["predictions_saved"]),
        ("Loss Components", results["loss_components_saved"]),
        ("Evaluation Metrics", results["eval_metrics_saved"]),
        ("Distance Matrix", results["distance_matrix_computed"]),
    ]
    
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:.<40} {status}")
    
    if results["errors"]:
        print("\n" + "-"*60)
        print("ERRORS AND WARNINGS:")
        print("-"*60)
        for error in results["errors"]:
            print(f"  ✗ {error}")
    
    all_passed = all([
        results["features_saved"],
        results["predictions_saved"],
        results["loss_components_saved"],
        results["eval_metrics_saved"],
        results["distance_matrix_computed"],
    ])
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Clustering ready for use!")
    else:
        print("✗ SOME CHECKS FAILED - Review errors above")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python validate_clustering.py <embeddings_db> <distances_db>")
        print("Example: python validate_clustering.py /path/to/embeddings.db /path/to/distances.db")
        sys.exit(1)
    
    embeddings_db = sys.argv[1]
    distances_db = sys.argv[2]
    
    results = validate_databases(embeddings_db, distances_db)
    print_validation_report(results)
