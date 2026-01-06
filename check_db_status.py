#!/usr/bin/env python3
"""
Check the status of projection database files.
Shows how many runs are completed and their embedding counts.

Usage:
    python check_db_status.py           # Just show status
    python check_db_status.py --clean   # Remove partial runs after showing status
"""
import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlvector_projector import connect_vec_db, compute_run_id

# Configure paths
PROJ_DB_DIR = "/globalscratch/ucl/irec/darimez/dino/proj/"

# Hyperparameter grids (same as project_refactor.py)
umap_params = {
    "n_neighbors": [10, 20, 40, 60],
    "min_dist": [0.02, 0.1, 0.5],
    "n_components": [2, 10, 25],
}
tsne_params = {
    "n_components": [2, 10, 25],
    "perplexity": [10, 30, 50],
    "early_exaggeration": [8, 16],
    "learning_rate": [100, 300],
    "max_iter": [1000],
}


def check_database_status(clean_partial=False):
    """Check the status of all runs in the databases"""
    parser = argparse.ArgumentParser()
    EMBEDDINGS_DIR = "/globalscratch/ucl/irec/darimez/dino/embeddings/"
    tables = [os.path.join(EMBEDDINGS_DIR, f) for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".db")]
    
    if not tables:
        print("No embedding files found!")
        return
    
    # Use the last table (same as project_refactor.py)
    table = tables[-1]
    NTABLES = len(tables)
    model_name = os.path.basename(table).split(".")[0]
    
    print(f"Model: {model_name}")
    print("=" * 60)
    
    all_runs = {}
    
    # UMAP runs
    for n_neighbors in umap_params["n_neighbors"]:
        for min_dist in umap_params["min_dist"]:
            for n_components in umap_params["n_components"]:
                hyperparams = {"n_neighbors": n_neighbors, "min_dist": min_dist, "n_components": n_components}
                run_id = compute_run_id(model_name, hyperparams)
                all_runs[("umap", run_id)] = hyperparams

    Numap = len(all_runs)
    
    # t-SNE runs
    for n_components in tsne_params["n_components"]:
        for perplexity in tsne_params["perplexity"]:
            for early_exaggeration in tsne_params["early_exaggeration"]:
                for learning_rate in tsne_params["learning_rate"]:
                    hyperparams = {
                        "n_components": n_components,
                        "perplexity": perplexity,
                        "early_exaggeration": early_exaggeration,
                        "learning_rate": learning_rate,
                        "max_iter": 1000
                    }
                    run_id = compute_run_id(model_name, hyperparams)
                    all_runs[("tsne", run_id)] = hyperparams
    
    Numap = int(Numap*NTABLES)
    Ntsne = int(len(all_runs)*NTABLES) - Numap
    # Check each database
    for algo in ["umap", "tsne"]:
        db_file = os.path.join(PROJ_DB_DIR, f"{algo}.db")
        print(f"\n{'='*60}")
        print(f"Database: {algo.upper()} ({os.path.basename(db_file)})")
        print("=" * 60)
        
        if not os.path.exists(db_file):
            print("Database file does not exist yet.")
            continue
        
        try:
            conn = connect_vec_db(db_file)
            cur = conn.cursor()
            
            # Get all runs from metadata
            cur.execute("SELECT run_id, model, params FROM metadata")
            metadata_rows = cur.fetchall()
            
            if not metadata_rows:
                print("No runs found in database.")
                conn.close()
                continue
            
            # Get counts for each run
            cur.execute("SELECT run_id, COUNT(*) as cnt FROM vec_projections GROUP BY run_id")
            count_rows = {row[0]: row[1] for row in cur.fetchall()}
            
            # Also check fallback table if exists
            try:
                cur.execute("SELECT run_id, COUNT(*) as cnt FROM embeddings GROUP BY run_id")
                for row in cur.fetchall():
                    if row[0] in count_rows:
                        count_rows[row[0]] = max(count_rows[row[0]], row[1])
                    else:
                        count_rows[row[0]] = row[1]
            except Exception:
                pass
            
            total_runs = len(metadata_rows)
            complete_runs = 0
            partial_runs = 0
            
            print(f"\n{'Run ID':<40} {'Status'}")
            print("-" * 70)
            
            for run_id, db_model, params_str in metadata_rows:
                count = count_rows.get(run_id, 0)
                status = "✓ COMPLETE" if count >= 5000 else ("⚠ PARTIAL" if count > 0 else "✗ EMPTY")
                
                if count >= 5000:
                    complete_runs += 1
                elif count > 0:
                    partial_runs += 1
                
                # Truncate run_id for display
                display_id = run_id[:38] + ".." if len(run_id) > 40 else run_id
                print(f"{display_id:<40} {status}")
            
            print("-" * 70)
            print(f"Total runs in DB: {total_runs}")
            print(f"Complete (≥5000): {complete_runs}")
            print(f"Partial (1-4999): {partial_runs}")
            print(f"Missing: {Numap - total_runs if algo == 'umap' else Ntsne - total_runs}")
            
            # Clean partial runs if requested
            if clean_partial and partial_runs > 0:
                print(f"\nRemoving {partial_runs} partial runs...")
                partial_run_ids = [run_id for run_id, db_model, params_str in metadata_rows 
                                   if count_rows.get(run_id, 0) > 0 and count_rows.get(run_id, 0) < 5000]
                
                for run_id in partial_run_ids:
                    try:
                        cur.execute("DELETE FROM vec_projections WHERE run_id = ?", (run_id,))
                        cur.execute("DELETE FROM metadata WHERE run_id = ?", (run_id,))
                        print(f"  Removed: {run_id}")
                    except Exception as e:
                        print(f"  Failed to remove {run_id}: {e}")
                
                try:
                    conn.commit()
                except Exception:
                    try:
                        conn.execute('COMMIT')
                    except Exception:
                        pass
                print(f"Removed {len(partial_run_ids)} partial runs from {algo.upper()} database.")
            
            conn.close()
            
        except Exception as e:
            print(f"Error reading database: {e}")
    
    # Summary of all expected runs
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Expected total runs: {int(len(all_runs)*NTABLES)} ({Numap} UMAP + {Ntsne} t-SNE)")
    print(f"Database location: {PROJ_DB_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check database status')
    parser.add_argument('--clean', action='store_true', help='Remove partial runs after showing status')
    args = parser.parse_args()
    
    check_database_status(clean_partial=args.clean)
