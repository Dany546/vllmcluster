"""
Diagnostic script to identify the bug in get_tasks()
Run this to see what's happening with the task completion logic.
"""
import sqlite3
import os
import json
import hashlib

# Simulate the compute_run_id function
def compute_run_id(model: str, hyperparams: dict) -> str:
    hp_items = sorted(hyperparams.items())
    hp_str = json.dumps(hp_items)
    run_hash = hashlib.sha1(f"{model}:{hp_str}".encode("utf-8")).hexdigest()[:8]
    return f"{model}_{run_hash}"


def diagnose_get_tasks_bug():
    """
    This script simulates what get_tasks() does and logs the values
    to identify why tasks are being skipped.
    """
    db_path = "/globalscratch/ucl/irec/darimez/dino/proj/"
    model_name = "test_model"

    # Sample hyperparams (like in the actual code)
    sample_params = {
        "n_neighbors": 10,
        "min_dist": 0.02,
        "n_components": 2,
    }

    run_id = compute_run_id(model_name, sample_params)
    print(f"Testing run_id: {run_id}")

    # Connect to database
    conn = sqlite3.connect(os.path.join(db_path, "umap.db"))
    cur = conn.cursor()

    # Step 1: Check if run_id exists in metadata
    cur.execute("SELECT 1 FROM metadata WHERE run_id=?", (run_id,))
    metadata_exists = cur.fetchone()
    print(f"\n1. Metadata check for {run_id}:")
    print(f"   fetchone() result: {metadata_exists}")
    print(f"   exists is None: {metadata_exists is None}")

    if metadata_exists is None:
        print("   -> Task would be ADDED (no metadata entry)")
    else:
        # Step 2: Check embeddings count
        cur.execute("SELECT COUNT(*) FROM embeddings WHERE run_id=?", (run_id,))
        count = cur.fetchone()
        print(f"\n2. Embeddings count for {run_id}:")
        print(f"   fetchone() result: {count}")
        print(f"   count[0]: {count[0] if count else 'N/A'}")
        print(f"   count[0] == 5000: {count[0] == 5000 if count else False}")

        # THE BUG: This is the problematic logic
        exists = count[0] == 5000 if count else False
        print(f"\n3. Bug analysis:")
        print(f"   exists (as 'complete') = {exists}")
        print(f"   not exists = {not exists}")
        print(f"   Task would be {'ADDED' if not exists else 'SKIPPED'}")

        # Check actual counts in the database
        print(f"\n4. Database state analysis:")
        cur.execute("SELECT run_id, COUNT(*) as cnt FROM embeddings GROUP BY run_id LIMIT 5")
        for row in cur.fetchall():
            print(f"   run_id: {row[0]}, count: {row[1]}")

        # Check if any run_id has exactly 5000
        cur.execute("SELECT run_id FROM embeddings GROUP BY run_id HAVING COUNT(*) = 5000")
        exact_5000 = cur.fetchall()
        print(f"\n   Number of run_ids with exactly 5000 embeddings: {len(exact_5000)}")

    conn.close()


if __name__ == "__main__":
    diagnose_get_tasks_bug()
