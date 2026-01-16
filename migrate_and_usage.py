"""
Migration helper and usage examples for SQL vector refactor.
"""

import numpy as np
import argparse
from pathlib import Path
from sqlvector_utils import (
    connect_vec_db,
    query_knn,
    get_embedding_by_id,
    get_all_embeddings,
)


def example_knn_evaluation(db_path: str, k: int = 10):
    """
    Example: Use kNN queries for evaluation instead of precomputed distances.
    
    This replaces the O(N²) distance matrix computation.
    """
    print(f"\n=== kNN Evaluation Example (k={k}) ===")
    
    conn = connect_vec_db(db_path)
    
    # Get all embeddings
    embeddings, ids = get_all_embeddings(conn)
    print(f"Loaded {len(ids)} embeddings")
    
    # Example: For each embedding, find its k nearest neighbors
    # This is what you'd do during training/evaluation
    
    num_samples = min(5, len(ids))  # Demo with first 5
    
    for i in range(num_samples):
        query_id = ids[i]
        query_vec = embeddings[i]
        
        # Find k+1 neighbors (including self)
        neighbors = query_knn(conn, query_vec, k=k+1, return_distances=True)
        
        # Skip self (first result)
        neighbors = neighbors[1:]
        
        print(f"\nQuery ID {query_id}:")
        print(f"  Top {k} neighbors:")
        for j, neighbor in enumerate(neighbors[:5], 1):  # Show top 5
            n_id, n_img_id, mean_conf, box_loss, cls_loss, dfl_loss, cat, supercat, dist = neighbor
            print(f"    {j}. ID={n_id}, dist={dist:.4f}, cat={cat}, loss={box_loss:.3f}")
    
    conn.close()


def example_weighted_knn(db_path: str, k: int = 10, loss_weight: float = 0.5):
    """
    Example: Weighted kNN using loss values.
    
    This demonstrates how to weight neighbors by both distance and loss.
    """
    print(f"\n=== Weighted kNN Example (k={k}, loss_weight={loss_weight}) ===")
    
    conn = connect_vec_db(db_path)
    
    # Get a query embedding
    embeddings, ids = get_all_embeddings(conn)
    
    if len(ids) == 0:
        print("No embeddings in database")
        conn.close()
        return
    
    query_vec = embeddings[0]
    
    # Get neighbors
    neighbors = query_knn(conn, query_vec, k=k*2)  # Get more, then filter
    
    # Compute weighted scores
    weighted_neighbors = []
    for neighbor in neighbors:
        n_id, n_img_id, mean_conf, box_loss, cls_loss, dfl_loss, cat, supercat, dist = neighbor
        
        # Normalize distance and loss to [0, 1]
        # This is simplified - in practice, you'd use running statistics
        norm_dist = dist
        norm_loss = (box_loss + cls_loss + dfl_loss) / 3.0
        
        # Weighted score (lower is better)
        weighted_score = (1 - loss_weight) * norm_dist + loss_weight * norm_loss
        
        weighted_neighbors.append((weighted_score, neighbor))
    
    # Sort by weighted score
    weighted_neighbors.sort(key=lambda x: x[0])
    
    print(f"\nTop {k} weighted neighbors:")
    for i, (score, neighbor) in enumerate(weighted_neighbors[:k], 1):
        n_id, n_img_id, mean_conf, box_loss, cls_loss, dfl_loss, cat, supercat, dist = neighbor
        print(f"  {i}. ID={n_id}, score={score:.4f}, dist={dist:.4f}, loss={box_loss:.3f}")
    
    conn.close()


def example_filtered_knn(db_path: str, k: int = 10, category: int = 1):
    """
    Example: kNN with category filtering.
    
    This shows how to find neighbors within a specific category.
    """
    print(f"\n=== Filtered kNN Example (category={category}) ===")
    
    conn = connect_vec_db(db_path)
    
    # Get a query embedding
    embeddings, ids = get_all_embeddings(conn)
    
    if len(ids) == 0:
        print("No embeddings in database")
        conn.close()
        return
    
    query_vec = embeddings[0]
    
    # Query with category filter
    neighbors = query_knn(
        conn,
        query_vec,
        k=k,
        filter_clause="flag_cat = ?",
        filter_params=(category,)
    )
    
    print(f"\nTop {k} neighbors in category {category}:")
    for i, neighbor in enumerate(neighbors, 1):
        n_id, n_img_id, mean_conf, box_loss, cls_loss, dfl_loss, cat, supercat, dist = neighbor
        print(f"  {i}. ID={n_id}, dist={dist:.4f}, cat={cat}, conf={mean_conf:.3f}")
    
    conn.close()


def analyze_database(db_path: str):
    """Analyze database contents and statistics."""
    print(f"\n=== Database Analysis: {db_path} ===")
    
    conn = connect_vec_db(db_path, load_extension=False)
    cur = conn.cursor()
    
    # Check if tables exist
    tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print(f"\nTables: {[t[0] for t in tables]}")
    
    # Get metadata
    try:
        metadata = cur.execute("SELECT key, value FROM metadata").fetchall()
        print(f"\nMetadata:")
        for key, value in metadata:
            print(f"  {key}: {value}")
    except:
        print("  No metadata table")
    
    # Get embeddings count
    try:
        count = cur.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        print(f"\nTotal embeddings: {count}")
        
        # Category distribution
        cat_dist = cur.execute("""
            SELECT flag_cat, COUNT(*) as cnt
            FROM embeddings
            GROUP BY flag_cat
            ORDER BY cnt DESC
            LIMIT 10
        """).fetchall()
        print(f"\nTop 10 categories:")
        for cat, cnt in cat_dist:
            print(f"  Category {cat}: {cnt} samples")
        
        # Loss statistics
        stats = cur.execute("""
            SELECT
                AVG(box_loss) as avg_box,
                AVG(cls_loss) as avg_cls,
                AVG(dfl_loss) as avg_dfl,
                AVG(mean_iou) as avg_iou,
                AVG(mean_conf) as avg_conf
            FROM embeddings
        """).fetchone()
        print(f"\nLoss statistics:")
        print(f"  Avg box loss: {stats[0]:.4f}")
        print(f"  Avg cls loss: {stats[1]:.4f}")
        print(f"  Avg dfl loss: {stats[2]:.4f}")
        print(f"  Avg IoU: {stats[3]:.4f}")
        print(f"  Avg confidence: {stats[4]:.4f}")
        
    except Exception as e:
        print(f"  Error analyzing embeddings: {e}")
    
    conn.close()


def compare_with_baseline(db_path: str, k: int = 10):
    """
    Compare vector index kNN with exact computation.
    
    This helps verify the index is working correctly.
    """
    print(f"\n=== Baseline Comparison (k={k}) ===")
    
    conn = connect_vec_db(db_path)
    
    # Load all embeddings
    embeddings, ids = get_all_embeddings(conn)
    
    if len(ids) < k + 1:
        print(f"Not enough embeddings (need at least {k+1}, have {len(ids)})")
        conn.close()
        return
    
    # Pick a query
    query_idx = 0
    query_vec = embeddings[query_idx]
    
    # Index-based kNN
    index_neighbors = query_knn(conn, query_vec, k=k+1)
    index_ids = [n[0] for n in index_neighbors[1:]]  # Skip self
    
    # Exact kNN (brute force)
    distances = np.linalg.norm(embeddings - query_vec, axis=1)
    exact_indices = np.argsort(distances)[1:k+1]  # Skip self
    exact_ids = [ids[i] for i in exact_indices]
    
    # Compare
    overlap = len(set(index_ids) & set(exact_ids))
    recall = overlap / k
    
    print(f"\nQuery ID: {ids[query_idx]}")
    print(f"Index kNN IDs: {index_ids[:5]}...")
    print(f"Exact kNN IDs: {exact_ids[:5]}...")
    print(f"Overlap: {overlap}/{k} ({recall:.1%})")
    
    if recall == 1.0:
        print("✓ Perfect recall - index matches exact computation")
    elif recall >= 0.9:
        print("✓ Good recall - index is working well")
    else:
        print("⚠ Low recall - check index configuration")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="SQL vector database utilities")
    parser.add_argument("db_path", help="Path to SQLite database")
    parser.add_argument("--analyze", action="store_true", help="Analyze database")
    parser.add_argument("--example-knn", action="store_true", help="Run kNN example")
    parser.add_argument("--example-weighted", action="store_true", help="Run weighted kNN example")
    parser.add_argument("--example-filtered", action="store_true", help="Run filtered kNN example")
    parser.add_argument("--compare-baseline", action="store_true", help="Compare with exact computation")
    parser.add_argument("-k", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--category", type=int, default=1, help="Category for filtered search")
    parser.add_argument("--loss-weight", type=float, default=0.5, help="Loss weight for weighted kNN")
    
    args = parser.parse_args()
    
    if not Path(args.db_path).exists():
        print(f"Error: Database not found: {args.db_path}")
        return
    
    if args.analyze:
        analyze_database(args.db_path)
    
    if args.example_knn:
        example_knn_evaluation(args.db_path, k=args.k)
    
    if args.example_weighted:
        example_weighted_knn(args.db_path, k=args.k, loss_weight=args.loss_weight)
    
    if args.example_filtered:
        example_filtered_knn(args.db_path, k=args.k, category=args.category)
    
    if args.compare_baseline:
        compare_with_baseline(args.db_path, k=args.k)
    
    if not any([args.analyze, args.example_knn, args.example_weighted, 
                args.example_filtered, args.compare_baseline]):
        print("No action specified. Use --help for options.")
        analyze_database(args.db_path)


if __name__ == "__main__":
    main()