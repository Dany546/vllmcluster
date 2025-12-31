"""
Updated knn_eval.py using vector search instead of precomputed distances.

This replaces O(N²) distance matrix lookups with on-demand kNN queries.
"""

import numpy as np
from typing import Dict, List, Tuple
from sqlvector_utils import connect_vec_db, query_knn, get_embedding_by_id


class KNNEvaluator:
    """
    KNN-based evaluator using vector search.
    
    Replaces distance matrix with on-demand kNN queries.
    """
    
    def __init__(self, embeddings_db_path: str, k: int = 10):
        """
        Initialize evaluator.
        
        Args:
            embeddings_db_path: Path to embeddings database
            k: Number of neighbors for kNN
        """
        self.embeddings_db_path = embeddings_db_path
        self.k = k
        self.conn = connect_vec_db(embeddings_db_path)
    
    def evaluate_sample(
        self,
        sample_id: int,
        weight_by_loss: bool = False,
        loss_weight: float = 0.3,
        filter_category: int = None
    ) -> Dict:
        """
        Evaluate a single sample using kNN.
        
        Args:
            sample_id: ID of sample to evaluate
            weight_by_loss: Whether to weight neighbors by loss
            loss_weight: Weight for loss term (0-1)
            filter_category: Optional category filter
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Get query embedding
        query_vec = get_embedding_by_id(self.conn, sample_id)
        
        if query_vec is None:
            return {"error": f"Sample {sample_id} not found"}
        
        # Build filter clause
        filter_clause = None
        filter_params = None
        if filter_category is not None:
            filter_clause = "flag_cat = ?"
            filter_params = (filter_category,)
        
        # Query kNN (get k+1 to exclude self)
        neighbors = query_knn(
            self.conn,
            query_vec,
            k=self.k + 1,
            filter_clause=filter_clause,
            filter_params=filter_params
        )
        
        # Remove self if present
        neighbors = [n for n in neighbors if n[0] != sample_id][:self.k]
        
        if len(neighbors) == 0:
            return {"error": "No neighbors found"}
        
        # Apply weighting if requested
        if weight_by_loss:
            weighted_neighbors = []
            for neighbor in neighbors:
                n_id, n_img_id, conf, box_loss, cls_loss, dfl_loss, cat, supercat, dist = neighbor
                
                # Compute weighted score
                total_loss = box_loss + cls_loss + dfl_loss
                score = (1 - loss_weight) * dist + loss_weight * total_loss
                
                weighted_neighbors.append((score, neighbor))
            
            # Sort by weighted score
            weighted_neighbors.sort(key=lambda x: x[0])
            neighbors = [n for _, n in weighted_neighbors]
        
        # Extract neighbor information
        neighbor_ids = [n[0] for n in neighbors]
        neighbor_cats = [n[6] for n in neighbors]
        neighbor_supercats = [n[7] for n in neighbors]
        neighbor_dists = [n[8] for n in neighbors]
        neighbor_losses = [(n[3], n[4], n[5]) for n in neighbors]  # box, cls, dfl
        
        # Compute statistics
        avg_distance = np.mean(neighbor_dists)
        std_distance = np.std(neighbor_dists)
        
        # Category consensus
        unique_cats, cat_counts = np.unique(neighbor_cats, return_counts=True)
        majority_cat = unique_cats[np.argmax(cat_counts)]
        cat_consensus = cat_counts.max() / len(neighbor_cats)
        
        # Supercat consensus
        unique_supercats, supercat_counts = np.unique(neighbor_supercats, return_counts=True)
        majority_supercat = unique_supercats[np.argmax(supercat_counts)]
        supercat_consensus = supercat_counts.max() / len(neighbor_supercats)
        
        return {
            "sample_id": sample_id,
            "k": len(neighbors),
            "neighbor_ids": neighbor_ids,
            "neighbor_categories": neighbor_cats,
            "neighbor_supercategories": neighbor_supercats,
            "avg_distance": avg_distance,
            "std_distance": std_distance,
            "majority_category": majority_cat,
            "category_consensus": cat_consensus,
            "majority_supercategory": majority_supercat,
            "supercategory_consensus": supercat_consensus,
            "neighbor_losses": neighbor_losses,
        }
    
    def evaluate_fold(
        self,
        validation_ids: List[int],
        ground_truth_categories: Dict[int, int],
        weight_by_loss: bool = False
    ) -> Dict:
        """
        Evaluate a validation fold using kNN.
        
        Args:
            validation_ids: List of validation sample IDs
            ground_truth_categories: Dict mapping sample_id -> true_category
            weight_by_loss: Whether to weight by loss
        
        Returns:
            Evaluation metrics for the fold
        """
        correct_predictions = 0
        total_samples = len(validation_ids)
        
        category_accuracies = {}
        confusion_matrix = {}
        
        for sample_id in validation_ids:
            result = self.evaluate_sample(sample_id, weight_by_loss=weight_by_loss)
            
            if "error" in result:
                continue
            
            predicted_cat = result["majority_category"]
            true_cat = ground_truth_categories.get(sample_id)
            
            if true_cat is None:
                continue
            
            # Update accuracy
            if predicted_cat == true_cat:
                correct_predictions += 1
            
            # Track per-category accuracy
            if true_cat not in category_accuracies:
                category_accuracies[true_cat] = {"correct": 0, "total": 0}
            
            category_accuracies[true_cat]["total"] += 1
            if predicted_cat == true_cat:
                category_accuracies[true_cat]["correct"] += 1
            
            # Update confusion matrix
            if true_cat not in confusion_matrix:
                confusion_matrix[true_cat] = {}
            if predicted_cat not in confusion_matrix[true_cat]:
                confusion_matrix[true_cat][predicted_cat] = 0
            confusion_matrix[true_cat][predicted_cat] += 1
        
        # Compute final metrics
        overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        per_category_acc = {
            cat: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            for cat, stats in category_accuracies.items()
        }
        
        return {
            "overall_accuracy": overall_accuracy,
            "correct_predictions": correct_predictions,
            "total_samples": total_samples,
            "per_category_accuracy": per_category_acc,
            "confusion_matrix": confusion_matrix,
        }
    
    def cross_validate(
        self,
        folds: List[Tuple[List[int], List[int]]],
        ground_truth_categories: Dict[int, int],
        weight_by_loss: bool = False,
        vary_k: List[int] = None
    ) -> Dict:
        """
        Perform k-fold cross-validation with varying k values.
        
        Args:
            folds: List of (train_ids, val_ids) tuples
            ground_truth_categories: Dict mapping sample_id -> category
            weight_by_loss: Whether to weight by loss
            vary_k: List of k values to try
        
        Returns:
            Cross-validation results
        """
        if vary_k is None:
            vary_k = [self.k]
        
        results = {}
        
        for k_val in vary_k:
            self.k = k_val
            fold_results = []
            
            for fold_idx, (train_ids, val_ids) in enumerate(folds):
                fold_result = self.evaluate_fold(
                    val_ids,
                    ground_truth_categories,
                    weight_by_loss=weight_by_loss
                )
                fold_result["fold"] = fold_idx
                fold_results.append(fold_result)
            
            # Aggregate across folds
            avg_accuracy = np.mean([r["overall_accuracy"] for r in fold_results])
            std_accuracy = np.std([r["overall_accuracy"] for r in fold_results])
            
            results[k_val] = {
                "k": k_val,
                "avg_accuracy": avg_accuracy,
                "std_accuracy": std_accuracy,
                "fold_results": fold_results,
            }
        
        return results
    
    def compare_weighting_schemes(
        self,
        validation_ids: List[int],
        ground_truth_categories: Dict[int, int],
        loss_weights: List[float] = None
    ) -> Dict:
        """
        Compare different loss weighting schemes.
        
        Args:
            validation_ids: Validation sample IDs
            ground_truth_categories: Ground truth labels
            loss_weights: List of loss weights to try
        
        Returns:
            Comparison results
        """
        if loss_weights is None:
            loss_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        results = {}
        
        for weight in loss_weights:
            # Temporarily set weight (we'll apply per-sample)
            fold_result = self.evaluate_fold(
                validation_ids,
                ground_truth_categories,
                weight_by_loss=(weight > 0)
            )
            
            results[weight] = {
                "loss_weight": weight,
                "accuracy": fold_result["overall_accuracy"],
                "per_category": fold_result["per_category_accuracy"],
            }
        
        return results
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def example_usage():
    """Example usage of the updated KNNEvaluator."""
    
    # Initialize evaluator
    evaluator = KNNEvaluator(
        embeddings_db_path="/path/to/embeddings.db",
        k=10
    )
    
    # Evaluate single sample
    result = evaluator.evaluate_sample(
        sample_id=42,
        weight_by_loss=True,
        loss_weight=0.3
    )
    print(f"Sample 42 evaluation:")
    print(f"  Majority category: {result['majority_category']}")
    print(f"  Category consensus: {result['category_consensus']:.2%}")
    print(f"  Average distance: {result['avg_distance']:.4f}")
    
    # Cross-validation with multiple k values
    folds = [
        ([1, 2, 3, 4, 5], [6, 7]),
        ([1, 2, 6, 7], [3, 4, 5]),
        # ... more folds
    ]
    ground_truth = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 0, 7: 1}
    
    cv_results = evaluator.cross_validate(
        folds=folds,
        ground_truth_categories=ground_truth,
        vary_k=[5, 10, 15, 20]
    )
    
    print("\nCross-validation results:")
    for k, result in cv_results.items():
        print(f"  k={k}: {result['avg_accuracy']:.2%} ± {result['std_accuracy']:.2%}")
    
    # Compare weighting schemes
    comparison = evaluator.compare_weighting_schemes(
        validation_ids=[6, 7],
        ground_truth_categories=ground_truth,
        loss_weights=[0.0, 0.2, 0.4, 0.6]
    )
    
    print("\nWeighting scheme comparison:")
    for weight, result in comparison.items():
        print(f"  Loss weight={weight}: accuracy={result['accuracy']:.2%}")
    
    evaluator.close()


if __name__ == "__main__":
    example_usage()