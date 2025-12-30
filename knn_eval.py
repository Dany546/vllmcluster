# knn_eval.py
import numpy as np
from sqlvector_utils import connect_vec_db, query_knn

def inverse_distance_weights(distances, eps=1e-6, power=1.0):
    w = 1.0 / (distances + eps)
    if power != 1.0:
        w = w ** power
    return w / (w.sum() + 1e-12)

def gaussian_weights(distances, sigma=1.0):
    w = np.exp(-0.5 * (distances / (sigma + 1e-12)) ** 2)
    return w / (w.sum() + 1e-12)

def evaluate_sample(db_path, query_vec: np.ndarray, K_max: int = 50, k_list=[1,3,5], weightings=None):
    if weightings is None:
        weightings = {
            "uniform": lambda d: np.ones_like(d) / len(d),
            "euclidian": lambda d: inverse_distance_weights(d, power=2.0),
            "gauss_0.5": lambda d: gaussian_weights(d, sigma=0.5),
        }
    conn = connect_vec_db(db_path)
    neighbors = query_knn(conn, query_vec.astype(np.float32), K_max)
    conn.close()

    if len(neighbors) == 0:
        return {}

    dists = np.array([r[-1] for r in neighbors], dtype=np.float32)
    # Example: use box_loss as target; adapt to the metric you want to predict
    losses = np.array([r[3] for r in neighbors], dtype=np.float32)

    results = {}
    for k in k_list:
        top_losses = losses[:k]
        top_dists = dists[:k]
        for wname, wfunc in weightings.items():
            w = wfunc(top_dists)
            pred = (w * top_losses).sum()
            results[(k, wname)] = float(pred)
    return results
