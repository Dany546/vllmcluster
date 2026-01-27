"""Feature pipeline primitives and extractors for gridsearch.

This module provides a `FeaturePipeline` wrapper and a set of transformer
classes (StandardScalerWrapper, SpearmanFeatureSelector, PLSExtractor,
KernelPCAExtractor, TSNEExtractor, UMAPExtractor).

Transformers implement `fit(X, y=None)`, `transform(X)` and optional
`fit_transform(X, y=None)` conveniences. `FeaturePipeline` composes steps and
exposes `fit_transform(X, y)` and `transform(X)`.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


class NoFeaturesLeft(Exception):
    pass


class StandardScalerWrapper(TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X, y=None):
        return self.scaler.fit_transform(X)


class SpearmanFeatureSelector(TransformerMixin):
    def __init__(self, threshold: float):
        self.threshold = float(threshold)
        self.mask: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # compute Spearman per feature
        rhos = []
        for i in range(X.shape[1]):
            rho, _ = spearmanr(X[:, i], y)
            if np.isnan(rho):
                rho = 0.0
            rhos.append(rho)
        rhos = np.array(rhos)
        self.mask = np.abs(rhos) >= self.threshold
        if not np.any(self.mask):
            raise NoFeaturesLeft("Spearman filter removed all features")
        return self

    def transform(self, X: np.ndarray):
        if self.mask is None:
            raise RuntimeError("SpearmanFeatureSelector not fitted")
        return X[:, self.mask]

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        self.fit(X, y)
        return self.transform(X)


class PLSExtractor(TransformerMixin):
    def __init__(self, n_components: int = 5):
        self.n_components = int(n_components)
        self.model: Optional[PLSRegression] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = PLSRegression(n_components=self.n_components)
        self.model.fit(X, y)
        return self

    def transform(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("PLSExtractor not fitted")
        return self.model.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        self.model = PLSRegression(n_components=self.n_components)
        res = self.model.fit_transform(X, y)
        # sklearn PLSRegression.fit_transform returns (X_scores, Y_scores)
        if isinstance(res, tuple):
            return res[0]
        return res


class KernelPCAExtractor(TransformerMixin):
    def __init__(self, n_components: int = 5, kernel: str = "rbf"):
        self.n_components = int(n_components)
        self.kernel = kernel
        self.model: Optional[KernelPCA] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.model = KernelPCA(n_components=self.n_components, kernel=self.kernel, fit_inverse_transform=False)
        self.model.fit(X)
        return self

    def transform(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("KernelPCAExtractor not fitted")
        return self.model.transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.model = KernelPCA(n_components=self.n_components, kernel=self.kernel, fit_inverse_transform=False)
        return self.model.fit_transform(X)


class TSNEExtractor(TransformerMixin):
    """TSNE extractor: fit_transform supported. transform uses a KNN-based mapper.

    We implement a nearest-neighbour regression from original X_train -> tsne_embeddings
    so that `transform(X_test)` returns a weighted average of neighbours' embeddings.
    This is an approximation but provides a deterministic transform behavior.
    """

    def __init__(self, n_components: int = 2, perplexity: float = 30.0, method: str = "barnes_hut"):
        self.n_components = int(n_components)
        self.perplexity = float(perplexity)
        self.method = method
        self._X_train: Optional[np.ndarray] = None
        self._emb_train: Optional[np.ndarray] = None
        self._nn: Optional[NearestNeighbors] = None
        self.init: Optional[str] = "pca"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self._X_train = X
        self._emb_train = TSNE(n_components=self.n_components, perplexity=self.perplexity, init=self.init, method=self.method, learning_rate="auto").fit_transform(X)
        # self._nn = NearestNeighbors(n_neighbors=min(5, max(5, X.shape[0]))).fit(X)
        return self

    def transform(self, X: np.ndarray):
        if self._nn is None or self._emb_train is None:
            raise RuntimeError("TSNEExtractor not fitted")
        dists, idxs = self._nn.kneighbors(X, n_neighbors=min(5, self._emb_train.shape[0]))
        # inverse-distance weighted average
        eps = 1e-8
        weights = 1.0 / (dists + eps)
        weights = weights / weights.sum(axis=1, keepdims=True)
        out = np.einsum('ij,ijk->ik', weights, self._emb_train[idxs])
        return out

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.fit(X, y)
        return self._emb_train


class UMAPExtractor(TransformerMixin):
    def __init__(self, n_components: int = 2, **kwargs):
        self.n_components = int(n_components)
        self.kwargs = kwargs
        try:
            import umap

            self.umap = umap.UMAP(n_components=self.n_components, **self.kwargs)
        except Exception:  # pragma: no cover - optional dependency
            self.umap = None

        self._fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        if self.umap is None:
            raise RuntimeError("UMAP not installed")
        self.umap.fit(X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray):
        if not self._fitted or self.umap is None:
            raise RuntimeError("UMAPExtractor not fitted or umap missing")
        return self.umap.transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        if self.umap is None:
            raise RuntimeError("UMAP not installed")
        out = self.umap.fit_transform(X)
        self._fitted = True
        return out


class FeaturePipeline:
    def __init__(self, steps: List[Any]):
        self.steps = steps

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        cur = X
        for step in self.steps:
            # some transformers expect y
            if hasattr(step, 'fit_transform'):
                cur = step.fit_transform(cur, y) 
            else:
                step.fit(cur, y) 
                cur = step.transform(cur)
        return cur

    def transform(self, X: np.ndarray) -> np.ndarray:
        cur = X
        for step in self.steps:
            # assume fitted
            cur = step.transform(cur)
        return cur

    def get_meta(self) -> Dict[str, Any]:
        # return descriptive metadata about pipeline
        meta = {}
        meta['steps'] = [type(s).__name__ for s in self.steps]
        return meta
