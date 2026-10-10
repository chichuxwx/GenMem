"""Residual KMeans with a weighted Euclidean/spherical distance.

The mixed distance for one level is

    (1 - spherical_weight) * ||x - c||^2
    + spherical_weight * (1 - cos(x, c)).

FAISS exposes Euclidean and spherical KMeans as separate modes, so the mixed
case is trained with a small batched Lloyd loop.  Pure endpoints still use
the existing FAISS implementation for exact comparability with the baseline
sweep.
"""

from __future__ import annotations

import os
from typing import Any, List

import faiss
import numpy as np

from .config import Config
from .faiss_gpu_kmeans import FaissGPUKMeans


class WeightedFaissKMeans:
    """KMeans whose assignment objective is a Euclidean/spherical mixture."""

    def __init__(
        self,
        n_clusters: int,
        niter: int = 20,
        nredo: int = 3,
        max_points_per_centroid: int = 256,
        spherical_weight: float = 0.5,
        seed: int = 42,
        verbose: bool = False,
        batch_size: int = 8192,
        centroid_update_steps: int = 6,
    ):
        if not 0.0 < spherical_weight < 1.0:
            raise ValueError("WeightedFaissKMeans is only for 0 < spherical_weight < 1")
        self.n_clusters = n_clusters
        self.niter = niter
        self.nredo = nredo
        self.max_points_per_centroid = max_points_per_centroid
        self.spherical_weight = float(spherical_weight)
        self.euclidean_weight = 1.0 - self.spherical_weight
        self.seed = seed
        self.verbose = verbose
        self.batch_size = batch_size
        self.centroid_update_steps = centroid_update_steps
        self.centroids: np.ndarray | None = None
        # Kept for compatibility with the standard wrapper.  Mixed prediction
        # intentionally uses the combined objective rather than this index.
        self.index = None

    @staticmethod
    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, 1e-12)

    def _distance_matrix(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        x_norm_sq = np.einsum("ij,ij->i", x, x)[:, None]
        c_norm_sq = np.einsum("ij,ij->i", centroids, centroids)[None, :]
        dots = x @ centroids.T
        euclidean = np.maximum(x_norm_sq + c_norm_sq - 2.0 * dots, 0.0)

        x_norm = np.sqrt(np.maximum(x_norm_sq, 1e-12))
        c_norm = np.sqrt(np.maximum(c_norm_sq, 1e-12))
        spherical = 1.0 - dots / (x_norm * c_norm)
        spherical = np.maximum(spherical, 0.0)
        return self.euclidean_weight * euclidean + self.spherical_weight * spherical

    def _predict_centroids(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        labels = np.empty(x.shape[0], dtype=np.int64)
        for start in range(0, x.shape[0], self.batch_size):
            stop = min(start + self.batch_size, x.shape[0])
            labels[start:stop] = np.argmin(
                self._distance_matrix(x[start:stop], centroids), axis=1
            )
        return labels

    def _objective(
        self, x: np.ndarray, centroids: np.ndarray, labels: np.ndarray
    ) -> float:
        total = 0.0
        count = 0
        for start in range(0, x.shape[0], self.batch_size):
            stop = min(start + self.batch_size, x.shape[0])
            distances = self._distance_matrix(x[start:stop], centroids)
            batch_labels = labels[start:stop]
            total += float(distances[np.arange(stop - start), batch_labels].sum())
            count += stop - start
        return total / max(count, 1)

    def _update_centroids(self, x: np.ndarray, labels: np.ndarray, old: np.ndarray) -> np.ndarray:
        counts = np.bincount(labels, minlength=self.n_clusters).astype(np.float32)
        sums = np.zeros((self.n_clusters, x.shape[1]), dtype=np.float32)
        np.add.at(sums, labels, x)

        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        normalized_x = x / np.maximum(x_norm, 1e-12)
        normalized_sums = np.zeros_like(sums)
        np.add.at(normalized_sums, labels, normalized_x)

        updated = old.copy()
        for cluster in range(self.n_clusters):
            if counts[cluster] <= 0:
                continue
            mean = sums[cluster] / counts[cluster]
            spherical_mean = normalized_sums[cluster]
            spherical_mean_norm = float(np.linalg.norm(spherical_mean))
            if spherical_mean_norm <= 1e-12:
                updated[cluster] = mean
                continue

            # Approximate the stationary point of the mixed objective in polar
            # coordinates c = r*u.  The endpoints reduce to the arithmetic
            # mean (Euclidean) and the normalized mean direction (spherical).
            mean_norm = float(np.linalg.norm(mean))
            if mean_norm <= 1e-12:
                updated[cluster] = spherical_mean / spherical_mean_norm
                continue
            direction = mean / mean_norm
            radius = max(float(np.dot(direction, sums[cluster]) / counts[cluster]), 1e-6)
            for _ in range(self.centroid_update_steps):
                vector = (
                    2.0 * self.euclidean_weight * radius * sums[cluster]
                    + self.spherical_weight * spherical_mean
                )
                vector_norm = float(np.linalg.norm(vector))
                if vector_norm <= 1e-12:
                    break
                direction = vector / vector_norm
                radius = max(float(np.dot(direction, sums[cluster]) / counts[cluster]), 1e-6)
            updated[cluster] = radius * direction
        return updated.astype(np.float32, copy=False)

    def _initial_centroids(self, x: np.ndarray, redo: int) -> np.ndarray:
        # A short FAISS warm start gives much more stable codebooks than
        # selecting random points, while the subsequent loop optimizes the
        # actual mixed objective.
        init = faiss.Kmeans(
            x.shape[1],
            self.n_clusters,
            niter=max(1, min(5, self.niter // 4)),
            nredo=1,
            seed=self.seed + redo,
            gpu=False,
            verbose=self.verbose,
            spherical=self.spherical_weight >= 0.5,
            max_points_per_centroid=self.max_points_per_centroid,
        )
        init.train(x)
        return init.centroids.astype(np.float32, copy=True)

    def fit(self, x: np.ndarray) -> None:
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if x.ndim != 2 or x.shape[0] < self.n_clusters:
            raise ValueError(f"Expected at least {self.n_clusters} two-dimensional samples")

        fit_count = min(x.shape[0], self.n_clusters * self.max_points_per_centroid)
        rng = np.random.default_rng(self.seed)
        if fit_count < x.shape[0]:
            fit_indices = rng.choice(x.shape[0], size=fit_count, replace=False)
            fit_x = np.ascontiguousarray(x[np.sort(fit_indices)])
        else:
            fit_x = np.ascontiguousarray(x)

        best_centroids = None
        best_objective = float("inf")
        for redo in range(self.nredo):
            centroids = self._initial_centroids(fit_x, redo)
            for iteration in range(self.niter):
                labels = self._predict_centroids(fit_x, centroids)
                updated = self._update_centroids(fit_x, labels, centroids)
                shift = float(np.max(np.abs(updated - centroids)))
                centroids = updated
                if self.verbose:
                    print(
                        f"mixed-kmeans redo={redo + 1}/{self.nredo} "
                        f"iter={iteration + 1}/{self.niter} shift={shift:.6g}"
                    )
                if shift < 1e-5:
                    break
            labels = self._predict_centroids(fit_x, centroids)
            objective = self._objective(fit_x, centroids, labels)
            if objective < best_objective:
                best_objective = objective
                best_centroids = centroids.copy()

        if best_centroids is None:
            raise RuntimeError("Mixed KMeans failed to produce centroids")
        self.centroids = best_centroids
        self.index = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("KMeans 未训练且未加载 centroids/index")
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        return self._predict_centroids(x, self.centroids)

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.predict(x)

    @property
    def cluster_centers_(self) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("KMeans 未训练")
        return self.centroids


class RQKMeansModel:
    """Residual quantization model with a per-level mixed distance weight."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dim = cfg.data.embedding_dim
        self.codebook_sizes = cfg.model.codebook_sizes
        self.n_levels = len(self.codebook_sizes)
        weights = getattr(cfg.model, "spherical_weight_per_level", None)
        if weights is None:
            weights = [float(getattr(cfg.model, "spherical_weight", 0.5))] * self.n_levels
        if len(weights) != self.n_levels:
            raise ValueError(f"spherical_weight_per_level must have {self.n_levels} entries")
        if any(not 0.0 <= float(w) <= 1.0 for w in weights):
            raise ValueError("spherical weights must be in [0, 1]")
        self.spherical_weight_per_level = [float(w) for w in weights]

        self.kmeans_layers: List[Any] = []
        self.codebooks: List[np.ndarray] = []
        for level, k in enumerate(self.codebook_sizes):
            spherical_weight = self.spherical_weight_per_level[level]
            if spherical_weight <= 0.0:
                kmeans: Any = FaissGPUKMeans(
                    n_clusters=k,
                    niter=cfg.model.max_iter,
                    nredo=cfg.model.nredo,
                    max_points_per_centroid=cfg.model.max_points_per_centroid,
                    spherical=False,
                    seed=cfg.model.random_state,
                    verbose=False,
                )
            elif spherical_weight >= 1.0:
                kmeans = FaissGPUKMeans(
                    n_clusters=k,
                    niter=cfg.model.max_iter,
                    nredo=cfg.model.nredo,
                    max_points_per_centroid=cfg.model.max_points_per_centroid,
                    spherical=True,
                    seed=cfg.model.random_state,
                    verbose=False,
                )
            else:
                kmeans = WeightedFaissKMeans(
                    n_clusters=k,
                    niter=cfg.model.max_iter,
                    nredo=cfg.model.nredo,
                    max_points_per_centroid=cfg.model.max_points_per_centroid,
                    spherical_weight=spherical_weight,
                    seed=cfg.model.random_state,
                    verbose=False,
                )
            self.kmeans_layers.append(kmeans)
        self.is_trained = False

    def fit(self, embeddings: np.ndarray) -> None:
        x = embeddings.astype(np.float32)
        residual = x.copy()
        self.codebooks = []
        print("Start training weighted RQ-KMeans")
        print("Embeddings shape:", x.shape)
        print("Spherical weights:", self.spherical_weight_per_level)
        for level, kmeans in enumerate(self.kmeans_layers):
            print(f"\nTraining level {level + 1}/{self.n_levels}")
            labels = kmeans.fit_predict(residual)
            centroids = kmeans.cluster_centers_
            self.codebooks.append(centroids.astype(np.float32, copy=True))
            residual = residual - centroids[labels]
        self.is_trained = True
        print("\nTraining finished")

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        residual = embeddings.astype(np.float32).copy()
        all_labels = []
        for kmeans in self.kmeans_layers:
            labels = kmeans.predict(residual)
            all_labels.append(labels)
            residual = residual - kmeans.centroids[labels]
        return np.stack(all_labels, axis=1)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        embeddings = np.zeros((codes.shape[0], self.dim), dtype=np.float32)
        for level in range(self.n_levels):
            embeddings += self.codebooks[level][codes[:, level]]
        return embeddings

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        meta = {
            "dim": self.dim,
            "codebook_sizes": self.codebook_sizes,
            "n_levels": self.n_levels,
            "spherical_weight_per_level": self.spherical_weight_per_level,
        }
        np.save(os.path.join(path, "meta.npy"), meta)
        for i, centroids in enumerate(self.codebooks):
            np.save(os.path.join(path, f"codebook_{i}.npy"), centroids)
        print("✅ Model saved:", path)

    def load(self, path: str) -> None:
        meta = np.load(os.path.join(path, "meta.npy"), allow_pickle=True).item()
        self.dim = meta["dim"]
        self.codebook_sizes = meta["codebook_sizes"]
        self.n_levels = meta["n_levels"]
        self.codebooks = []
        for i in range(self.n_levels):
            centroids = np.load(os.path.join(path, f"codebook_{i}.npy")).astype(np.float32)
            self.codebooks.append(centroids)
            self.kmeans_layers[i].centroids = centroids
            self.kmeans_layers[i].index = None
        self.is_trained = True
