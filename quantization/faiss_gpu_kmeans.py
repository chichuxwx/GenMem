import faiss
import numpy as np


class FaissGPUKMeans:

    def __init__(
        self,
        n_clusters,
        niter=20,
        nredo=1,
        max_points_per_centroid=256,
        spherical=False,
        seed=42,
        verbose=False,
    ):
        self.n_clusters = n_clusters
        self.niter = niter
        self.nredo = nredo
        self.max_points_per_centroid = max_points_per_centroid
        self.spherical = spherical
        self.seed = seed
        self.verbose = verbose

        self.kmeans = None
        self.centroids = None
        self.index = None

    def fit(self, x):
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        dim = x.shape[1]

        self.kmeans = faiss.Kmeans(
            dim,
            self.n_clusters,
            niter=self.niter,
            nredo=self.nredo,
            seed=self.seed,
            gpu=False,
            verbose=self.verbose,
            spherical=self.spherical,
            max_points_per_centroid=self.max_points_per_centroid,
        )

        self.kmeans.train(x)

        self.centroids = self.kmeans.centroids
        self.index = self.kmeans.index

    def predict(self, x):
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        if self.index is None:
            if self.centroids is None:
                raise RuntimeError("KMeans 未训练且未加载 centroids/index")
            # 兜底重建
            d = self.centroids.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(self.centroids.astype(np.float32))

        _, labels = self.index.search(x, 1)
        return labels.reshape(-1)

    def fit_predict(self, x):
        self.fit(x)
        return self.predict(x)

    @property
    def cluster_centers_(self):
        return self.centroids
