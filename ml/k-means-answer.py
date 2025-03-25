from typing import List
import random
import numpy as np

class KMeans:
    def __init__(self, k: int, max_iters: int = 100, vectorize=False):
        """
        Initializes the KMeans clustering algorithm.

        Args:
            k (int): Number of clusters.
            max_iters (int): Maximum number of iterations.
        """
        self.k = k
        self.max_iters = max_iters
        self.vectorize = vectorize

    def fit(self, data: List[List[float]]) -> None:
        """
        Runs the k-means algorithm to fit the model to the given data.

        Args:
            data (List[List[float]]): A list of data points, each data point is a list of floats.
        """

        data = np.array(data)  # n x f
        total_dist = float('inf')

        # using Forgy initialization: high spread
        cent_i = np.random.choice(np.arange(len(data)), size=self.k, replace=False)  # could be done with a loop but this is obviously faster
        self.centroids = data[cent_i]
        
        # alternate between expectation and maximization: assign clusters, then calculate total distance
        # looping version..numpy later
        for _ in range(self.max_iters):
            # assign clusters
            if self.vectorize:
                import ipdb; ipdb.set_trace()
                all_dists = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)  # broadcasting, data is n x f, centroids is k x f
                dists = all_dists.min(axis=1)
                closests = all_dists.argmin(axis=1)

            else:
                closests = []
                dists = []

                for d in data:
                    dist = float('inf')
                    closest_cent = 0
                    for c_i, c in enumerate(self.centroids):
                        c_dist = np.linalg.norm(d - c)
                        if c_dist < dist:
                            dist = c_dist
                            closest_cent = c_i
                    closests.append(closest_cent)
                    dists.append(dist)
                closests = np.array(closests)
                dists = np.array(dists)

            # calculate total distance
            new_total_dist = dists.sum()
            # print(f"dist: {new_total_dist}")

            if new_total_dist < total_dist:
                total_dist = new_total_dist
            else:
                break

            # new centroids from means of new clusters
            if self.vectorize:
                for k in range(self.k):
                    pts = data[closests == k]
                    if len(pts) > 0:  # if a centroid has no points assigned, leave it as is
                        self.centroids[k] = pts.mean(axis=0)
            else:
                for k in range(self.k):
                    mean = data[np.argwhere(closests == k)].mean(axis=0)
                    self.centroids[k] = mean

    def predict(self, points: List[List[float]]) -> List[int]:
        """
        Predicts the cluster index for each point.

        Args:
            points (List[List[float]]): A list of data points to classify.

        Returns:
            List[int]: A list of predicted cluster indices.
        """

        points = np.array(points)
        cluster_is = []

        if self.vectorize:
            all_dists = np.linalg.norm(points[:, np.newaxis] - self.centroids, axis=-1)
            cluster_is = np.argmin(all_dists, axis=-1)
        else:
            for p in points:
                all_dists = np.linalg.norm(self.centroids - p, axis=-1)
                cluster_is.append(int(all_dists.argmin()))
            
        return cluster_is

    def get_centroids(self) -> List[List[float]]:
        """
        Returns the current centroids.

        Returns:
            List[List[float]]: List of centroid coordinates.
        """
        return self.centroids


if __name__ == "__main__":

    # Sample 2D data: three visible clusters
    data = [
        [1.0, 2.0], [1.5, 1.8], [2.0, 2.2],
        [8.0, 8.0], [8.5, 8.3], [7.5, 7.8],
        [0.5, 0.8], [0.3, 0.2], [0.8, 0.6]
    ]

    print("Data: ", data)

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create KMeans instance
    kmeans = KMeans(k=3, max_iters=100, vectorize=True)

    # Fit model
    kmeans.fit(data)

    # Get centroids
    centroids = kmeans.get_centroids()
    print("Centroids:", centroids)

    # Predict cluster assignments for training data
    assignments = kmeans.predict(data)
    print("Cluster assignments:", assignments)