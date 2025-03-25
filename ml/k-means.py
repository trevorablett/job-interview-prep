class KMeans:
    def __init__(self, k: int, max_iters: int = 100):
        """
        Initializes the KMeans clustering algorithm.

        Args:
            k (int): Number of clusters.
            max_iters (int): Maximum number of iterations.
        """
        pass

    def fit(self, data: List[List[float]]) -> None:
        """
        Runs the k-means algorithm to fit the model to the given data.

        Args:
            data (List[List[float]]): A list of data points, each data point is a list of floats.
        """
        pass

    def predict(self, points: List[List[float]]) -> List[int]:
        """
        Predicts the cluster index for each point.

        Args:
            points (List[List[float]]): A list of data points to classify.

        Returns:
            List[int]: A list of predicted cluster indices.
        """
        pass

    def get_centroids(self) -> List[List[float]]:
        """
        Returns the current centroids.

        Returns:
            List[List[float]]: List of centroid coordinates.
        """
        pass


if __name__ == "__main__":
    import random

    # Sample 2D data: three visible clusters
    data = [
        [1.0, 2.0], [1.5, 1.8], [2.0, 2.2],
        [8.0, 8.0], [8.5, 8.3], [7.5, 7.8],
        [0.5, 0.8], [0.3, 0.2], [0.8, 0.6]
    ]

    # Set seed for reproducibility
    random.seed(42)

    # Create KMeans instance
    kmeans = KMeans(k=3, max_iters=100)

    # Fit model
    kmeans.fit(data)

    # Get centroids
    centroids = kmeans.get_centroids()
    print("Centroids:", centroids)

    # Predict cluster assignments for training data
    assignments = kmeans.predict(data)
    print("Cluster assignments:", assignments)