def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    import numpy as np
    
    points = np.asarray(points, dtype=float)
    assignments = np.asarray(assignments)
    
    n_features = points.shape[1]
    centroids = np.zeros((k, n_features))
    counts = np.zeros(k)

    for i, cluster in enumerate(assignments):
        centroids[cluster] += points[i]
        counts[cluster] += 1

    for j in range(k):
        if counts[j] > 0:
            centroids[j] /= counts[j]

    return centroids.tolist()