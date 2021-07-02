"""Utils relate to constraint clustering."""
import numpy as np
from cop_kmeans import cop_kmeans


def get_cannot_link_pairs(X):
    """Return can not link indice pairs based on X.

    Args:
        X (List[np.ndarray]): a list of ndarray. Each array shape in (B, F).

    Return:
        List[Tuple[int, int]]: cannot link indices before transitive closure.
    """
    cannot_link = []
    idx = 0
    for arr in X:
        n_item, _ = arr.shape
        for i in range(1, n_item):
            cannot_link.append((idx, idx + i))
        idx += n_item
    return cannot_link


def contraint_kmeans(X, n_clusters):
    """Do contraint KMeans(COP-KMeans) on X.

    X (List[np.ndarray]): shape (C, FS), clustering is done on feature space,
    where different point along C dimension should not be in same cluster.
    Point of same cluster can only origin from different array of the list.

    Returns:
        clusters (List[int]): list of cluster id correspond to each embedding
        centers (List[np.ndarray]): list of embeddings of each cluster center
    """
    cannot_link = get_cannot_link_pairs(X)
    padded_X = np.vstack(X)  # [(C, FS)] -> (sum(C), FS)
    clusters_, centers_ = cop_kmeans(
        padded_X,
        k=n_clusters,
        cl=cannot_link,  # can not link contraint
        initialization='kmpp',
        max_iter=300,
        tol=1e-4,
    )
    # TODO: remap cluster index in ascending order.
    # reshape cluster_ to that of X
    cluster_ids = []
    k = 0
    for i in range(len(X)):
        chunk_result = []
        for _ in range(len(X[i])):
            chunk_result.append(clusters_[k])
            k += 1
        cluster_ids.append(chunk_result)
    assert k == len(clusters_)
    return cluster_ids, centers_
