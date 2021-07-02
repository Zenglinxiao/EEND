"""Utils relate to constraint clustering."""
import numpy as np
from cop_kmeans import cop_kmeans


def transitive_tuple(index_set):
    """Return transitive set for index list.
    
    Example:
        Given a set {3, 5, 7}, this function returns list of tuple as
        [(3, 5), (3, 7), (5, 7)]
    """
    result = []
    n_items = len(index_set)
    for i in range(n_items):
        cur_elem = index_set[i]
        for j in range(i+1, n_items):
            result.append((cur_elem, index_set[j]))
    return result


def batch_silence_detect(array_list, threshold=0.05):
    """Return silence index for each item in the array_list.
    Mean value for each item below threshold is considered as silence.

    Args:
        array_list (List[np.ndarray]): list of shape (frames, speakers)

    Returns:
        silence_ids (List[Set[int]])
    """
    silence_ids = []
    for X in array_list:
        silence_set = set()
        mean_logits = np.mean(X, axis=0) # shape: (speakers)
        silence_set = {i for i, logit in enumerate(mean_logits) if logit < threshold}
        silence_ids.append(silence_set)
    return silence_ids


def get_cannot_link_pairs(X, exclude=None):
    """Return can not link indice pairs based on X.

    Args:
        X (List[np.ndarray]): a list of ndarray. Each array shape in (C, FS).
        exclude (List[Set[int]]): a list of id to exclude in result.

    Return:
        List[Tuple[int, int]]: cannot link indices tuple list.
    """
    batch_size = len(X)
    if exclude is None:
        # init exclude list
        exclude = [set() * batch_size]
    cannot_link = []
    idx = 0
    for arr, _ex_set in zip(X, exclude):
        n_spks, _ = arr.shape
        mutuel_exclusive_set = []
        for i in range(n_spks):
            if i not in _ex_set:
                abs_i = idx + i
                mutuel_exclusive_set.append(abs_i)
        if len(mutuel_exclusive_set) > 1:
            not_links = transitive_tuple(mutuel_exclusive_set)
            cannot_link.extend(not_links)
        idx += n_spks
    return cannot_link


def contraint_kmeans(X, n_clusters, Y=None, th_silent=0.05):
    """Do contraint KMeans(COP-KMeans) on X.

    Args:
        X (List[np.ndarray]): shape (C, FS), clustering is done on feature space,
        where different point along C dimension should not be in same cluster.
        Point of same cluster can only origin from different array of the list.

        Y (List[np.darray]): shape (Frames, C), tagging prediction for each chunk.
        If provided, this will be used to detect silent speaker which will be exclude
        from can not link contraint list.

        th_silent (float): silent threshold to use when Y is provided.

    Returns:
        clusters (List[int]): list of cluster id correspond to each embedding
        centers (List[np.ndarray]): list of embeddings of each cluster center
    """
    silences = batch_silence_detect(Y, threshold=th_silent) if Y is not None else None
    cannot_link = get_cannot_link_pairs(X, exclude=silences)
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
