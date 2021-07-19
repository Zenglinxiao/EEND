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
        exclude = [set() for _ in range(batch_size)]
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


def increasing_ids(clusters, centers):
    """Remap clusters to be increasing order, and adjust centers accordingly.

    This should change [1, 2, 0, 1, 1] -> [0, 1, 2, 0, 0].
    """
    id_map = {}
    id_real = 0
    for value in clusters:
        if value not in id_map:
            id_map[value] = id_real
            id_real += 1  
    new_clusters = [id_map[v] for v in clusters]
    new_centers = [centers[id_map[i]] for i in range(len(centers))]
    return new_clusters, new_centers


def _find_duplicate(numbers):
    seen = set()
    duplicate = set()
    for x in numbers:
        if x not in seen:
            seen.add(x)
        else:
            duplicate.add(x)
    return duplicate, seen


def fix_duplicate(numbers, variable_pos):
    """Fix duplicate by change those to other number in the range of [0, M].

    Examples:
        [0, 1, 2, 2, 2], {2, 4} -> [0, 1, 3, 2, 4]
        [0, 1, 2, 2, 2], {2, 3} -> [0, 1, 3, 4, 2]
        [0, 1, 2, 2, 2], {3, 4} -> [0, 1, 2, 3, 4]
        [0, 1, 3, 4, 2], {2, 4} -> won't change if no duplicate
        [0, 1, 2, 3, 2], {2, 4} -> [0, 1, 4, 3, 2]
        [0, 1, 2, 2, 2], {1, 4} -> Not possible!
        [1, 1], {0, 1} -> [0, 1]
        [0, 0], {0, 1} -> [1, 0]
        [2, 2], {1} -> [2, 0]
        [2, 2], {0, 1} -> [0, 1]
    """
    duplicates, seen = _find_duplicate(numbers)
    if len(variable_pos) > 0:
        if len(duplicates) > 0:
            replace_candidates = []
            position2change = []
            for i in range(len(numbers)):
                if i not in seen:
                    replace_candidates.append(i)
                v = numbers[i]
                if v in duplicates and i in variable_pos:
                    position2change.append(i)
            # if len(position2change) != len(replace_candidates):
            #     print(f"numbers: {numbers}, variable_pos: {variable_pos}")
            # if len(position2change) < len(replace_candidates):
            #     import pdb; pdb.set_trace()
            #     raise ValueError("input argument not valid!")
            for pos, candidate in zip(position2change, replace_candidates):
                numbers[pos] = candidate       
    else:
        if len(duplicates) != 0:
            raise ValueError("Argument not meet function require!")
    return numbers


def silence_fix(clusters, lengths, silences):
    """Resolve duplicated id for clusters.

    Args:
        clusters (List[int]): list of cluster id
        lengths (List[int]): list of length
        silences (List[Set[int]]): list of silent relative id set

    Returns:
        cluster_ids (List[List[int]]): list of list int

    Examples:
        Given [0, 1, 1, 1] lengths [2, 2] silence [{}, {1}]
        -> [0, 1, 1, 0]
    """
    cluster_ids = [i for i in clusters]
    cursor = 0  # number of element visited
    for chunk_i, silence in enumerate(silences):
        if len(silence) > 0:
            chunk2fix = cluster_ids[cursor: cursor+lengths[chunk_i]]
            # elements in chunk2fix should be unique after fix
            fixed = fix_duplicate(chunk2fix, variable_pos=silence)
            cluster_ids[cursor: cursor+lengths[chunk_i]] = fixed
        cursor += lengths[chunk_i]
    return cluster_ids


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
    silences = None
    if th_silent > 0 and Y is not None:
        silences = batch_silence_detect(Y, threshold=th_silent)
    cannot_link = get_cannot_link_pairs(X, exclude=silences)
    padded_X = np.vstack(X)  # [(C, FS)] -> (sum(C), FS)
    try:
        clusters_, centers_ = cop_kmeans(
            padded_X,
            k=n_clusters,
            cl=cannot_link,  # can not link contraint
            initialization='kmpp',
            max_iter=300,
            tol=1e-4,
        )
    except Exception as err:
        print(err)
        # import pdb; pdb.set_trace()
        raise
    # remap cluster index in ascending order.
    clusters_, centers_ = increasing_ids(clusters_, centers_)
    # silences may result duplicate assign of cluster id due to removal of
    # can not link for these pair, thus need to resolve them
    if silences is not None:
        _lengths = [len(arr) for arr in X]  # first dims of the array list
        clusters_ = silence_fix(clusters_, _lengths, silences)
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
