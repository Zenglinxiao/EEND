# Copyright 2021 Ubiqus, Ltd. (author: Linxiao ZENG)
# Licensed under the MIT license.
import numpy as np
from scipy.signal import medfilt
from collections import Counter
from resemblyzer import preprocess_wav, VoiceEncoder
from eend.clusters import regular_clustering, rechunk_prediction


def get_resemblyzer_model(device=None):
    """Load voice encoder model."""
    return VoiceEncoder(device=device)


def spk_activte_window_gate(T_hat, threshold, median):
    """
    Given a audio speaker activity mesure calculate on frame,
    consider the spk is active if value > threshold in most of
    time for a span of at least `median` frames.
    
    Args:
        T_hat (ndarray): shape in (n_frames, n_spks)
        threshold (float): value above which considered as active
        median (int): length of the median filter for smooth result

    Returns:
        shape (n_frames, n_spks)
    """
    a = np.where(T_hat > threshold, 1, 0)
    a = medfilt(a, (median, 1))  # smooth result by average of median frames
    return a


def mask_multiple_active(spks_actives):
    """Return a mask that indicate frames active with multiple speakers.
    
    Args:
        spks_actives (ndarray): shape in (n_frames, n_spks)

    Returns:
        mask (ndarray): shape in (n_frames)
    """
    # n_frames, n_spks = spks_actives.shape
    active_spks = np.sum(spks_actives, axis=1)
    overlap_mask = (active_spks > 1)
    return overlap_mask


def find_nonoverlap_range(overlap_mask):
    result = []
    st_idx = -1
    for i, v in enumerate(overlap_mask):
        if v and st_idx > 0:
            # first overlap position after no overlap
            result.append((st_idx, i))
            st_idx = -1
        if not v and st_idx < 0:
            # begin of no overlap
            st_idx = i
    if st_idx >= 0:
        result.append((st_idx, i+1))
    return result


def load_record_del_overlap(kaldi_obj, recid, st, ed, T_hat, threshold, median, frame_shift, subsampling):
    spk_active_tags = spk_activte_window_gate(T_hat, threshold, median)
    overlap_mask = mask_multiple_active(spk_active_tags)
    no_overlap_ranges = find_nonoverlap_range(overlap_mask)
    total_frames_in_inf = len(overlap_mask)
    assert (ed - st) == total_frames_in_inf, "chunk size not match"
    record_loaded = []
    for st_idx, ed_idx in no_overlap_ranges:
        st_sample = (st + st_idx) * frame_shift * subsampling
        ed_sample = (st + ed_idx) * frame_shift * subsampling
        record_partial, rate = kaldi_obj.load_wav(recid, start=st_sample, end=ed_sample)
        record_loaded.append(record_partial)
    concat_partials = np.concatenate(record_loaded, axis=0)
    return concat_partials, rate


def dvector_chunked_record(
    recid,
    voice_encoder,
    kaldi_obj,
    predict_h5,
    threshold,
    median,
    frame_shift,
    subsampling,
):
    """Return the dvector for all chunks of a record.

    helper function: for each chunk of a record, do encoder.embed_utterance of resemblyzer,
    return list of partial_embs
    """
    # prepare the inference into desired form
    predict_chunks = rechunk_prediction(predict_h5)
    T_hat_chunk = predict_chunks['T_hat']
    _chunk_sizes = predict_chunks['chunk_sizes']

    num_chunks = len(_chunk_sizes)
    assert len(T_hat_chunk) == num_chunks, "chunk_size not match"

    #record_dur = kaldi_obj.reco2dur[recid]
    #_time_per_frame = record_dur / sum(_chunk_sizes)
    #chunk_times = [_size * _time_per_frame for _size in _chunk_sizes]  # elapsed times of chunks

    cur_frame_id = 0
    all_partials_embeds = []
    for chunk_i in range(num_chunks):
        T_hat_chunk_i = T_hat_chunk[chunk_i]  # EEND prediction for chunk i of this record, shape: (n_frame, n_spk)
        # 1. Load wav correspond to the chunk
        end_frame_id = cur_frame_id + _chunk_sizes[chunk_i]
        # NOTE: load record w/o overlap
        record_chunk, _sr = load_record_del_overlap(
            kaldi_obj, recid, cur_frame_id, end_frame_id, T_hat_chunk_i, threshold, median,
            frame_shift=frame_shift, subsampling=subsampling
        )
        # 2. process the wav chunk as required by resemblyzer
        wav_chunk = preprocess_wav(record_chunk, _sr)  # resample -> normalize -> trim silence
        # 3. embed this wav with resemblyzer encoder: split wav into partials and encode partials' mel to embeddings
        # return partial_embeds in shape of (n_partials, embed_dim)
        _, chunk_partials_embeds, _ = voice_encoder.embed_utterance(
            wav_chunk, return_partials=True #, rate=rate, min_coverage=min_coverage
        )
        all_partials_embeds.append(chunk_partials_embeds)
        cur_frame_id = end_frame_id
    return all_partials_embeds, T_hat_chunk


def reorder_by_mass(T_hat, cluster_ids, n_clusters=None):
    """Return reordered T_hat according to freqs of cluster_ids.

    Args:
        T_hat (ndarray): shape in (n_frames, n_spks)
        cluster_ids (List[int]): 0 <= values < n_spks
        n_clusters (int): number of cluster after reorder.

    Returns:
        T_hat_reordered (ndarray) reordered on axis=1
    """
    _n_spks = T_hat.shape[1]
    if n_clusters is not None and n_clusters > _n_spks:
        _n_spks = n_clusters
    cluster_id_counts = Counter(cluster_ids)
    # update count with all spk id only to make sure all id is present
    cluster_id_counts.update(list(range(_n_spks)))
    cluster_id_mass_order = [k for k, _ in cluster_id_counts.most_common()]
    assert list(range(_n_spks)) == sorted(cluster_id_mass_order), \
        f"{cluster_id_mass_order} not match {_n_spks}"
    T_hat_mass = np.sum(T_hat, axis=0)
    # sort desending by EEND prediction logit level
    T_hat_mass_order = np.argsort(T_hat_mass)[::-1]
    T_hat_reordered = np.zeros_like(T_hat)
    try:
        T_hat_reordered[:, cluster_id_mass_order] = T_hat[:, T_hat_mass_order]
    except Exception as err:
        print(f"{cluster_id_mass_order} - {T_hat_mass_order}")
        raise
    return T_hat_reordered


def reorder_That_by_dvector(T_hat_chunks, d_vector_chunks, n_clusters=None, clustering_method="kmeans"):
    # print(f"shape: {[arr.shape for arr in T_hat_chunks]}")
    if n_clusters is None or n_clusters <= 0:
        n_clusters = max([arr.shape[1] for arr in T_hat_chunks])
        # print(f"Infer num_clusters to be: {n_clusters}")
    assert len(T_hat_chunks) == len(d_vector_chunks), "input not match"
    if n_clusters == 1:
        return T_hat_chunks
    num_chunks = len(T_hat_chunks)
    # clustering on all_partials_embeds and return cluster_id group for each chunk's partials
    all_cluster_ids = regular_clustering(d_vector_chunks, n_clusters, method=clustering_method)  # shape: (num_chunks, n_partials)
    # TODO: How to revise T_hat based on all_cluster_ids?
    reordered_T_hat = []
    for chunk_i in range(num_chunks):
        chunk_cluster_ids = all_cluster_ids[chunk_i]
        T_hat_chunk_i = T_hat_chunks[chunk_i]
        try:
            reordered_T_hat_chunk_i = reorder_by_mass(T_hat_chunk_i, chunk_cluster_ids, n_clusters)
        except AssertionError as err:
            print(f"number Cluster: {n_clusters}")
            print(all_cluster_ids)
            raise
        reordered_T_hat.append(reordered_T_hat_chunk_i)
    return reordered_T_hat


def resemblyzer_realign_main(
    recid,
    resemblyzer_model,
    kaldi_obj,
    predict_h5,
    threshold,
    median,
    frame_shift,
    subsampling,
    num_clusters,
    cluster_method,
):
    """Main entry point to realign T_hat based on d_vector embeddings."""
    dvector_embs_chunks, T_hat_chunks = dvector_chunked_record(
        recid, resemblyzer_model, kaldi_obj, predict_h5,
        threshold, median, frame_shift, subsampling,
    )
    reordered_T_hat = reorder_That_by_dvector(
        T_hat_chunks, dvector_embs_chunks,
        n_clusters=num_clusters, clustering_method=cluster_method,
    )
    realigned_T_hat = np.concatenate(reordered_T_hat, axis=0)
    return realigned_T_hat
