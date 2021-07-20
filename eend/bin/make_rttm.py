#!/usr/bin/env python3

# Copyright 2021 Ubiqus Labs. (author: Linxiao ZENG)
# Licensed under the MIT license.

import argparse
import h5py
import numpy as np
import os
from scipy.signal import medfilt
from eend.clusters import regular_clustering  #, contraint_kmeans


parser = argparse.ArgumentParser(description='make rttm from decoded result')
parser.add_argument('file_list_hdf5')
parser.add_argument('out_rttm_file')
parser.add_argument('--threshold', default=0.5, type=float)
parser.add_argument('--frame_shift', default=256, type=int)
parser.add_argument('--subsampling', default=1, type=int)
parser.add_argument('--median', default=1, type=int)
parser.add_argument('--sampling_rate', default=16000, type=int)
vector_args = parser.add_argument_group('EEND-vector')
vector_args.add_argument('--num-clusters', default=-1, type=int)
vector_args.add_argument('--cluster-method', default="none", type=str,
                         choices=["none", "kmeans", "ahc", "sc", "cop_kmeans"],
                         help='clustering method to use')

args = parser.parse_args()

filepaths = [line.strip() for line in open(args.file_list_hdf5)]
filepaths.sort()


def rechunk_data(T_hat, spk_embs, chunk_sizes):
    """Split T_hat and spk_embs into chunks as described in chunk_sizes.

    Example:
        T_hat shape in (5980, 2),
        spk_embs shape in (24, 256)
        chunk_sizes [500, ..., 480] 11 500 followed by 480
        -> chunk_T_hat: [(500, 2)*11, (480, 2)]
        -> chunk_spk_embs: [(2, 256) * 12]
    """
    # rebuild T_hat in chunk as in shape [(#frames, #spk)]
    _cumsum_chunk_sizes = np.cumsum(chunk_sizes)
    chunk_T_hat = np.split(T_hat, _cumsum_chunk_sizes[:-1])
    # NOTE! can have different active speaker for each chunk as in EDA
    # in this case, need another field num_spk!
    num_spk = [arr.shape[1] for arr in chunk_T_hat]
    # rebuild out_spks in chunk as in shape [(#spk, #emb_size)]
    _cumsum_num_spk = np.cumsum(num_spk)
    if _cumsum_num_spk[-1] != spk_embs.shape[0]:
        msg = f"spk embedding shape {spk_embs.shape}, chunk speaker {num_spk}"
        raise ValueError(f"Input argument not match: {msg}")
    chunk_spk_embs= np.split(spk_embs, _cumsum_num_spk[:-1])
    return chunk_T_hat, chunk_spk_embs


def fill_within_right_cluster(T_hat, chunked_T_hat, cluster_ids):
    """Get fix T_hat after clustering.

    Args:
        T_hat (ndarray): shape in (sum(frame), max_n_spks)
        chunked_T_hat (List[ndarray]): list of shape (n_frame, n_spks)
        cluster_ids (List[List[int]]): clusters ids conform to chunked_T_hat

    Returns:
        * T_hat_clustered (ndarray): shape in (sum(frame), max_n_spks)
    """
    T_hat_clustered = np.zeros_like(T_hat)
    begin_frame = 0
    for T_hat_chunk_i, cids in zip(chunked_T_hat, cluster_ids):
        _n_frame, _n_spks = T_hat_chunk_i.shape
        end_frame = begin_frame + _n_frame
        seen = set()
        assert _n_spks == len(cids), "shape not match!"
        for local_id, cid in enumerate(cids):
            local_T = T_hat_chunk_i[:, local_id]
            if cid not in seen:
                seen.add(cid)
                T_hat_clustered[begin_frame:end_frame, cid] = local_T
            else:
                # take bigger value for same cluster at each frame
                T_compare = T_hat_clustered[begin_frame:end_frame, cid]
                T_bigger = np.where(
                    local_T > T_compare, local_T, T_compare,
                )
                T_hat_clustered[begin_frame:end_frame, cid] = T_bigger
    return T_hat_clustered


def predict(data, threshold, num_clusters, cluster_method):
    """Get prediction based on data and threshold, may involve clustering."""
    if cluster_method != "none":
        for field_name in ["out_spks", "chunk_sizes"]:
            if field_name not in data:
                raise ValueError(f"Missing {field_name} for clustering!")
        # extract ndarray from h5py Dataset
        chunk_sizes = data['chunk_sizes'][:]
        _T_hat = data['T_hat'][:]
        _out_spks = data['out_spks'][:]
        # sanity check
        max_n_speakers = _T_hat.shape[1]
        if max_n_speakers > num_clusters:
            raise RuntimeError("num_clusters is set too low")
        # rebuild embs chunks from data
        chunk_T_hat, chunk_spk_embs = rechunk_data(
            _T_hat, _out_spks, chunk_sizes
        )
        # clustering by speaker embeddings, return cluster id
        # in list of list id as shape (#chunk, #speaker)
        if cluster_method == "cop_kmeans":
            raise NotImplementedError("Currently do this in infer.py")
            # perform instance constrained clustering:
            # cluster id in same chunk suppose to be different
            # cluster_ids, cluster_centers = contraint_kmeans(
            #     chunk_spk_embs, n_clusters=num_clusters,
            #     Y=chunk_T_hat, th_silent=silent_threshold,
            # )
        else:
            # perform unconstrained clustering (kmeans/ahc/sc):
            # may assign same cluster id for speaker in the same chunk
            cluster_ids, cluster_centers = regular_clustering(
                chunk_spk_embs, n_clusters=num_clusters,
                # Y=chunk_T_hat, th_silent=silent_threshold,
                method=cluster_method,
            )
            # resolve same speaker id in a single chunk
            # fill a w/ chunk_T_hat as cluster_ids while fix conflict
            T_hat = fill_within_right_cluster(
                _T_hat,
                chunk_T_hat,
                cluster_ids,
            )
            a = np.where(T_hat > threshold, 1, 0)
    else:
        a = np.where(data['T_hat'][:] > threshold, 1, 0)
    return a


with open(args.out_rttm_file, 'w') as wf:
    for filepath in filepaths:
        session, _ = os.path.splitext(os.path.basename(filepath))
        data = h5py.File(filepath, 'r')
        a = predict(
            data,
            threshold=args.threshold,
            num_clusters=args.num_clusters,
            cluster_method=args.cluster_method,
        )
        # a = np.where(data['T_hat'][:] > args.threshold, 1, 0)
        if args.median > 1:
            a = medfilt(a, (args.median, 1))
        for spkid, frames in enumerate(a.T):
            frames = np.pad(frames, (1, 1), 'constant')
            changes, = np.where(np.diff(frames, axis=0) != 0)
            fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
            for s, e in zip(changes[::2], changes[1::2]):
                print(fmt.format(
                      session,
                      s * args.frame_shift * args.subsampling / args.sampling_rate,
                      (e - s) * args.frame_shift * args.subsampling / args.sampling_rate,
                      session + "_" + str(spkid)), file=wf)
