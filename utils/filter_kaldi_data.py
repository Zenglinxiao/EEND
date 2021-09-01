#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
from eend.kaldi_data import KaldiData


def get_record_labeled_ratio(kaldi_data_obj):
    records_labeled_ratio = {}
    for recid, utts in kaldi_data_obj.segments.items():
        labeled_dur = sum([utt['et'] - utt['st'] for utt in utts])
        rec_dur = kaldi_data_obj.reco2dur[recid]
        labeled_ratio = labeled_dur / rec_dur
        if labeled_ratio > 1.0:
            raise ValueError(f"Illegal label ratio: {labeled_ratio}")
        records_labeled_ratio[recid] = labeled_ratio
    return records_labeled_ratio


def filter_data_by_labeled_ratio(kaldi_data, cutoff_ratio=0.8):
    rec_label_ratio = get_record_labeled_ratio(kaldi_data)
    eligible_recids = [
        rid for rid, ratio in rec_label_ratio.items()
        if ratio > cutoff_ratio
    ]
    if len(eligible_recids) == 0:
        raise RuntimeError("No record meet setting ratio requirement!")
    print(
        "{}/{} records with labeled ratio > {}".format(
            len(eligible_recids), len(rec_label_ratio), cutoff_ratio
        )
    )
    return eligible_recids


def extract_selected_kaldi(origin_kaldi_data, selected_rec_ids):
    segments = {
        rid: utts
        for rid, utts in origin_kaldi_data.segments.items()
        if rid in selected_rec_ids
    }
    utt_ids = {utt['utt'] for utts in segments.values() for utt in utts}
    utt2spk = {
        uttid: spkid
        for uttid, spkid in origin_kaldi_data.utt2spk.items()
        if uttid in utt_ids
    }
    wav_scp = {
        rid: wav_rxfilename
        for rid, wav_rxfilename in origin_kaldi_data.wavs.items()
        if rid in selected_rec_ids
    }
    reco2dur = {
        rid: duration
        for rid, duration in origin_kaldi_data.reco2dur.items()
        if rid in selected_rec_ids
    }
    spk2utt = {
        spkid: [uttid for uttid in uttids if uttid in utt_ids]
        for spkid, uttids in origin_kaldi_data.spk2utt.items()
    }
    return segments, utt2spk, wav_scp, reco2dur, spk2utt


def save_dict_like_file(dict_like, output_filename, fmt_fn):
    print(f"Redirecting result to {output_filename}...")
    with open(output_filename, "w", encoding='utf-8') as out_file:
        for k, v in dict_like.items():
            line2write = fmt_fn(k, v)
            out_file.write(line2write + "\n")


def build_kaldi_data(origin_kaldi_data, selected_rec_ids, output_dir):
    try:
        os.makedirs(output_dir)
    except FileExistsError as err:
        raise FileExistsError(f"{err}. Remove {output_dir} to continue")
    segments, utt2spk, wav_scp, reco2dur, spk2utt = extract_selected_kaldi(
        origin_kaldi_data, selected_rec_ids
    )
    save_dict_like_file(
        segments,
        os.path.join(output_dir, "segments"),
        fmt_fn=lambda rid, utts: "\n".join([
            f"{utt['utt']} {rid} {utt['st']} {utt['et']}" for utt in utts
        ])
    )
    save_dict_like_file(
        utt2spk,
        os.path.join(output_dir, "utt2spk"),
        fmt_fn=lambda uttid, spkid: f"{uttid} {spkid}"
    )
    save_dict_like_file(
        wav_scp,
        os.path.join(output_dir, "wav.scp"),
        fmt_fn=lambda rid, wav_rxfilename: f"{rid} {wav_rxfilename}"
    )
    save_dict_like_file(
        reco2dur,
        os.path.join(output_dir, "reco2dur"),
        fmt_fn=lambda rid, duration: f"{rid} {duration}"
    )
    save_dict_like_file(
        spk2utt,
        os.path.join(output_dir, "spk2utt"),
        fmt_fn=lambda spkid, uttids: f"{spkid} {' '.join(uttids)}"
    )


def filter_kaldi_data_main(input_dir, output_dir, cutoff_ratio=0.8):
    input_kaldi_data = KaldiData(input_dir)
    rec_ids = filter_data_by_labeled_ratio(input_kaldi_data, cutoff_ratio)
    build_kaldi_data(input_kaldi_data, rec_ids, output_dir)


def _get_opts():
    parser = ArgumentParser(description='filter kaldi data')
    parser.add_argument("input", type=str, help="input kaldi data directory.")
    parser.add_argument("output", type=str, help="output kaldi data directory")
    # Options
    parser.add_argument("-r", "--ratio", default=0.8, type=float,
                        help="labelled ratio below which to filter a record.")
    # Parse args and options
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opts = _get_opts()
    filter_kaldi_data_main(opts.input, opts.output, opts.ratio)
