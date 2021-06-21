#!/usr/bin/env python3
#
# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
import os
import h5py
import numpy as np
import chainer
from chainer import Variable
from chainer import serializers
from scipy.ndimage import shift
from eend.chainer_backend.models import BLSTMDiarization
from eend.chainer_backend.models import TransformerDiarization, TransformerEDADiarization
from eend.chainer_backend.utils import use_single_gpu
from eend import feature
from eend import kaldi_data
from eend import system_info


def infer(args):
    system_info.print_system_info()

    inference_kwargs = {
        "chunk_size": args.chunk_size,
        "gpu": -1,
        "out_dir": args.out_dir,
        "save_attention_weight": args.save_attention_weight
    }

    # Prepare model
    in_size = feature.get_input_dim(
        args.frame_size,
        args.context_size,
        args.input_transform)

    if args.model_type == "BLSTM":
        model = BLSTMDiarization(
            in_size=in_size,
            n_speakers=args.num_speakers,
            hidden_size=args.hidden_size,
            n_layers=args.num_lstm_layers,
            embedding_layers=args.embedding_layers,
            embedding_size=args.embedding_size
        )
    elif args.model_type == 'Transformer':
        if args.use_attractor:
            model = TransformerEDADiarization(
                in_size,
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                dropout=0,
                attractor_encoder_dropout=args.attractor_encoder_dropout,
                attractor_decoder_dropout=args.attractor_decoder_dropout,
            )
            inference_kwargs["num_speakers"] = args.num_speakers
            inference_kwargs["attractor_threshold"] = args.attractor_threshold
            inference_kwargs["shuffle"] = args.shuffle
        else:
            model = TransformerDiarization(
                args.num_speakers,
                in_size,
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                dropout=0
            )
    else:
        raise ValueError('Unknown model type.')

    serializers.load_npz(args.model_file, model)

    if args.gpu >= 0:
        gpuid = use_single_gpu()
        model.to_gpu()
        inference_kwargs["gpu"] = gpuid

    kaldi_obj = kaldi_data.KaldiData(args.data_dir)
    for recid in kaldi_obj.wavs:
        data, rate = kaldi_obj.load_wav(recid)
        Y = feature.stft(data, args.frame_size, args.frame_shift)
        Y = feature.transform(Y, transform_type=args.input_transform)
        Y = feature.splice(Y, context_size=args.context_size)
        Y = Y[::args.subsampling]
        out_chunks = model.inference(
            Y,
            recid=recid,
            **inference_kwargs
        )
        outfname = recid + '.h5'
        outpath = os.path.join(args.out_dir, outfname)
        if hasattr(model, 'label_delay'):
            outdata = shift(np.vstack(out_chunks), (-model.label_delay, 0))
        else:
            max_n_speakers = max([o.shape[1] for o in out_chunks])
            # out_chunks: padding [(T, speaker_active)]  --> [(T, max_n_speakers)]
            # FIXME: where inter-chunk label permutation comes
            out_chunks = [np.insert(o, o.shape[1], np.zeros((max_n_speakers - o.shape[1], o.shape[0])), axis=1) for o in out_chunks]
            # outdata: --vstack-> (B, T, max_n_speakers)
            outdata = np.vstack(out_chunks)
        with h5py.File(outpath, 'w') as wf:
            wf.create_dataset('T_hat', data=outdata)
