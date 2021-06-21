# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

import chainer
import numpy as np
from eend import kaldi_data
from eend import feature


def _count_frames(data_len, size, step):
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)


def _gen_frame_indices(
        data_length, size=2000, step=2000,
        use_last_samples=False,
        label_delay=0,
        subsampling=1):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step - subsampling * label_delay > 0:
            yield (i + 1) * step, data_length


class KaldiDiarizationDataset(chainer.dataset.DatasetMixin):

    def __init__(
            self,
            data_dir,
            dtype=np.float32,
            chunk_size=2000,
            context_size=0,
            frame_size=1024,
            frame_shift=256,
            subsampling=1,
            rate=16000,
            input_transform=None,
            use_last_samples=False,
            label_delay=0,
            n_speakers=None,
            shuffle=False,
            use_global_speaker_id=False,
    ):
        self.data_dir = data_dir
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.subsampling = subsampling
        self.input_transform = input_transform
        self.n_speakers = n_speakers
        self.chunk_indices = []
        self.label_delay = label_delay

        self.data = kaldi_data.KaldiData(self.data_dir)

        # make chunk indices: filepath, start_frame, end_frame
        for rec in self.data.wavs:
            # data_len: num of frames after frame subsampling
            data_len = int(self.data.reco2dur[rec] * rate / frame_shift)
            data_len = int(data_len / self.subsampling)
            # start and end frame index for each chunk with size chunk_size
            for st, ed in _gen_frame_indices(
                    data_len, chunk_size, chunk_size, use_last_samples,
                    label_delay=self.label_delay,
                    subsampling=self.subsampling):
                # return start and end frame index before subsampling
                self.chunk_indices.append(
                    (rec, st * self.subsampling, ed * self.subsampling))
        print(len(self.chunk_indices), " chunks")

        self.shuffle = shuffle
        self.use_global_speaker_id = use_global_speaker_id

    def __len__(self):
        return len(self.chunk_indices)

    def get_example(self, i):
        """Return transformed chunk[i].

        get chunk i (may contain multiple segment from the same record)
        -> get all speaker in the record, Dict them
        -> get Y = STFT(chunk), build speaker label T cor. to Y w kaldi file
        Y -> transform -> splice -> subsample [-> shuffle]
        T -> subsample [-> trim if given n_speaker -> shuffle]

        Returns:
            * Y (n_frames_ss, D)-shaped np.complex64 array
            * T (n_frames_ss, n_speakers)
        """
        # st, ed: start and end frame index before subsampling for chunked rec
        rec, st, ed = self.chunk_indices[i]
        # T: (n_frames, num_speakers)
        # spk_idxs: (num_speakers) map from local spk to global spk
        _feature_res = feature.get_labeledSTFT(
            self.data,
            rec,
            st,
            ed,
            self.frame_size,
            self.frame_shift,
            self.n_speakers,
            use_global_speaker_id=self.use_global_speaker_id,
        )
        if self.use_global_speaker_id:
            Y, T, spk_idxs = _feature_res
        else:
            Y, T = _feature_res
        # Y: (n_frames, num_ceps)
        Y = feature.transform(Y, self.input_transform)
        # Y_spliced: (n_frames, num_ceps * (context_size * 2 + 1))
        Y_spliced = feature.splice(Y, self.context_size)
        # Y_ss: (n_frames / subsampling, num_ceps * (context_size * 2 + 1))
        Y_ss, T_ss = feature.subsample(
            Y_spliced, T, subsampling=self.subsampling
        )

        # If the sample contains more than "self.n_speakers" speakers,
        #  extract top-(self.n_speakers) speakers
        if self.n_speakers and T_ss.shape[1] > self.n_speakers:
            selected_speakers = np.argsort(T_ss.sum(axis=0))[::-1][:self.n_speakers]
            T_ss = T_ss[:, selected_speakers]
            if self.use_global_speaker_id:
                # TODO: how to deal with spk_idxs?
                spk_idxs = spk_idxs[selected_speakers]

        # If self.shuffle is True, shuffle the order in time-axis
        # This operation improves the performance of EEND-EDA
        if self.shuffle:
            order = np.arange(Y_ss.shape[0])
            np.random.shuffle(order)
            Y_ss = Y_ss[order]
            T_ss = T_ss[order]

        if self.use_global_speaker_id:
            return Y_ss, T_ss, spk_idxs
        else:
            return Y_ss, T_ss
