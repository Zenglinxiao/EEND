#!/bin/bash

# Copyright 2021 Ubiqus Labs. (author: Linxiao ZENG)
# Licensed under the MIT license.
#
stage=7
EXP_DIR=exp_vector_2spk
# The datasets for training must be formatted as kaldi data directory.
# Also, make sure the audio files in wav.scp are 'regular' wav files.
# Including piped commands in wav.scp makes training very slow
train_set=data/simu/data/train-all-960_ns2_beta2_nu_3_20 #data/simu/data/train-clean-100_ns2_beta2_500
valid_set=data/simu/data/dev-clean_ns2_beta2_500 #data/npr_tal_1-5spk.dev.8k
adapt_set=data/npr/npr_tal_2spk.train.8k
adapt_valid_set=data/npr/npr_tal_2spk.dev.8k

# Base config files for {train,infer}.py
train_config=conf/vector/train_2spk_l2norm.yaml # train.yaml
infer_config=conf/vector/infer_2spk_l2norm.yaml # _chunk500
adapt_config=conf/vector/adapt_2spk_l2norm.yaml

# Additional arguments passed to {train,infer}.py.
# You need not edit the base config files above
train_args=
infer_args=
adapt_args=

# Model averaging options
average_start=68
average_end=83

# Adapted model averaging options
adapt_average_start=40
adapt_average_end=50

# Resume training from snapshot at this epoch
# TODO: not tested
resume=-1

# Debug purpose
debug=

. path.sh
. cmd.sh
. parse_options.sh || exit

set -eu

if [ "$debug" != "" ]; then
    # debug mode
    train_set=data/simu/data/swb_sre_tr_ns2_beta2_1000
    train_config=conf/debug/train.yaml
    average_start=3
    average_end=5
    adapt_config=conf/debug/adapt.yaml
    adapt_average_start=6
    adapt_average_end=10
fi

# Parse the config file to set bash variables like: $train_frame_shift, $infer_gpu
eval `yaml2bash.py --prefix train $train_config`
eval `yaml2bash.py --prefix infer $infer_config`

# Append gpu reservation flag to the queuing command
if [ $train_gpu -le 0 ]; then
    train_cmd+=" --gpu 1"
fi
if [ $infer_gpu -le 0 ]; then
    infer_cmd+=" --gpu 1"
fi

# Build directry names for an experiment
#  - Training
#     exp/diarize/model/{train_id}.{valid_id}.{train_config_id}
#  - Decoding
#     exp/diarize/infer/{train_id}.{valid_id}.{train_config_id}.{infer_config_id}
#  - Scoring
#     exp/diarize/scoring/{train_id}.{valid_id}.{train_config_id}.{infer_config_id}
#  - Adapation from non-adapted averaged model
#     exp/diarize/model/{train_id}.{valid_id}.{train_config_id}.{avgid}.{adapt_config_id}
train_id=$(basename $train_set)
valid_id=$(basename $valid_set)
train_config_id=$(echo $train_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')
infer_config_id=$(echo $infer_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')
adapt_config_id=$(echo $adapt_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')

# Additional arguments are added to config_id
train_config_id+=$(echo $train_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')
infer_config_id+=$(echo $infer_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')
adapt_config_id+=$(echo $adapt_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')

model_id=$train_id.$valid_id.$train_config_id
model_dir=$EXP_DIR/diarize/model/$model_id
if [ $stage -le 1 ]; then
    echo "training model at $model_dir."
    if [ -d $model_dir ]; then
        echo "$model_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    work=$model_dir/.work
    mkdir -p $work
    $train_cmd $work/train.log \
        train.py \
            -c $train_config \
            $train_args \
            $train_set $valid_set $model_dir \
            || exit 1
fi
# ------------------------TO UNCOMMENT------------------------
ave_id=avg${average_start}-${average_end}
if [ $stage -le 2 ]; then
    echo "averaging model parameters into $model_dir/$ave_id.nnet.npz"
    if [ -s $model_dir/$ave_id.nnet.npz ]; then
        echo "$model_dir/$ave_id.nnet.npz already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    models=`eval echo $model_dir/snapshot_epoch-{$average_start..$average_end}`
    model_averaging.py $model_dir/$ave_id.nnet.npz $models || exit 1
fi

# # INFER_SET="dev.2spk.8k dev.3spk.8k dev.4spk.8k tal.dev.1spk.8k tal.dev.2spk.8k tal.dev.3spk.8k tal.dev.4spk.8k tal.dev.5spk.8k"
# # INFER_SET="dev.2spk.8k dev.3spk.8k dev.4spk.8k tal.dev.2spk.8k" #" tal.dev.3spk.8k tal.dev.4spk.8k tal.dev.5spk.8k"
# INFER_SET="simu/data/dev-clean_ns2_beta2_500"
INFER_SET="npr/dev.2spk.8k npr/dev.3spk.8k npr/tal.dev.2spk.8k"

infer_dir=$EXP_DIR/diarize/infer/$model_id.$ave_id.${infer_config_id}
if [ $stage -le 3 ]; then
    echo "inference at $infer_dir"
    # if [ -d $infer_dir ]; then
    #     echo "$infer_dir already exists. "
    #     echo " if you want to retry, please remove it."
    #     exit 1
    # fi
    #for dset in callhome2_spk2; do
    for dset in $INFER_SET; do
        # dev.5spk.8k
        if [ -d $infer_dir/$dset ]; then
            echo "$infer_dir/$dset already exists. "
            echo " if you want to retry, please remove it."
            exit 1
        fi
	    work=$infer_dir/$dset/.work
        mkdir -p $work
        # gold_nspk=`echo "$dset" | grep "[1-9]*spk" -o | sed "s/spk//g"`
        # gold_nspk=2
        echo "infer.py -c $infer_config data/${dset} $model_dir/$ave_id.nnet.npz $infer_dir/$dset"
        #data/eval/$dset \
        $infer_cmd $work/infer.log \
            infer.py \
            -c $infer_config \
            $infer_args \
            --num-clusters -1 \
            data/$dset \
            $model_dir/$ave_id.nnet.npz \
            $infer_dir/$dset \
            || exit 1
    done
fi

CLUSTER_METHOD="kmeans ahc sc none"
scoring_dir=$EXP_DIR/diarize/scoring/$model_id.$ave_id.${infer_config_id}
if [ $stage -le 4 ]; then
    echo "scoring at $scoring_dir"
    # if [ -d $scoring_dir ]; then
    #     echo "$scoring_dir already exists. "
    #     echo " if you want to retry, please remove it."
    #     exit 1
    # fi
    for dset in $INFER_SET; do
        if [ -d $scoring_dir/$dset ]; then
            echo "$scoring_dir/$dset already exists. "
            echo " if you want to retry, please remove it."
            exit 1
        fi
        # dev.5spk.8k
        gold_nspk=`echo "$dset" | grep "[1-9]*spk" -o | sed "s/spk//g"`
        # gold_nspk=-1
	    work=$scoring_dir/$dset/.work
        mkdir -p $work
        file_list_dset=$work/file_list_`basename $dset`
        find $infer_dir/$dset -iname "*.h5" > $file_list_dset
        # for med in 1 11; do
        med=11
        # for method in $CLUSTER_METHOD; do
        for method in none; do
            for th in 0.3 0.4 0.5 0.6 0.7; do
                make_rttm.py --median=$med --threshold=$th \
                    --frame_shift=$infer_frame_shift --subsampling=$infer_subsampling --sampling_rate=$infer_sampling_rate \
                    --num-clusters $gold_nspk --cluster-method $method \
                    $file_list_dset $scoring_dir/$dset/hyp_${method}_${th}_$med.rttm
                # -r data/eval/$dset/rttm \
                md-eval.pl -c 0.25 \
                    -r data/$dset/rttm \
                    -s $scoring_dir/$dset/hyp_${method}_${th}_$med.rttm > $scoring_dir/$dset/result_${method}_th${th}_med${med}_collar0.25 2>/dev/null || exit
            done
        done
    done
fi
# ------------------------/TO UNCOMMENT------------------------
adapt_model_dir=$EXP_DIR/diarize/model/$model_id.$ave_id.$adapt_config_id
if [ $stage -le 5 ]; then
    echo "adapting model at $adapt_model_dir"
    if [ -d $adapt_model_dir ]; then
        echo "$adapt_model_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    work=$adapt_model_dir/.work
    mkdir -p $work
    $train_cmd $work/train.log \
        train.py \
            -c $adapt_config \
            $adapt_args \
            --initmodel $model_dir/$ave_id.nnet.npz \
            $adapt_set $adapt_valid_set $adapt_model_dir \
                || exit 1
fi

adapt_ave_id=avg${adapt_average_start}-${adapt_average_end}
if [ $stage -le 6 ]; then
    echo "averaging models into $adapt_model_dir/$adapt_ave_id.nnet.gz"
    if [ -s $adapt_model_dir/$adapt_ave_id.nnet.npz ]; then
        echo "$adapt_model_dir/$adapt_ave_id.nnet.npz already exists."
        echo " if you want to retry, please remove it."
        exit 1
    fi
    models=`eval echo $adapt_model_dir/snapshot_epoch-{$adapt_average_start..$adapt_average_end}`
    model_averaging.py $adapt_model_dir/$adapt_ave_id.nnet.npz $models || exit 1
fi

infer_dir=$EXP_DIR/diarize/infer/$model_id.$ave_id.$adapt_config_id.$adapt_ave_id.${infer_config_id}
if [ $stage -le 7 ]; then
    echo "inference at $infer_dir"
    if [ -d $infer_dir ]; then
        echo "$infer_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    for dset in $INFER_SET; do
        if [ -d $infer_dir/$dset ]; then
            echo "$infer_dir/$dset already exists. "
            echo " if you want to retry, please remove it."
            exit 1
        fi
        work=$infer_dir/$dset/.work
        mkdir -p $work
        # gold_nspk=`echo "$dset" | grep "[1-9]*spk" -o | sed "s/spk//g"`
        gold_nspk=-1
        echo "infer.py -c $infer_config data/${dset} $model_dir/$ave_id.nnet.npz $infer_dir/$dset"
        #data/eval/$dset \
        $infer_cmd $work/infer.log \
            infer.py \
            -c $infer_config \
            $infer_args \
            --num-clusters $gold_nspk \
            data/$dset \
            $adapt_model_dir/$adapt_ave_id.nnet.npz \
            $infer_dir/$dset \
            || exit 1
    done
fi

scoring_dir=$EXP_DIR/diarize/scoring/$model_id.$ave_id.$adapt_config_id.$adapt_ave_id.${infer_config_id}
if [ $stage -le 8 ]; then
    echo "scoring at $scoring_dir"
    if [ -d $scoring_dir ]; then
        echo "$scoring_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    for dset in $INFER_SET; do
        if [ -d $scoring_dir/$dset ]; then
            echo "$scoring_dir/$dset already exists. "
            echo " if you want to retry, please remove it."
            exit 1
        fi
        # gold_nspk=`echo "$dset" | grep "[1-9]*spk" -o | sed "s/spk//g"`
        gold_nspk=-1
	    work=$scoring_dir/$dset/.work
        mkdir -p $work
        file_list_dset=$work/file_list_`basename $dset`
        find $infer_dir/$dset -iname "*.h5" > $file_list_dset
        # for med in 1 11; do
        med=11
        # for method in $CLUSTER_METHOD; do
        for method in none; do
            for th in 0.3 0.4 0.5 0.6 0.7; do
                make_rttm.py --median=$med --threshold=$th \
                    --frame_shift=$infer_frame_shift --subsampling=$infer_subsampling --sampling_rate=$infer_sampling_rate \
                    --num-clusters $gold_nspk --cluster-method $method \
                    $file_list_dset $scoring_dir/$dset/hyp_${method}_${th}_$med.rttm
                # -r data/eval/$dset/rttm \
                md-eval.pl -c 0.25 \
                    -r data/$dset/rttm \
                    -s $scoring_dir/$dset/hyp_${method}_${th}_$med.rttm > $scoring_dir/$dset/result_${method}_th${th}_med${med}_collar0.25 2>/dev/null || exit
            done
        done
    done
fi
# ------------------------TO UNCOMMENT------------------------
if [ $stage -le 9 ]; then
    # for dset in callhome2_spk2; do
    for dset in $INFER_SET; do
        best_score.sh $scoring_dir/$dset
    done
fi
echo "Finished !"
