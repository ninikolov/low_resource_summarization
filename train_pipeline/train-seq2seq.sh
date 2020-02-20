#!/usr/bin/env bash

### Prepare data, train model, generate

# The noisy input file to train on
src_name=$1
# The target train file
tgt_name=$2
# GPUs to use for training/generation
cuda_dev=$3
# If set to true, will run scripts to convert the data to BPE and generate the training files.
# Otherwise go directly to training.
prepare_data=$4
# Model to use
model_architecture=$5
# BPE dict src
bpe_dict_src=$6
# BPE dict tgt
bpe_dict_tgt=$7
# SRC dict to reuse
src_dict=$8
# TGT dict to reuse
tgt_dict=$9

voc_size=20000
data_save_dir=fairseq-data

curr_dir=${PWD##*/}
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# Dictionary args
if [ -z "$tgt_dict" ] || [ -z "$src_dict" ] ; then
    dict_arg=""
else
    echo "Reusing dictionaries: $src_dict $tgt_dict"
    dict_arg="--srcdict $src_dict --tgtdict $tgt_dict"
    if [ "$tgt_dict" == "$src_dict" ]; then
        share_embs="--share-all-embeddings"
    else
        share_embs=""
    fi
fi

## Data preparation
if [ "$prepare_data" = true ] ; then
    # Learn BPE rules from data
    if [ -z "$bpe_dict_src" ] ; then
        echo " * Learning BPE from train.$src_name.clean * "
        bpe_dict_src=train.$src_name.clean.bpe
        rm $bpe_dict_src
        python $BPE_CODE/learn_bpe.py --input train.$src_name.clean --output $bpe_dict_src --symbols 50000
        echo ""
    else
        echo " * Using existing BPE dictionary $bpe_dict_src * "
    fi

    if [ -z "$bpe_dict_tgt" ] ; then
        echo " * Learning BPE from train.$tgt_name.clean * "
        bpe_dict_tgt=train.$tgt_name.clean.bpe
        rm $bpe_dict_tgt
        python $BPE_CODE/learn_bpe.py --input train.$tgt_name.clean --output $bpe_dict_tgt --symbols 50000
    else
        echo " * Using existing BPE dictionary $bpe_dict_src * "
    fi

    # Apply BPE
    echo ""
    echo " * Apply BPE to data * "
    echo "$1"
    bash $SCRIPT_DIR/bpe_pipeline.sh $src_name $bpe_dict_src
    echo "$2"
    bash $SCRIPT_DIR/bpe_pipeline.sh $tgt_name $bpe_dict_tgt
    echo ""

    ## Prepare Fairseq training data
    # Remove any old data dir
    rm -rf $data_save_dir
    echo " * Prepare Fairseq training data * "
    python $FAIRSEQ_PATH/preprocess.py --source-lang $src_name --target-lang $tgt_name \
        --trainpref train --validpref valid --testpref test --destdir $data_save_dir \
        --nwordssrc $voc_size --nwordstgt $voc_size $dict_arg --workers 4
fi
echo ""
echo " * Train model saving at $model_architecture * "

gpus=(${3//,/ })
gpu_count=${#gpus[@]}
# Divide learning rate by number of GPUs available
lr=$(bc -l <<< "0.0003125/$gpu_count")
echo "Learning rate with $gpu_count GPUs: $lr"
echo

if [[ $model_architecture == lstm* ]]; then
    # Train LSTM
    CUDA_VISIBLE_DEVICES=$3 python $FAIRSEQ_PATH/train.py $data_save_dir \
        --optimizer adam --lr $lr --dropout 0.2 --arch $model_architecture --save-dir $model_architecture \
        --no-progress-bar --skip-invalid-size-inputs-valid-test --max-tokens 8000 $share_embs #--no-epoch-checkpoints
elif [[ $model_architecture == transformer* ]]; then
    # Train Transformer
    CUDA_VISIBLE_DEVICES=$3 python $FAIRSEQ_PATH/train.py $data_save_dir \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 --warmup-updates 4000 --lr $lr --min-lr 1e-09 --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --arch $model_architecture --save-dir $model_architecture --no-progress-bar \
        --skip-invalid-size-inputs-valid-test --max-tokens 1000 $share_embs --dropout 0.1 --no-epoch-checkpoints 
else
    echo "ERROR: $model_architecture is an invalid choice of architecture. See the fairseq documentation for the options."
fi

echo ""
echo " * Finished training! * "