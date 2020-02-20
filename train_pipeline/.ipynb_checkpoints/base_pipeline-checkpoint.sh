#!/usr/bin/env bash
# Base seq2seq pipeline using Fairseq
# Example Usage: bash base_pipeline.sh 0 source target true true

curr_dir=${PWD##*/}
export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
export BASE_PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export FAIRSEQ_PATH=$BASE_PROJECT_DIR/fairseq-apr19
export BPE_CODE=$BASE_PROJECT_DIR/subword-nmt

# Device to use for training
export CUDA_DEV=$1
# The input file to train on
src_name=$2
# The target train file
tgt_name=$3
# if set to true, will regenerate the data. False reuses old preprocessed data.
prepare_data=$4
# if set to true, will train/continue training a model. False assumes we already have a trained model:
# we'll just run the generation
train_model=$5
# Architecture to train: choose between lstm or transformer
architecture=$6
# BPE dict SRC (not providing will learn one)
bpe_dict_src=$7
# BPE dict TGT (not providing will learn one)
bpe_dict_tgt=$8
# SRC dict to reuse
src_dict=$9
# TGT dict to reuse
tgt_dict="${10}"

if [ -z "$FAIRSEQ_PATH" ] || [ -z $BPE_CODE ]; then
    echo "You didn't set the FAIRSEQ_PATH or the CNNDM_PATH or the BPE_CODE variables. Check the documentation."
    exit
else
    echo " * Starting base pipeline. * "
    echo ""
    echo " * Environment variables: * "
    echo "FAIRSEQ_PATH=$FAIRSEQ_PATH"
    echo "BPE_CODE=$BPE_CODE"
    echo ""
fi

#if [ "$architecture" == "lstm" ]; then
#    arch=lstm_tiny
#elif [ "$architecture" == "transformer" ]; then
#    arch=transformer_vaswani_wmt_en_de_big
#fi

### Train
if [ "$train_model" = true ] ; then
    bash $SCRIPT_DIR/train-seq2seq.sh \
        $src_name $tgt_name $CUDA_DEV $prepare_data $architecture \
        $bpe_dict_src $bpe_dict_tgt \
        $src_dict $tgt_dict
else
    echo " * Skipping training. * "
    echo ""
fi

if [ ! -f $arch/checkpoint_best.pt ]; then
    echo "ERROR: Can't find the model checkpoint $architecture/checkpoint_best.pt. Training may have failed."
    exit 1
fi

# Generate from the test set
CUDA_VISIBLE_DEVICES=$CUDA_DEV python $FAIRSEQ_PATH/generate_file.py fairseq-data \
    --path $architecture/checkpoint_best.pt --batch-size 100 --beam 5 --output_file output.txt \
    --remove-bpe --quiet --max-len-a 3