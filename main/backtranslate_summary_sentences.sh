#!/usr/bin/env bash
# Backtranslate a file line by line


export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
export BASE_PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export FAIRSEQ_PATH=$BASE_PROJECT_DIR/fairseq-apr19
export BPE_CODE=$BASE_PROJECT_DIR/subword-nmt

# The file to backtranslate
input_file=$1
# GPU id to use
gpu=$2

#### Model details -- modify accordingly
model_path=data/backtranslation-model/lstm/checkpoint_best.pt
bpe_code_path=data/backtranslation-model/train.press.clean.bpe
src=press
tgt=paper
####


# Apply BPE, translate input file line by line
cat $input_file | python $BPE_CODE/apply_bpe.py --codes $bpe_code_path | \
    CUDA_VISIBLE_DEVICES=$gpu python $FAIRSEQ_PATH/interactive_file.py $model_path/fairseq-data \
    --path $model_path --output_file $input_file.back.src --output_file_src $input_file.back.tgt \
    --nbest 1 --beam 5 --remove-bpe -s $src -t $tgt \
    --buffer-size 50000 --max-tokens 10000 --no-repeat-ngram-size 3 #--sampling
