#!/usr/bin/env bash

target_file=$1
bpe_dict=$2

echo " * Applying BPE encoding to $target_file * "

# Apply BPE to train/valid/test
python $BPE_CODE/apply_bpe.py --input train.$1.clean --output train.$1 --codes $bpe_dict
python $BPE_CODE/apply_bpe.py --input valid.$1.clean --output valid.$1 --codes $bpe_dict
python $BPE_CODE/apply_bpe.py --input test.$1.clean --output test.$1 --codes $bpe_dict
