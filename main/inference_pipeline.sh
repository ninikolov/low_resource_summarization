#!/usr/bin/env bash
# Run the summarization pipeline: extraction followed by sentence paraphrasing.

# The input file to summarize, containing one article per line
input_file=$1

# The GPU id to use
if [ -z "$2" ] ; then
    gpu=1
else
    gpu=$2
fi

# The length of the output summary
if [ -z "$3" ] ; then
    summary_length=-1
else
    summary_length=$3
fi

# The extractive summarization approach to use. Choose between lead and lexrank.
if [ -z "$4" ] ; then
    extractive_approach=""
else
    extractive_approach="--extractive_approach $4"
fi

# Size of buffer during decoding
if  [ -z "$5" ] ; then
    buff_size=1
else
    buff_size=$5
fi

curr_dir=${PWD##*/}
export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
export BASE_PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export FAIRSEQ_PATH=$BASE_PROJECT_DIR/fairseq-apr19
export BPE_CODE=$BASE_PROJECT_DIR/subword-nmt

# CNNDM joint BPE path
bpe_code_path=~/PhD/code/subword-nmt/
# The link to the BPE codes to use. Update accordingly.
bpe_codes=~/data/raw/cnn_dailymail/finished_files/flat_dataset/train.bpe

# The fairseq-data folder of the paraphrase model
paraphrase_data=/home/nikola/data/raw/cnn_dailymail/unsupervised-summarization/paraphrase-model-combined+backtranslation+sampling/fairseq-data
# The checkpoint of the paraphrase model
paraphrase_model=/home/nikola/data/raw/cnn_dailymail/unsupervised-summarization/paraphrase-model-combined+backtranslation+sampling/lstm_tiny/checkpoint_best.pt

# Science source BPE path
bpe_code_path=$BASE_PROJECT_DIR/subword-nmt/
bpe_codes=/home/nikola/data/raw/science-journalism/lha-paper-data/paraphrase-model/train.paper.clean.bpe

# The root dir of the paraphrase model (trained using fairseq)
root=/home/nikola/data/raw/science-journalism/lha-paper-data/paraphrase-model+backtranslation

paraphrase_data=$root/fairseq-data
paraphrase_model=$root/lstm_tiny/checkpoint_best.pt

base_input_file=`basename $input_file`

sentence_batch_size=150

echo "$FAIRSEQ_PATH"

# Apply BPE, run summarization pipeline line by line
cat $input_file | python $bpe_code_path/apply_bpe.py --codes $bpe_codes | \
    CUDA_VISIBLE_DEVICES=$gpu python $FAIRSEQ_PATH/paraphrase-rescore-batch.py $paraphrase_data \
    --path $paraphrase_model \
    --output_file $base_input_file.par.buff=$buff_size.nbest=$n_best_reranking \
    -s paper -t press \
    --beam 5 --nbest 5 --no-beamable-mm --no-repeat-ngram-size 3 \
    --max-sentences $sentence_batch_size --buffer-size $buff_size \
    --num_output_sentences $summary_length \
    $extractive_approach

