Code and Data for the paper 'Abstractive Document Summarization Without Parallel Data'. 

<img src="./summarization_approach.png" width="500" >

# Datasets 

In the paper, we use two datasets: 
 * The CNN/DailyMail parallel summarization dataset. Our processed version is available [here](https://drive.google.com/file/d/10qeztf26eVxmHzUX2JwvC3E0F0BsWcg2/view?usp=sharing). 
 * Our test dataset for the scientific summarization task

# Running the system

The pipeline for training the sentence paraphrasing model consists of the following components: 

<img src="./pipeline.png" width="800" >

## Extracting pseudo-parallel data 

Follow the instructions from [this repository](https://github.com/ninikolov/lha) to extract pseudo-parallel data from 
your raw datasets. 

## Training a backtranslation model 

Given a dataset which contains files with the following naming convention: `train.article.clean`, `train.summary.clean`, 
`valid.article.clean`, `valid.summary.clean`, `test.article.clean`, `test.summary.clean`

You can train a backtranslation model on your pseudo-parallel data using the 
following commands: 

```
# Define the path to the BPE and Fairseq dictionaries you will use. 
# If you don't provide them, they will be learned from the data automatically. 
export BPE_DICT=~/nikola/joint/joint50k.bpe
export FAIRSEQ_DICT=~/nikola/joint/data/dict.clean.txt
# Start the pipeline
bash train_pipeline/base_pipeline.sh 1 summary article true true lstm $BPE_DICT $BPE_DICT $FAIRSEQ_DICT $FAIRSEQ_DICT
```

The command will take care of dataset preparation, conversion to BPE and training. Read
the scripts for more info on the specific commands.  

## Generating synthetic sentences using the backtranslation model

Once the backtranslation model is trained, you can use it to synthesize additional 
source sentences. This can be done using the following script: 

```
bash main/backtranslate_summary_sentences.sh train.summary.clean 2
```

The script expects one sentence per line. 

## Training the final paraphrasing model 

Once you have all your data prepared, you can train your final paraphrasing model using 
the same training script, but by setting the source and target to be your 
article and summary sentences: 

```
bash train_pipeline/base_pipeline.sh 1 article summary true true lstm $BPE_DICT $BPE_DICT $FAIRSEQ_DICT $FAIRSEQ_DICT
```

## Running the whole system   

Once the final paraphrasing model is trained, you can run the whole extractive-abstractive 
pipeline using the following command: 

```
bash main/paraphrase-with-lm.sh valid.paper.clean 2 0. 10 1 lead False True 
```

The above will first apply the Lead extractor to your article, and will then
paraphrase each of the extracted sentences. 
You'll need to modify some variables in the `paraphrase-with-lm.sh` script 
to point to the final model files of your paraphrasing model. 

# Citation 

```
@InProceedings{nikolov2020abstractive,
  author = {Nikola I. Nikolov and Richard Hahnloser},
  title = {Abstractive Document Summarization without Parallel Data},
  booktitle = {Proceedings of the Twelfth International Conference on Language Resources and Evaluation (LREC 2020)},
  year = {2020},
  month = {may},
  date = {11-16},
  location = {Marseille, France},
  editor = {},
  publisher = {European Language Resources Association (ELRA)},
  address = {Paris, France},
  language = {english}
  }
```