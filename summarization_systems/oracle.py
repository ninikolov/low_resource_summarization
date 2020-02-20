"""Extractive oracle"""



import argparse
import itertools
import logging
import nltk
import numpy as np
import sys
from multiprocessing import Process, Manager, cpu_count
from tqdm import *
from nltk.tokenize import ToktokTokenizer
toktok = ToktokTokenizer().tokenize


def set_overlap(source_set, target_set):
    """Compute the overlap score between a source and a target set.
    It is the intersection of the two sets, divided by the length of the target set."""
    word_overlap = target_set.intersection(source_set)
    overlap = len(word_overlap) / float(len(target_set))
    assert 0. <= overlap <= 1.
    return overlap


def jaccard_similarity(source, target):
    """Compute the jaccard similarity between two texts."""
    if len(source) == 0 or len(target) == 0:
        return 0.
    source_set = set(source)
    target_set = set(target)
    try:
        return set_overlap(source_set, target_set.union(source_set))
    except ZeroDivisionError as e:
        logging.error(e)
        return 0.


def copy_rate(source, target, tokenize=False):
    """
    Compute copy rate

    :param source:
    :param target:
    :return:
    """
    if tokenize:
        source = toktok(source)
        target = toktok(target)
    source_set = set(source)
    target_set = set(target)
    if len(source_set) == 0 or len(target_set) == 0:
        return 0.
    return set_overlap(source_set, target_set)


def repeat_rate(sents):
    """
    Compute the repeat rate of a text

    :param sents:
    :return:
    """
    if len(sents) == 1:
        return 0.
    else:
        repeat_rates = []
        for i, sent in enumerate(sents):
            rest = " ".join([sents[j] for j, s in enumerate(sents) if j != i])
            repeat_rates.append(copy_rate(rest, sent, True))
    return np.mean(repeat_rates)
    

def get_sentence_ranking(sentences, target, metric="copy"):
    target_array = toktok(target)
    similarities = []
    for sent in sentences:
        if metric == "jacc":
            similarities.append(jaccard_similarity(toktok(sent), target_array))
        elif metric == "copy":
            similarities.append(copy_rate(toktok(sent), target_array))
    return np.argsort(similarities)[::-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute the Oracle extractive summary.")
    parser.add_argument('-src', help='The source file', required=True)
    parser.add_argument('-tgt', help='The target file', required=True)
    parser.add_argument('-output', help='The output file to write to', required=True)
    parser.add_argument('-metric', help='The metric to use, default Jaccard similarity', default="copy")
    args = parser.parse_args()

    with open(args.src) as src:
        with open(args.tgt) as tgt:
            output = open(args.output, "w")
            with tqdm(desc="Oracle") as pbar:
                for i, (src_line, tgt_line) in enumerate(zip(src, tgt)):
                    src_sents = src_line.strip().split(" <s> ")
                    tgt_sents = tgt_line.strip().split(" <s> ")
                    optimal_summary = []
                    for summary_sent in tgt_sents:
                        # Get the close
                        selection_order = get_sentence_ranking(src_sents, summary_sent, args.metric)
                        for input_index in selection_order:
                            if input_index not in optimal_summary:
                                optimal_summary.append(input_index)
                                break
                    output.write("{}\n".format(" <s> ".join([src_sents[j] for j in optimal_summary])))
                    pbar.update()
