"""Analyze a text file"""

from tqdm import *
import nltk, sys
import numpy as np
from multiprocessing import Process, Manager, cpu_count
import itertools
import linecache
# from llama.tools.readability import dc_score, fk_score, f_score

from nltk.tokenize import ToktokTokenizer
toktok = ToktokTokenizer().tokenize


def get_stat(in_queue, out_list):
    while True:
        line_no, line = in_queue.get()
        if line is None:
            return
        line = line.strip()
        if args.sentence_separator is not None:
            sents = line.split(args.sentence_separator)
        else:
            sents = nltk.sent_tokenize(line)
        sent_count = len(sents)
        token_count = len(line.split(" "))
        char_count = len(line)
        char_per_word_count = np.array([len(word) for word in line.split(" ")])
        sent_token_count = [len(toktok(s)) for s in sents]
#         doc_dc_score = dc_score(line)
#         doc_fk_score = fk_score(line)
        out_list.append((
            sent_count, token_count, sent_token_count, 
#             doc_dc_score, doc_fk_score, 
            char_count, char_per_word_count))


import argparse
parser = argparse.ArgumentParser(description="Analyze an input text file, computing statistics.")
parser.add_argument('-target_file', help='The text file to process', required=True)
parser.add_argument('-limit', help='Limit processing to this number of lines.', default=None, type=int)
parser.add_argument('-simple', help='', default=True, type=bool)
parser.add_argument('-sentence_separator', help='The sentence separator to use. By default, use NLTK.', default=None)
args = parser.parse_args()


if __name__ == '__main__':
    manager = Manager()
    results = manager.list()
    # Use half of available CPUs
    num_workers = int(cpu_count()/2)
    work = manager.Queue(num_workers)

    if args.sentence_separator is not None:
        args.sentence_separator = str(args.sentence_separator)

    pool = []
    for i in range(num_workers):
        p = Process(target=get_stat, args=(work, results))
        p.start()
        pool.append(p)

    print("Computing stat on {}".format(args.target_file))

    with tqdm(desc="Counting total sentences") as pbar:
        with open(args.target_file) as f:
            if args.limit is not None:
                file_iter = itertools.islice(f, args.limit)
            else:
                file_iter = iter(f)
            iters = itertools.chain(file_iter, (None,) * num_workers)

            for id_pair in enumerate(iters):
                work.put(id_pair)
                pbar.update()

    for p in pool:
        p.join()

    sent_counts = [r[0] for r in results]
    unique, counts = np.unique(sent_counts, return_counts=True)
    sentences_per_line = dict(zip(unique, counts))

    tok_counts = [r[1] for r in results]
    sent_tok_counts = []
    for r in results:
        sent_tok_counts += r[2]
#     dc_scores = [r[3] for r in results]
#     fk_scores = [r[4] for r in results]
    char_counts = [r[3] for r in results]
    # char_per_word_counts = np.array([r[5] for r in results]).flatten()
    char_per_word_counts = [r[4] for r in results]
    char_per_word_counts = [c for s in char_per_word_counts for c in s]

    total_lines = len(results)

    if args.simple: 
        print("Sentences per doc: {} ({})".format(np.round(np.mean(sent_counts), 3), np.round(np.std(sent_counts), 3)))
        print("Mean token count per sentence: {} ({})".format(np.round(np.mean(sent_tok_counts), 2),
                                                        np.round(np.std(sent_tok_counts), 2)))
        print("Mean tokens per doc: {} ({})".format(np.round(np.mean(tok_counts),2), np.round(np.std(tok_counts), 2)))
        print("{} ({})\t{} ({})\t{} ({})\t".format(
            np.round(np.mean(sent_counts), 3), np.round(np.std(sent_counts), 3),
            np.round(np.mean(sent_tok_counts), 2), np.round(np.std(sent_tok_counts), 2),
            np.round(np.mean(tok_counts), 2), np.round(np.std(tok_counts), 2)
        ))
    else:
        print("-"*30)
        print("Total lines: {}".format(total_lines))
        print("-"*30)
        print("Total sentences in {}: {} ".format(args.target_file, np.sum(sent_counts)))
        print("Sentences per doc: {} ({})".format(np.round(np.mean(sent_counts), 3), np.round(np.std(sent_counts), 3)))
        print("Sentence counts: {}".format(sentences_per_line))

        # count_1 = sentences_per_line[1]
        # del sentences_per_line[1]
        # count_multi = sum(sentences_per_line.values())
        # print("Multi >1 {}%".format(np.round((count_multi/total_lines) * 100, 3)))
        # del sentences_per_line[2]
        # count_multi = sum(sentences_per_line.values())
        # print("Multi >2 {}%".format(np.round((count_multi / total_lines) * 100, 3)))

        print("-"*30)
        print("Total tokens in {}: {} ".format(args.target_file, np.sum(tok_counts)))
        print("Mean tokens per doc: {} ({})".format(np.round(np.mean(tok_counts),2), np.round(np.std(tok_counts), 2)))
        print("-"*30)
        print("Mean characters per doc: {} ({})".format(np.mean(char_counts), np.std(char_counts)))
        print("Mean characters per word: {} ({})".format(np.mean(char_per_word_counts), np.std(char_per_word_counts)))
        print("-"*30)
        print("Mean token count per sentence: {} ({})".format(np.round(np.mean(sent_tok_counts), 2),
                                                            np.round(np.std(sent_tok_counts), 2)))
#         print("Mean DC score per doc: {} ({})".format(np.mean(dc_scores), np.std(dc_scores)))
#         print("Mean FK score per doc: {} ({})".format(np.round(np.mean(fk_scores),2), np.round(np.std(fk_scores),2)))
        print("-"*30)