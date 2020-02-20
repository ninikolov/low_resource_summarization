
import sys
from gensim.summarization.summarizer import summarize
from nltk.tokenize import ToktokTokenizer
toktok = ToktokTokenizer().tokenize
from tqdm import *

import logging, argparse
logging.getLogger("gensim").setLevel(logging.ERROR)


# -*- coding: utf-8 -*-


import math

from itertools import combinations
from collections import defaultdict
from sumy.summarizers._summarizer import AbstractSummarizer


class TextRankSummarizer(AbstractSummarizer):
    """Source: https://github.com/adamfabish/Reduction"""

    _stop_words = frozenset()

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document, sentences_count):
        ratings = self.rate_sentences(document)
        return self._get_best_sentences(document, sentences_count, ratings)

    def rate_sentences(self, document):
        sentences_words = [(s, self._to_words_set(s)) for s in document]
        # print(sentences_words)
        ratings = defaultdict(float)

        for (sentence1, words1), (sentence2, words2) in combinations(sentences_words, 2):
            rank = self._rate_sentences_edge(words1, words2)
            ratings[sentence1] += rank
            ratings[sentence2] += rank

        return ratings

    def _to_words_set(self, sentence):
        words = map(self.normalize_word, sentence.split())
        return [self.stem_word(w) for w in words if w not in self._stop_words]

    def _rate_sentences_edge(self, words1, words2):
        rank = 0
        for w1 in words1:
            for w2 in words2:
                rank += int(w1 == w2)

        if rank == 0:
            return 0.0

        assert len(words1) > 0 and len(words2) > 0
        norm = math.log(len(words1)) + math.log(len(words2))
        return 0.0 if norm == 0.0 else rank / norm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-articles', help='The file containing the articles to summarize.')
    parser.add_argument('-output', help='The file to write the summaries to.')
    parser.add_argument('-length', type=int, help='The number of sentences to extract.', default=4)
    parser.add_argument('-ratio', type=float, help='The number of sentences to extract.', default=0.05)
    parser.add_argument('-word_count', type=int, help='The number of sentences to extract.', default=100)
    args = parser.parse_args()

    summarizer = TextRankSummarizer()

    with tqdm(total=None, desc="TextRank") as pbar:
        with open(args.articles, "r") as input_file:
            with open(args.output, "w") as output_file:
                for article in input_file:
                    article = article.strip()
                    article = article.split(" <s> ")
                    summary = summarizer(article, args.length)
                    # summary = summarize(article, word_count=args.word_count, split=True)
                    summary = " <s> ".join([str(sent) for sent in summary])
                    output_file.write("{}\n".format(summary.strip()))
                    pbar.update()
