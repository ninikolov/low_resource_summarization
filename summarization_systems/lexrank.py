""""""

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import sys, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.summarization.summarizer import summarize
from nltk.tokenize import ToktokTokenizer
toktok = ToktokTokenizer().tokenize
import nltk
from tqdm import *

import math

try:
    import numpy
except ImportError:
    numpy = None

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers._summarizer import AbstractSummarizer
from sumy._compat import Counter

from collections import namedtuple
from sumy.utils import ItemsCount
from operator import attrgetter

SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))


class LexRankSummarizer(AbstractSummarizer):
    """
    LexRank: Graph-based Centrality as Salience in Text Summarization
    Source: http://tangra.si.umich.edu/~radev/lexrank/lexrank.pdf
    """
    threshold = 0.16
    epsilon = 0.1
    _stop_words = frozenset()

    def __init__(self, threshold=0.16, *args, **kw):
        super(self.__class__, self).__init__(*args, **kw)
        self.threshold = threshold
        print("Starting LexRank with th={}".format(self.threshold))

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document, sentences_count, order_by_article=True):
        self._ensure_dependencies_installed()

        sentences_words = [self._to_words_set(s) for s in document]
        if not sentences_words:
            return tuple()

        tf_metrics = self._compute_tf(sentences_words)
        idf_metrics = self._compute_idf(sentences_words)

        matrix = self._create_matrix(sentences_words, self.threshold, tf_metrics, idf_metrics)
        scores = self.power_method(matrix, self.epsilon)
        ratings = dict(zip(document, scores))

        summary, infos = self._get_best_sentences(document, sentences_count,
                                                  ratings, order_by_article)

        return summary, infos

    def _get_best_sentences(self, sentences, count, rating, order_by_article=True,
                            *args, **kwargs):
        rate = rating
        if isinstance(rating, dict):
            assert not args and not kwargs
            rate = lambda s: rating[s]

        infos = (SentenceInfo(s, o, rate(s, *args, **kwargs))
            for o, s in enumerate(sentences))

        # sort sentences by rating in descending order
        infos = sorted(infos, key=attrgetter("rating"), reverse=True)
        # get `count` first best rated sentences
        if not isinstance(count, ItemsCount):
            count = ItemsCount(count)
        infos = count(infos)
        # sort sentences by their order in document
        if order_by_article:
            infos = sorted(infos, key=attrgetter("order"))

        return tuple(i.sentence for i in infos), infos

    @staticmethod
    def _ensure_dependencies_installed():
        if numpy is None:
            raise ValueError("LexRank summarizer requires NumPy. Please, install it by command 'pip install numpy'.")

    def _to_words_set(self, sentence):
        words = map(self.normalize_word, sentence.split())
        return [self.stem_word(w) for w in words if w not in self._stop_words]

    def _compute_tf(self, sentences):
        tf_values = map(Counter, sentences)

        tf_metrics = []
        for sentence in tf_values:
            metrics = {}
            max_tf = self._find_tf_max(sentence)

            for term, tf in sentence.items():
                metrics[term] = tf / max_tf

            tf_metrics.append(metrics)

        return tf_metrics

    @staticmethod
    def _find_tf_max(terms):
        return max(terms.values()) if terms else 1

    @staticmethod
    def _compute_idf(sentences):
        idf_metrics = {}
        sentences_count = len(sentences)

        for sentence in sentences:
            for term in sentence:
                if term not in idf_metrics:
                    n_j = sum(1 for s in sentences if term in s)
                    idf_metrics[term] = math.log(sentences_count / (1 + n_j))

        return idf_metrics

    def _create_matrix(self, sentences, threshold, tf_metrics, idf_metrics):
        """
        Creates matrix of shape |sentences|×|sentences|.
        """
        # create matrix |sentences|×|sentences| filled with zeroes
        sentences_count = len(sentences)
        matrix = numpy.zeros((sentences_count, sentences_count))
        degrees = numpy.zeros((sentences_count, ))

        for row, (sentence1, tf1) in enumerate(zip(sentences, tf_metrics)):
            for col, (sentence2, tf2) in enumerate(zip(sentences, tf_metrics)):
                matrix[row, col] = self.cosine_similarity(sentence1, sentence2, tf1, tf2, idf_metrics)

                if matrix[row, col] > threshold:
                    matrix[row, col] = 1.0
                    degrees[row] += 1
                else:
                    matrix[row, col] = 0

        for row in range(sentences_count):
            for col in range(sentences_count):
                if degrees[row] == 0:
                    degrees[row] = 1

                matrix[row][col] = matrix[row][col] / degrees[row]

        return matrix

    @staticmethod
    def cosine_similarity(sentence1, sentence2, tf1, tf2, idf_metrics):
        """
        We compute idf-modified-cosine(sentence1, sentence2) here.
        It's cosine similarity of these two sentences (vectors) A, B computed as cos(x, y) = A . B / (|A| . |B|)
        Sentences are represented as vector TF*IDF metrics.

        :param sentence1:
            Iterable object where every item represents word of 1st sentence.
        :param sentence2:
            Iterable object where every item represents word of 2nd sentence.
        :type tf1: dict
        :param tf1:
            Term frequencies of words from 1st sentence.
        :type tf2: dict
        :param tf2:
            Term frequencies of words from 2nd sentence
        :type idf_metrics: dict
        :param idf_metrics:
            Inverted document metrics of the sentences. Every sentence is treated as document for this algorithm.
        :rtype: float
        :return:
            Returns -1.0 for opposite similarity, 1.0 for the same sentence and zero for no similarity between sentences.
        """
        unique_words1 = frozenset(sentence1)
        unique_words2 = frozenset(sentence2)
        common_words = unique_words1 & unique_words2

        numerator = 0.0
        for term in common_words:
            numerator += tf1[term]*tf2[term] * idf_metrics[term]**2

        denominator1 = sum((tf1[t]*idf_metrics[t])**2 for t in unique_words1)
        denominator2 = sum((tf2[t]*idf_metrics[t])**2 for t in unique_words2)

        if denominator1 > 0 and denominator2 > 0:
            return numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
        else:
            return 0.0

    @staticmethod
    def power_method(matrix, epsilon):
        transposed_matrix = matrix.T
        sentences_count = len(matrix)
        p_vector = numpy.array([1.0 / sentences_count] * sentences_count)
        lambda_val = 1.0

        while lambda_val > epsilon:
            next_p = numpy.dot(transposed_matrix, p_vector)
            lambda_val = numpy.linalg.norm(numpy.subtract(next_p, p_vector))
            p_vector = next_p

        return p_vector


def summary_length(summary_sents):
    return sum([len(str(s).split()) for s in summary_sents])

import random

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-articles', help='The file containing the articles to summarize.')
    parser.add_argument('-output', help='The file to write the summaries to.')
    parser.add_argument('-length', type=int, help='The number of sentences to extract.', default=4)
    parser.add_argument('-n_tokens', help='Stop adding sentences to summary after this number of tokens.',
                        default=-1, type=int)
    parser.add_argument('-sentence_separator', default=" <s> ", help='The sentence separator to expect.')
    parser.add_argument('-threshold', type=float, help='The threshold.', default=0.16)
    parser.add_argument('-tuning', type=bool, help='The threshold.', default=False)
    args = parser.parse_args()

    logging.info("Running LexRank baseline on {} extracting {} sentences...".format(
        args.articles, args.length))
    print("Tuning: {}".format(args.tuning))
    summarizer = LexRankSummarizer(threshold=args.threshold)
    summaries = open(args.output, "w")

    with open(args.articles, "r") as input_file:
        with tqdm(total=None, desc="LexRank") as pbar:
            for line in input_file:
                line = line.strip()
                if args.sentence_separator not in ["None", None]:
                    sents = line.split(args.sentence_separator)
                    if args.tuning:
                        random.shuffle(sents)
                    # line = " ".join(sents)
                else:
                    args.sentence_separator = " "
                # parser = PlaintextParser.from_string(line, Tokenizer("english"))
                if args.n_tokens > 0:
                    summary = []
                    summary_sents, _ = summarizer(sents, args.length, False)
                    for sent in summary_sents:
                        if summary_length(summary) > args.n_tokens:
                            break
                        summary.append(sent)
                else:
                    summary, _ = summarizer(sents, args.length)

                summaries.write("{}\n".format(
                    args.sentence_separator.join([str(sent) for sent in summary])))
                pbar.update()
    logging.info("LexRank baseline finished!")
