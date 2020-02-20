# -*- coding: utf-8 -*-

import sys, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.getLogger("gensim").setLevel(logging.ERROR)
from nltk.tokenize import ToktokTokenizer
toktok = ToktokTokenizer().tokenize
from tqdm import *
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from sumy.summarizers._summarizer import AbstractSummarizer
from nltk.corpus import stopwords

import string
stop = stopwords.words('english') + list(string.punctuation)
import argparse

# w2v_path = "/home/nikola/data/raw/ncbi/plaintext/bio_nlp_vec/PubMed-shuffle-win-30.bin"
w2v_path = "/home/nikola/data/raw/wiki/GoogleNews-vectors-negative300.bin"


def get_word_vector_list(doc, w2v):
    """Get all the vectors for a text"""
    vectors = []
    for word in doc:
        try:
            vectors.append(w2v.wv[word])
        except KeyError:
            continue
    return vectors


def relaxed_wmd(doc_1, doc_2, w2v, distance_matrix=None, normed=False,
                doc_1_vectors=None, doc_2_vectors=None, use_distance=False):
    """
    Compute the Relaxed Word Mover's Distance score between doc_1 and doc_2.
    See http://proceedings.mlr.press/v37/kusnerb15.pdf for more info.

    :param doc_1:
    :param doc_2:
    :param w2v: the word2vec model to use
    :param distance_matrix:
    :param doc_1_vectors:
    :param doc_2_vectors:
    :param use_distance: if true, will use the euclidean distance; otherwise the cosine similarity
    :return:
    """
    # Compute the distance/similarity matrix between the documents
    if distance_matrix is None:
        if doc_1_vectors is None:
            doc_1_vectors = np.array(get_word_vector_list(doc_1, w2v), dtype=float)
        if doc_2_vectors is None:
            doc_2_vectors = np.array(get_word_vector_list(doc_2, w2v), dtype=float)
        if len(doc_1_vectors) == 0 or len(doc_2_vectors) == 0:
            if use_distance:
                return np.inf
            else:
                return 0.
        if use_distance:
            distance_matrix = cdist(doc_1_vectors, doc_2_vectors, "euclidean")
        else:
            distance_matrix = cosine_similarity(doc_1_vectors, doc_2_vectors)
            # Round and clip similarity matrix to make sure it is between 0 and 1
            distance_matrix = np.round(distance_matrix, 5)
            distance_matrix = np.nan_to_num(distance_matrix)
            distance_matrix[distance_matrix == np.inf] = 0.
            distance_matrix = distance_matrix.clip(min=0.)

    if use_distance:
        score = np.mean(np.min(distance_matrix, 1))
        return 1. / (1. + score)
    else:
        return np.mean(np.max(distance_matrix, 1))


def relaxed_wmd_combined(doc_1, doc_2, w2v, normed=False, distance=False,
                         combination="mean", return_parts=False):
    """


    :param doc_1:
    :param doc_2:
    :param w2v:
    :param normed:
    :param distance:
    :param combination:
    :return:
    """
    doc_1_vectors = np.array(get_word_vector_list(doc_1, w2v), dtype=float)
    doc_2_vectors = np.array(get_word_vector_list(doc_2, w2v), dtype=float)

    if len(doc_1_vectors) == 0 or len(doc_2_vectors) == 0:
        if distance:
            return np.inf if not return_parts else np.inf, np.inf, np.inf
        else:
            return 0. #if not return_parts else 0., 0., 0.
            # return 0. if not return_parts else 0., 0., 0.
    if distance:
        D = cdist(doc_1_vectors, doc_2_vectors, "euclidean")
    else:
        D = cosine_similarity(doc_1_vectors, doc_2_vectors)
        D = np.round(D, 5)
        D = np.nan_to_num(D)
        D[D == np.inf] = 0.
        D = D.clip(min=0.)

    # Compute left and right RWMD
    l1 = relaxed_wmd(doc_1, doc_2, w2v, distance_matrix=D, normed=normed, use_distance=distance)
    l2 = relaxed_wmd(doc_2, doc_1, w2v, distance_matrix=D.T, normed=normed, use_distance=distance)

    # Combine the two RWMD approximations
    if combination == "mean":
        combined = np.mean([l1, l2])
    elif combination == "min":
        combined = np.min([l1, l2])
    elif combination == "max":
        combined = np.max([l1, l2])

    if distance:
        combined = 1. / (1. + combined)

    if return_parts:
        return combined, l1, l2
    else:
        return combined

from collections import namedtuple
from sumy.utils import ItemsCount
from operator import attrgetter

SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))


class RWMDRankSummarizer(AbstractSummarizer):
    """

    """
    threshold = 0.45
    epsilon = 0.1
    _stop_words = frozenset()

    def __init__(self, w2v_model_path="~/data/raw/wiki/GoogleNews-vectors-negative300.bin",
                 threshold=0.45,
                 *args, **kw):
        super(self.__class__, self).__init__(*args, **kw)

        # Load word2vec model
        w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        w2v.syn0norm = w2v.syn0
        logging.warning("Loaded Word2Vec model {} with embedding dim {} and vocab size {}".format(
            w2v_path, w2v.vector_size, len(w2v.syn0norm)))

        self.w2v_model = w2v
        self.threshold = threshold
        self.max_sim = 0.

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document, sentences_count, order_by_article=True, summary_history=None):
        sentences_words = [self._to_words_set(s) for s in document]
        if not sentences_words:
            return tuple()

        matrix = self._create_matrix(sentences_words, self.threshold, summary_history)
        scores = self.power_method(matrix, self.epsilon)
        ratings = dict(zip(document, scores))
        # print(ratings.values())

        summary, infos = self._get_best_sentences(document, sentences_count,
                                                  ratings, order_by_article)
        # print(infos)
        return summary, infos

    def _get_best_sentences(self, sentences, count, rating, order_by_article=True, *args, **kwargs):
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

    def _to_words_set(self, sentence):
        # Skip stemming for RWMDRank
        return [w for w in sentence.split() if w not in stop]

    def _create_matrix(self, sentences, threshold, summary_history=[]):
        """
        Creates matrix of shape |sentences|×|sentences|.
        """
        # create matrix |sentences|×|sentences| filled with zeroes
        sentences_count = len(sentences)
        matrix = np.zeros((sentences_count, sentences_count))
        degrees = np.zeros((sentences_count, ))

        for row, sentence1 in enumerate(sentences):
            for col, sentence2 in enumerate(sentences):
                rwmd_score = relaxed_wmd_combined(sentence1, sentence2, self.w2v_model)
                # rwmd_score = 1. / (1. + self.w2v_model.wmdistance(sentence1, sentence2))
                # print(rwmd_score)
                matrix[row, col] = rwmd_score
                if rwmd_score != 1. and rwmd_score > self.max_sim:
                    self.max_sim = rwmd_score
                    # print(self.max_sim)

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
    def power_method(matrix, epsilon):
        transposed_matrix = matrix.T
        sentences_count = len(matrix)
        p_vector = np.array([1.0 / sentences_count] * sentences_count)
        lambda_val = 1.0

        while lambda_val > epsilon:
            next_p = np.dot(transposed_matrix, p_vector)
            lambda_val = np.linalg.norm(np.subtract(next_p, p_vector))
            p_vector = next_p

        return p_vector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-articles', help='The file containing the articles to summarize.')
    parser.add_argument('-output', help='The file to write the summaries to.')
    parser.add_argument('-length', type=int, help='The number of sentences to extract.', default=4)
    parser.add_argument('-threshold', type=float, help='The threshold.', default=0.45)
    parser.add_argument('-tuning', type=bool, help='The threshold.', default=False)
    args = parser.parse_args()

    import random

    input_file_path = args.articles
    logging.info("Running RWMDRank baseline on {} extracting {} sentences...".format(
        input_file_path, args.length))
    logging.info("Threshold: {}".format(args.threshold))

    summarizer = RWMDRankSummarizer(threshold=args.threshold)

    with open(args.articles, "r") as input_file:
        with open(args.output, "w") as output_file:
            with tqdm(total=None, desc="RWMDRank") as pbar:
                for line in input_file:
                    line = line.strip()
                    line = line.split(" <s> ")
                    if args.tuning:
                        random.shuffle(line)
                    # parser = PlaintextParser.from_string(line, Tokenizer("english"))
                    summary, _ = summarizer(line, args.length)
                    # print(summary)
                    summary_txt = " <s> ".join([str(sent) for sent in summary])
                    output_file.write("{}\n".format(summary_txt))
                    pbar.update()
    logging.info("RWMDRank baseline finished!")
