# -*- coding: utf-8 -*-

import sys, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
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
# w2v_path = "/home/nikola/data/raw/wiki/GoogleNews-vectors-negative300.bin"

from bert_serving.client import BertClient


from collections import namedtuple
from sumy.utils import ItemsCount
from operator import attrgetter
import scipy as sp
import sent2vec

SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))


class CentroidSummarizer(AbstractSummarizer):
    """

    """
    threshold = 0.4
    alpha = 0.5
    epsilon = 0.1
    _stop_words = frozenset()

    def __init__(self, *args, **kw):
        super(self.__class__, self).__init__(*args, **kw)

        # self.embedding_model = BertClient(check_length=False)

        self.embedding_model = sent2vec.Sent2vecModel()
        self.embedding_model.load_model('/home/nikola/data/raw/wiki/wiki_unigrams.bin')
        print("Loaded sent2vec model")

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def get_embedding(self, text):
        # return self.embedding_model.encode([text])
        return self.embedding_model.embed_sentence(text)

    def get_embeddings(self, texts):
        # return self.embedding_model.encode(texts)
        return self.embedding_model.embed_sentences(texts)

    def __call__(self, document, sentences_count, order_by_article=True, summary_history=None):
        sentences_words = [self._to_words_set(s) for s in document]
        if not sentences_words:
            return tuple()

        # print(document)
        document_emb = self.get_embedding(" ".join(document))
        # print(document_emb)
        sentence_embs = self.get_embeddings(document)
        assert len(sentence_embs) == len(document)
        scores = [
            1. / (1. + sp.spatial.distance.euclidean(
                sentence_embedding, document_emb))
            if len(sent.split()) > 3 else 0.
            for sentence_embedding, sent in zip(sentence_embs, document)
        ]
        # print(scores)

        ratings = dict(zip(document, scores))

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-articles', help='The file containing the articles to summarize.')
    parser.add_argument('-output', help='The file to write the summaries to.')
    parser.add_argument('-length', type=int, help='The number of sentences to extract.', default=3)
    parser.add_argument('-threshold', type=float, help='The number of sentences to extract.', default=0.4)
    args = parser.parse_args()

    input_file_path = args.articles
    logging.info("Running Centroid baseline on {} extracting {} sentences...".format(
        input_file_path, args.length))

    summarizer = CentroidSummarizer()

    with open(args.articles, "r") as input_file:
        with open(args.output, "w") as output_file:
            with tqdm(total=None, desc="Centroid") as pbar:
                for line in input_file:
                    line = line.strip()
                    # line = " ".join(line.split(" <s> "))
                    document = line.split(" <s> ")
                    # parser = PlaintextParser.from_string(line, Tokenizer("english"))
                    summary, _ = summarizer(document, args.length)
                    # print(summary)
                    summary_txt = " <s> ".join([str(sent) for sent in summary])
                    output_file.write("{}\n".format(summary_txt))
                    pbar.update()
    logging.info("RWMDRank baseline finished!")
