"""
Lead baseline -- extract first N sentences of article.
"""
import sys, string, re, logging, argparse
import nltk

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


parser = argparse.ArgumentParser(description="LEAD summarizer: extract first N sentences of input article.")
parser.add_argument('-input', help='The input file to summarize', required=True)
parser.add_argument('-output', help='The output file to write the sumamaries to.', required=True)
parser.add_argument('-n_lines', help='Number of lines to extract', default=3, type=int)
parser.add_argument('-n_tokens', help='Number of lines to extract', default=-1, type=int)
parser.add_argument('-sentence_separator', help='The sentence separator', default=" <s> ")
args = parser.parse_args()

from sumy.summarizers._summarizer import AbstractSummarizer


class LeadSummarizer(AbstractSummarizer):
    """Summarizer that picks first k sentences of the article."""

    def __call__(self, document, sentences_count):
        sentences = document.sentences
        return sentences[:sentences_count]


def summary_length(summary_sents):
    return sum([len(s.split()) for s in summary_sents])


if __name__ == '__main__':
    logging.info("Running Lead baseline on {} extracting {} lines/{} tokens...".format(
        args.input, args.n_lines, args.n_tokens))
    with open(args.input, "r") as input_file:
        with open(args.output, "w") as output_file:
            for line in input_file:
                line = line.strip()
                if args.sentence_separator in [None, "None"]:
                    sentences = nltk.sent_tokenize(line)
                else:
                    sentences = line.split(args.sentence_separator)
                if args.n_tokens > 0:
                    selected_sentences = []
                    for sent in sentences:
                        if summary_length(selected_sentences) > args.n_tokens:
                            break
                        selected_sentences.append(sent)
                else:
                    selected_sentences = sentences[:args.n_lines]
                summary = " <s> ".join(selected_sentences)
                output_file.write("{}\n".format(summary))
    logging.info("Finished!")
