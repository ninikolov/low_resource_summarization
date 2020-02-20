#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput
from tqdm import *

import torch
import numpy as np

from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_scorer import SequenceScorer
from fairseq.utils import import_user_module
from fairseq.data.data_utils import process_bpe_symbol
from fairseq.data import data_utils
import logging
from summary_rewriting.summarization_systems.rwmdrank import RWMDRankSummarizer
from summary_rewriting.summarization_systems.lexrank import LexRankSummarizer
# from sumy.summarizers.lex_rank import LexRankSummarizer


Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def set_overlap(source_set, target_set):
    """Compute the overlap score between a source and a target set.
    It is the intersection of the two sets, divided by the length of the target set."""
    word_overlap = target_set.intersection(source_set)
    overlap = len(word_overlap) / float(len(target_set))
    assert 0. <= overlap <= 1.
    return overlap


def copy_rate(source, target):
    """
    Compute copy rate

    :param source:
    :param target:
    :param stem: whether to perform stemming using nltk
    :return:
    """
    source_set = set(source.split())
    target_set = set(target.split())
    if len(source_set) == 0 or len(target_set) == 0:
        return 0.
    return set_overlap(source_set, target_set)


def jaccard_similarity(source, target):
    """Compute the jaccard similarity between two texts."""
    if type(source) == str:
        source = source.split()
    if type(target) == str:
        target = target.split()
    if len(source) == 0 or len(target) == 0:
        return 0.
    source_set = set(source)
    target_set = set(target)
    try:
        return set_overlap(source_set, target_set.union(source_set))
    except ZeroDivisionError as e:
        logging.error(e)
        return 0.


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions):
    tokens = [
        task.source_dictionary.encode_line(src_str, add_if_not_exist=False).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'],
            src_lengths=batch['net_input']['src_lengths'],
        )


def make_history_batches(lines, history, args, task, max_positions):
    sentence_tokens = [
        task.source_dictionary.encode_line(src_str, add_if_not_exist=False).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in sentence_tokens])
    history_tokens = [
        task.source_dictionary.encode_line(src_str, add_if_not_exist=False).long()
        for src_str in history
    ]
    article_itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(sentence_tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)

    history_itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(history_tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)

    for src_batch, history_batch in zip(article_itr, history_itr):
        yield Batch(
            ids=src_batch['id'],
            src_tokens=src_batch['net_input']['src_tokens'],
            src_lengths=src_batch['net_input']['src_lengths'],
        ), Batch(
            ids=history_batch['id'],
            src_tokens=history_batch['net_input']['src_tokens'],
            src_lengths=history_batch['net_input']['src_lengths'],
        )


def main(args):
    import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    if args.detokenize:
        from nltk.tokenize.moses import MosesDetokenizer
        detokenizer = MosesDetokenizer()

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    # assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
    #     '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    print("\n * Summarization args: * \nAlpha: {} \tSentences per summary: {} Max tokens per summary: {} " \
          "\tRescore nbest: {}\n Paraphrasing + Extract: {}".format(
        args.alpha, "all" if args.num_output_sentences == -1 else args.num_output_sentences,
        args.max_summary_tokens, args.rescore_nbest, args.paraphrase_extract))
    print("Article buffer size: {} \tSentence batch size: {} \n".format(
        args.buffer_size, args.max_sentences))

    if args.num_output_sentences > 0:
        if args.extractive_approach == "lexrank":
            print(" * Using LexRank extractive summarizer * \n")
            extractor = LexRankSummarizer()
        elif args.extractive_approach == "rwmdrank":
            print(" * Using RWMDRank extractive summarizer * \n")
            extractor = RWMDRankSummarizer()
        elif args.extractive_approach == "lead":
            print(" * Using Lead extractive summarizer * \n")
        else:
            logging.error(" * Wrong extractive summarizer name. * ")
            raise Exception()

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = utils.load_ensemble_for_inference(
        args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides),
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # if len(models) == 2:
    #     paraphrase_models = models
    # else:
    #     paraphrase_models = models[:2]

    # Initialize generator
    generator = task.build_generator(args)
    scorer = SequenceScorer(task.target_dictionary)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in [models[0]]]
    )

    # if args.buffer_size > 1:
    #     print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    start_id = 0

    if args.output_file is not None:
        if args.extractive_only:
            args.output_file = "{}.extract-only={}.out_sent={}".format(
                args.output_file, args.extractive_approach, args.num_output_sentences)
        else:
            args.output_file = "{}.ext={}.out_sent={}.max_tok={}.alpha={}.rescore={}.par_ext={}".format(
                args.output_file, args.extractive_approach, args.num_output_sentences,
                args.max_summary_tokens, args.alpha, args.rescore_nbest, args.paraphrase_extract)
        out_file = open(args.output_file, "w")
        print('| Writing output to: {}'.format(args.output_file))

    selection_stat = {}

    def make_history(history_sents, articles_sents, multiplier=1):
        articles_history = []
        for hist, art in zip(history_sents, articles_sents):
            history_str = "" if len(hist) == 0 else "<s> " + " <s> ".join(hist)
            # history_str = "<s> " + " <s> ".join(hist) + " <s>" if len(hist) > 0 else "<s>"
            articles_history.append([history_str for i in range(len(art) * multiplier)])
        return articles_history

    def generate_paraphrases(paraphrase_models, article_sentences, history_sents, return_n_best=1):
        # Generate the batch of sentences to paraphrase
        sentence_batch_flat = [sent for article in article_sentences for sent in article]
        # Generate the history batch
        history = make_history(history_sents, article_sentences)
        history_batch_flat = [sent for article in history for sent in article]

        combined_batch = ["<n> {} {}".format(sent, hist).strip()
            for sent, hist in zip(sentence_batch_flat, history_batch_flat)]
        print(combined_batch)

        sentence_batch_iter = make_batches(
            combined_batch, args, task, max_positions)

        results = []
        input_idx = 0
        # for sentence_batch, history_batch in zip(sentence_batch_iter, history_batch_iter):
        for sentence_batch in sentence_batch_iter:
            src_tokens = sentence_batch.src_tokens
            src_lengths = sentence_batch.src_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths
                }
            }

            # Generate the next abstractive sentences
            paraphrase_predictions = task.inference_step(
                generator, paraphrase_models, sample)

            input_idx += args.max_sentences

            for i, (id, hypos) in enumerate(zip(sentence_batch.ids.tolist(), paraphrase_predictions)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

        sentence_paraphrases = []
        sentence_paraphrases_clean = []
        scores = []

        # sort output to match input order
        for i, (id, src_tokens, hypos) in enumerate(sorted(results, key=lambda x: x[0])):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)

            # Process top predictions
            for hypo in hypos[:min(len(hypos), return_n_best)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )
                # print(hypo)
                # score = hypo["lm_score"].item()
                score = hypo["score"]
                sentence_paraphrases.append(hypo_str)
                # Remove BPE
                sentence_paraphrases_clean.append(process_bpe_symbol(hypo_str, "@@ "))
                scores.append(score)

        # Split the paraphrases per article
        curr_idx = 0
        out_paraphrases = []
        out_paraphrases_clean = []
        out_scores = []
        for article in article_sentences:
            out_paraphrases.append(
                sentence_paraphrases[curr_idx:curr_idx + len(article) * return_n_best])
            out_paraphrases_clean.append(
                sentence_paraphrases_clean[curr_idx:curr_idx + len(article) * return_n_best])
            out_scores.append(
                scores[curr_idx:curr_idx + len(article) * return_n_best])
            curr_idx += len(article) * return_n_best

        return out_paraphrases, out_paraphrases_clean, out_scores

    def extractive_summarization(article, length):
        assert type(article) == list
        if args.extractive_approach == "lead":
            return article[:length], list(range(length))

        import copy
        article_copy = copy.deepcopy(article)
        # Remove BPE
        article = [process_bpe_symbol(sent, "@@ ") for sent in article]
        summary = []
        order = []
        for s, info in zip(*extractor(
                article, length, order_by_article=not args.paraphrase_extract)):
            order.append(info.order)
            summary.append(article_copy[info.order])
        return summary, order

    import time

    total_processed = 0

    pbar = tqdm(desc="Summarized")

    start_time_global = time.time()
    for iteration_n, input_lines in enumerate(
            buffered_read(args.input, args.buffer_size)):
        start_time = time.time()

        if args.num_output_sentences > 0:
            max_summary_length = args.num_output_sentences
            if args.paraphrase_extract:
                articles = [inp.strip().split(" <s> ") for inp in input_lines]
            else:
                articles = [
                    extractive_summarization(inp.strip().split(" <s> "), max_summary_length)[0]
                    for inp in input_lines
                ]
        else:
            articles = [inp.strip().split(" <s> ") for inp in input_lines]
            max_summary_length = len(articles[0])

        article_lengths = [len(article) for article in articles]
        finished_generation = [False for article in articles]
        total_sentences_in_buffer = np.sum(article_lengths)
        sentence_selection_indices = [[] for i in range(len(articles))]
        summary_history = [[] for j in range(len(articles))]
        final_clean_summaries = [[] for j in range(len(articles))]
        final_clean_summaries_lengths = [len(s) for s in final_clean_summaries]

        if total_sentences_in_buffer * args.rescore_nbest < args.max_sentences:
            args.max_sentences = total_sentences_in_buffer * args.rescore_nbest
            print("WARNING: you can increase your buffer size")

        # Disable LM for first sentence
        alpha = 0.

        if args.extractive_only:
            for i, article in enumerate(articles):
                for sent in article:
                    final_clean_summaries[i].append(process_bpe_symbol(sent, "@@ "))
        else:
            for sentence_num in range(max_summary_length):

                if all(finished_generation):
                    break

                if sentence_num > 0:
                    alpha = args.alpha

                # if sentence_num == 0 or alpha > 0.:  # Only regenerate paraphrases first time or if alpha > 0.
                paraphrases, paraphrases_clean, paraphrase_scores = \
                    generate_paraphrases(
                        models, articles, summary_history,
                        args.rescore_nbest
                    )
                # print(paraphrases_clean)

                if args.paraphrase_extract:
                    for article_id, (article_paraphrases, article_paraphrases_clean) \
                            in enumerate(zip(paraphrases, paraphrases_clean)):
                        extracted_summary, sentence_location = extractive_summarization(
                            article_paraphrases_clean, len(article_paraphrases_clean))
                        assert len(sentence_location) == len(extracted_summary) \
                               == len(article_paraphrases) == len(article_paraphrases_clean)
                        for location, paraphrase in zip(sentence_location, extracted_summary):
                            if location in sentence_selection_indices[article_id]:  # Sentence already in summary
                                continue
                            skip_sent = False
                            for existing_summary_sent in final_clean_summaries[article_id]:
                                if jaccard_similarity(paraphrase, existing_summary_sent) > 0.8 \
                                        or len(paraphrase) < 3:
                                    skip_sent = True
                                    break
                            if not skip_sent:
                                paraphrase_len = len(paraphrase.split())
                                if args.max_summary_tokens > 0:
                                    if final_clean_summaries_lengths[article_id] + paraphrase_len > args.max_summary_tokens:
                                        finished_generation[article_id] = True
                                        break

                                final_clean_summaries[article_id].append(paraphrase)
                                final_clean_summaries_lengths[article_id] += paraphrase_len
                                sentence_selection_indices[article_id].append(location)
                                summary_history[article_id].append(article_paraphrases[location])
                                break
                else:
                    for article_id, (article_paraphrases, article_paraphrases_clean) \
                            in enumerate(zip(paraphrases, paraphrases_clean)):
                        if sentence_num > len(article_paraphrases) - 1:  # Article shorter than expected summary length
                            continue
                        next_sent = article_paraphrases[sentence_num]
                        next_sent_clean = article_paraphrases_clean[sentence_num]
                        final_clean_summaries[article_id].append(next_sent_clean)
                        final_clean_summaries_lengths[article_id] += len(next_sent_clean.split())
                        sentence_selection_indices[article_id].append(sentence_num)
                        summary_history[article_id].append(next_sent)

        for sel_idx in sentence_selection_indices:
            for j in sel_idx:
                if j in selection_stat:
                    selection_stat[j] += 1
                else:
                    selection_stat[j] = 1

        if args.print_summary:
            for article_sentences, summary_sentences, summary_history_sents in zip(
                    articles, final_clean_summaries, summary_history):
                print("Input:")
                for i, paraphrase in enumerate(article_sentences):
                    print("  {}) {}".format(i, paraphrase.replace("@@ ", " ")))
                print()
                print("Summary:")
                for i, paraphrase in enumerate(summary_sentences):
                    print("  {}) {}".format(i, paraphrase))
                print("Summary history:")
                for i, paraphrase in enumerate(summary_history_sents):
                    print("  {}) {}".format(i, paraphrase))
                print("*"*50)
                print()

        for final_clean_summary in final_clean_summaries:
            if args.buffer_size == 1:
                pbar.update()
            if args.detokenize:
                final_clean_summary = [detokenizer.detokenize(s.split(), return_str=True) for s in final_clean_summary]
                out_file.write("{}\n".format(" ".join(final_clean_summary)))
            else:
                out_file.write("{}\n".format(" <s> ".join(final_clean_summary)))

        out_file.flush()
        total_processed += len(input_lines)

        end_time = (time.time() - start_time)
        if args.buffer_size > 1:
            print("--- Processed {} articles ({}s, {}s/article) ---".format(
                total_processed, np.round(end_time, 4),
                np.round(end_time / len(input_lines), 4)))

        # update running id counter
        start_id += total_sentences_in_buffer

    end_time_global = (time.time() - start_time_global)
    print("--- Total time for {} articles ({}s, {}s/article) ---".format(
        total_processed, np.round(end_time_global, 4),
        np.round(end_time_global / total_processed, 4)))

    print("Selection stat: {}".format(selection_stat))

    out_file.close()


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    parser.add_argument('--output_file', help='file to write the output to')
    parser.add_argument('--alpha', help='file to write the output to', type=float, default=0.)
    parser.add_argument('--num_output_sentences', help='file to write the output to',
                        default=-1, type=int)
    parser.add_argument('--max_summary_tokens', help='file to write the output to',
                        default=-1, type=int)
    parser.add_argument('--rescore_nbest',
        help='Rescore nbest paraphrases with the help of the LM',
        default=1, type=int)
    parser.add_argument('--extractive_approach', help="Paraphrase then extract",
                        default="lead")
    parser.add_argument('--paraphrase_extract', help="Paraphrase then extract",
                        default=False, type=bool)
    parser.add_argument('--extractive_only', help="Only extractive summarization",
                        default=False, type=bool)
    parser.add_argument('--print_summary', help="print summaries to std out",
                        default=True, type=bool)
    parser.add_argument('--detokenize',
                        help="detokenize output summary sentences with NLTK", default=False, type=bool)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
