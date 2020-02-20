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
import sys
from tqdm import *

import torch
import numpy as np

from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator
from fairseq.sequence_scorer import SequenceScorer
from fairseq.utils import import_user_module
from fairseq.data.data_utils import process_bpe_symbol
from fairseq.data import data_utils
from fairseq.data import LanguagePairDataset

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


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
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


def main(args):
    import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

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

    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    start_id = 0

    if args.output_file is not None:
        args.output_file = "{}.{}".format(args.output_file, args.alpha)
        out_file = open(args.output_file, "w")
        print('| Writing output to: {}'.format(args.output_file))

    selection_stat = {}

    input_id = -1

    def make_history(summary, size):
        # Compose the history batch (all past sentences). Input to the LM.
        current_history = ["<s> " + " <s> ".join(summary)
                           if len(summary) > 0 else "<s>"
                           for i in range(size)]

        history_batch = next(make_batches(
            current_history, args, task, max_positions))

        history_src_tokens = history_batch.src_tokens
        history_src_lengths = history_batch.src_lengths

        if use_cuda:
            history_src_tokens = history_src_tokens.cuda()
            history_src_lengths = history_src_lengths.cuda()
        return history_src_tokens, history_src_lengths

    def generate_paraphrases(sentences_batch, history, alpha, return_n_best=1):
        paraphrase_batch = next(make_batches(
            sentences_batch, args, task, max_positions))

        src_tokens = paraphrase_batch.src_tokens
        src_lengths = paraphrase_batch.src_lengths
        if use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()

        sample = {
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths
            }
        }

        if alpha > 0.: # Generate history for the language model; run LM and PM as an ensemble
            sample['alpha'] = alpha
            history_src_tokens, history_src_lengths = make_history(history, args.max_sentences)
            sample['net_input']["history_tokens"] = history_src_tokens
            sample['net_input']["history_src_lengths"] = history_src_lengths
            # Generate the next abstractive sentences
            paraphrase_predictions = task.inference_step(generator, models, sample)
        else:
            # Generate the next abstractive sentences using paraphrasing model only
            paraphrase_predictions = task.inference_step(generator, [models[0]], sample)

        results = []

        for i, (id, hypos) in enumerate(zip(paraphrase_batch.ids.tolist(), paraphrase_predictions)):
            src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
            results.append((start_id + id, src_tokens_i, hypos))

        paraphrases = []
        paraphrases_clean = []

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

                paraphrases.append(hypo_str)
                # Remove BPE
                paraphrases_clean.append(process_bpe_symbol(hypo_str, "@@ "))
        assert int(len(paraphrases) / return_n_best) == len(sentences_batch)
        return paraphrases, paraphrases_clean

    def score_paraphrases(paraphrases, paraphrases_clean):
        paraphrases_batch = next(make_batches(
            paraphrases, args, task, max_positions))
        paraphrases_src_tokens = paraphrases_batch.src_tokens
        if use_cuda:
            paraphrases_src_tokens = paraphrases_src_tokens.cuda()

        prev_output_tokens = data_utils.collate_tokens(
            paraphrases_src_tokens,
            src_dict.pad(), src_dict.eos(), True, True,
        )
        # prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        if use_cuda:
            prev_output_tokens = prev_output_tokens.cuda()

        # Expand history to accomodate for all the hypotheses on the beam
        history_src_tokens, history_src_lengths = make_history(
            abstractive_summary, args.max_sentences * args.beam)
        history_sample = {
            'net_input': {
                'src_tokens': history_src_tokens,
                'src_lengths': history_src_lengths,
                'prev_output_tokens': prev_output_tokens
            },
            "target": paraphrases_src_tokens
        }

        lm_predictions = task.inference_step(scorer, [models[-1]], history_sample)
        # Get the scores of the language model
        lm_scores = np.array([hypo[0]["score"].data.cpu().numpy() for hypo in lm_predictions])
        # print("Tr {}".format(len(translations)))
        # print("LM SCores {} {}".format(lm_scores.shape, lm_scores))

        # Pick the best paraphrases for each input sentence (out of args.beam possibilities)
        beam_best_paraphrases = []
        beam_best_paraphrases_clean = []
        beam_best_score = []
        for j in np.arange(0, len(lm_scores), args.beam):
            beam_hypo_scores = lm_scores[j:j + args.beam]
            best = np.argmax(beam_hypo_scores)
            # print(best)
            beam_best_score.append(beam_hypo_scores[best])
            beam_best_paraphrases.append(paraphrases[j + best])
            beam_best_paraphrases_clean.append(paraphrases_clean[j + best])

        return beam_best_score, beam_best_paraphrases, beam_best_paraphrases_clean

    with tqdm() as pbar:
        for inputs in buffered_read(args.input, args.buffer_size):
            input_id += 1
            article_sentences = inputs[0].split(" <s> ")

            selection_idx = []
            abstractive_summary = []
            final_clean_summary = []

            if args.num_output_sentences > 0:
                summary_length = args.num_output_sentences
                paraphrase_all_sentences = False
            else:
                summary_length = len(article_sentences)
                paraphrase_all_sentences = True

            # Disable LM for first sentence
            alpha = 0.

            for sentence_num in range(summary_length):
                # Set batch size to total number of sentences in input article
                args.max_sentences = len(article_sentences)

                if sentence_num > 0:
                    alpha = args.alpha

                if sentence_num == 0 or alpha > 0.:
                    paraphrases, paraphrases_clean = generate_paraphrases(
                        article_sentences, abstractive_summary, alpha, args.rescore_nbest)

                if args.rescore_nbest == 1 and paraphrase_all_sentences:
                    abstractive_summary.append(paraphrases[sentence_num])
                    final_clean_summary.append(paraphrases_clean[sentence_num])
                    continue

                # print(" # Paraphrases: {}".format(len(paraphrases)))

                # Increase batch size to accommodate for each paraphrase beam hypothesis
                args.max_sentences *= args.rescore_nbest

                beam_best_score, beam_best_paraphrases, beam_best_paraphrases_clean = score_paraphrases(
                    paraphrases, paraphrases_clean
                )

                # print("# Paraphrases after rescoring: {}".format(len(beam_best_paraphrases_clean)))

                # Pick the best paraphrase and add it to the summary
                next_sentence_order = np.argsort(beam_best_score)[::-1]
                for idx in next_sentence_order:
                    if idx not in selection_idx:
                        next_sentence_idx = idx
                        selection_idx.append(next_sentence_idx)
                        break

                abstractive_summary.append(beam_best_paraphrases[next_sentence_idx])
                final_clean_summary.append(beam_best_paraphrases_clean[next_sentence_idx])

            # print(selection_idx)
            for j in selection_idx:
                if j in selection_stat:
                    selection_stat[j] += 1
                else:
                    selection_stat[j] = 1

            if not args.no_print:
                print("Input:")
                for i, s in enumerate(article_sentences):
                    print("  {}) {}".format(i, s.replace("@@ ", " ")))
                print()
                print("Summary:")
                for i, s in enumerate(final_clean_summary):
                    print("  {}) {}".format(i, s))
                print("*"*50)

            out_file.write("{}\n".format(" <s> ".join(final_clean_summary)))

            out_file.flush()
            pbar.update()

        print("Selection stat: {}".format(selection_stat))

    out_file.close()


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    parser.add_argument('--output_file', help='file to write the output to')
    parser.add_argument('--alpha', help='file to write the output to', type=float, default=0.)
    parser.add_argument('--num_output_sentences', help='file to write the output to', default=-1, type=int)
    parser.add_argument('--rescore_nbest', help='Rescore nbest paraphrases with the help of the LM',
                        default=1, type=int)
    parser.add_argument('--no_print', help="Don't print to std out", default=True, type=bool)
    args = options.parse_args_and_arch(parser)

    print("\nSummarization args:\nAlpha: {} Num output sentences: {} Rescore nbest: {}\n".format(
        args.alpha, args.num_output_sentences, args.rescore_nbest))

    main(args)


if __name__ == '__main__':
    cli_main()
