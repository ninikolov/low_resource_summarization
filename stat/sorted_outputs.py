import argparse
from random import shuffle
import numpy as np
from rouge import Rouge
import sys
from summary_rewriting.summarization_systems.oracle import repeat_rate

parser = argparse.ArgumentParser()
parser.add_argument('--input', "-i", help='The input file.', required=True)
parser.add_argument('--systems', "-s", nargs='+', help='The system files. If you pass multiple files, it '
                                                       'will print all of them, sorting by the first one.'
                    , required=True)
parser.add_argument('--ground_truth', "-g", help='The ground truth file.')
parser.add_argument('--line_limit', "-l", help='Print only this many lines.', default=-1, type=int)
parser.add_argument('--print_only', "-p", help='Print only the results for this specific line.',
                    default=-1, type=int)
args = parser.parse_args()

rouge = Rouge()


def pretty_print(text_line, ref, label):
    rouge_scores = rouge.get_scores(text_line, ref)
    r1 = np.round(rouge_scores[0]["rouge-1"]["f"] * 100, 2)
    r2 = np.round(rouge_scores[0]["rouge-2"]["f"] * 100, 2)
    rl = np.round(rouge_scores[0]["rouge-l"]["f"] * 100, 2)

    repeat = np.round(repeat_rate(text_line.split(" <s> ")) * 100, 2)

    print(" * {} * - * {} * {} : \n{}".format(
        label, len(text_line.split()), "R1:{}-R2:{}-RL:{}-Rep:{}".format(r1, r2, rl, repeat), text_line.strip()))


def clean_line(line):
    return " ".join(line.strip().split(" <s> "))


if __name__ == '__main__':
    gt_lines = open(args.ground_truth).readlines()
    input_lines = open(args.input).readlines()
    all_system_lines = {
        s: open(s).readlines() for s in args.systems
    }

    if args.print_only > 0:
        args.print_only -= 1
        # gt_lines = gt_lines[args.print_only]
        # input_lines = input_lines[args.print_only]
        # all_system_lines = {s: lines[args.print_only] for s, lines in all_system_lines.items()}

        pretty_print(gt_lines[args.print_only], gt_lines[args.print_only], args.ground_truth.split("/")[-1:])
        pretty_print(input_lines[args.print_only], gt_lines[args.print_only], args.input.split("/")[-1:])
        system_lines = [all_system_lines[s][args.print_only].strip() for s in args.systems]
        for j, system_name in enumerate(args.systems):
            pretty_print(system_lines[j], gt_lines[args.print_only], system_name.split("/")[-2:])
        print("-" * 10)
        sys.exit(0)

    # token_diffs = [len(inp.strip().split()) - len(sys.strip().split())
    #                for inp, sys in zip(input_lines, gt_lines)]
    token_diffs = [repeat_rate(inp.split(" <s> ")) for inp in input_lines]
    sorted_diffs = np.argsort(token_diffs)[::-1]

    if args.line_limit > 0:
        sorted_diffs_lim = sorted_diffs[:args.line_limit]
    else:
        sorted_diffs_lim = sorted_diffs

    for i in sorted_diffs[:args.line_limit]:
        input_line = input_lines[i].strip()
        gt_line = gt_lines[i].strip()
        system_lines = [all_system_lines[s][i].strip() for s in args.systems]

        br = True
        for system_line in system_lines:
            if system_line not in input_line:
                br = False

        if br == False:
            for system_line in system_lines:
                if len(system_line.split()) == len(input_line.split()):
                    br = True
                    break
        if br:
            continue

        pretty_print(gt_line, gt_line, args.ground_truth.split("/")[-1:])
        pretty_print(input_line, gt_line, args.input.split("/")[-1:])
        for j, system_name in enumerate(args.systems):
            pretty_print(system_lines[j], gt_line, system_name.split("/")[-2:])
        print("-" * 10)

