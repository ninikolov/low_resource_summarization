import argparse
from random import shuffle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-gt', help='The gt file.')
parser.add_argument('-system', help='The system file.')
args = parser.parse_args()


if __name__ == '__main__':
    sentence_lengths = {}
    token_length_diff = []

    def add_result(num):
        if num in sentence_lengths.keys():
            sentence_lengths[num] += 1
        else:
            sentence_lengths[num] = 1

    gt_lines = open(args.gt).readlines()
    system_lines = open(args.system).readlines()
    for gt_line, system_line in zip(gt_lines, system_lines):
        gt_len = len(gt_line.split("<s>"))
        system_len = len(system_line.split("<s>"))
        add_result(system_len - gt_len)

    print(sentence_lengths)

    for k in np.sort(list(sentence_lengths.keys())):
        # if k >= 0:
        print("{}: {}".format(k, np.round(sentence_lengths[k] / len(gt_lines), 3)))