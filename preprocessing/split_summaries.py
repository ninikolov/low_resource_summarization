
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-summary_file', help='The summary file.',
                    default="shuffle")
parser.add_argument('-out_dir', help='The output dir to store the summaries.', required=True)
args = parser.parse_args()
target_key = "abstract"
source_key = "article"

if __name__ == '__main__':
    summaries = open(args.summary_file).readlines()
    for i in range(len(summaries)):
        summary = summaries[i].strip()
        summary_lines = summary.split(" <s> ")
        with open("{}/{}.dec".format(args.out_dir, i), "w") as out:
            for line in summary_lines:
                out.write("{}\n".format(line.strip()))