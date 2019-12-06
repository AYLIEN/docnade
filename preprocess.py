import argparse
import os
import csv
import numpy as np
import tensorflow as tf
import nltk
import model.data

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

csv.field_size_limit(2**28)


def tokens(text):
    return [w.lower() for w in nltk.word_tokenize(text)]


def counts_to_sequence(counts):
    seq = []
    for i in range(len(counts)):
        seq.extend([i] * int(counts[i]))
    return seq


def log_counts(ids, vocab_size):
    counts = np.bincount(ids, minlength=vocab_size)
    return np.floor(0.5 + np.log(counts + 1))


def preprocess(text, vocab_to_id):
    ids = [
        vocab_to_id[x] for x in tokens(text)
        if vocab_to_id.get(x) is not None
    ]
    counts = log_counts(ids, len(vocab_to_id))
    sequence = counts_to_sequence(counts)
    return ' '.join([str(x) for x in sequence])


def main(args):
    data = model.data.Dataset(args.input)
    with open(args.vocab, 'r') as f:
        vocab = [w.strip() for w in f.readlines()]
    vocab_to_id = dict(zip(vocab, range(len(vocab))))

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    labels = {}
    for collection in data.collections:
        output_path = os.path.join(args.output, '{}.csv'.format(collection))
        with open(output_path, 'w', newline='') as f:
            w = csv.writer(f, delimiter=',')
            for y, x in data.rows(collection, num_epochs=1):
                if y not in labels:
                    labels[y] = len(labels)

                w.writerow((labels[y], preprocess(x, vocab_to_id)))

    with open(os.path.join(args.output, 'labels.txt'), 'w') as f:
        f.write('\n'.join([k for k in sorted(labels, key=labels.get)]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='path to the input dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='path to the output dataset')
    parser.add_argument('--vocab', type=str, required=True,
                        help='path to the vocab')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
