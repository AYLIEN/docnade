import os
import glob
import random
import csv
import itertools
import numpy as np
import keras.preprocessing.sequence as pp
import tensorflow as tf
from itertools import count

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

csv.field_size_limit(2**28)


def file_name(abs_path):
    return os.path.basename(abs_path).replace('.csv', '')


def format_row(row):
    y, raw_x = row
    x = np.array([int(v) for v in raw_x.split()])
    np.random.shuffle(x)
    return [int(y), x, len(x)]


class Dataset(object):
    def __init__(self, path):
        self.path = path
        files = glob.glob(path + '/*.csv')
        self.collections = {file_name(file): file for file in files}
        self.collection_sizes = {}

    def _get_collection_size(self, collection):
        size = self.collection_sizes.get(collection, None)
        if size is not None:
            return size
        file = self.collections.get(collection, None)
        if file is None:
            raise ValueError("Collection %s not present!"%collection)
        with open(file, 'r', newline='') as f:
            r = csv.reader(f)
            size = sum(1 for _ in r)
        self.collection_sizes[collection] = size
        return size

    def rows(self, collection_name, num_epochs=None, sample_size=None):
        if collection_name not in self.collections:
            raise ValueError(
                'Collection not found: {}'.format(collection_name)
            )
        epoch = 0
        if sample_size is not None:
            collection_size = self._get_collection_size(collection_name)
            assert sample_size < collection_size, "Only samples smaller than available data are provided for."
            sample_indexes = np.random.choice(collection_size, sample_size, replace=False)
            sample_indexes.sort()
            sample_indexes = iter(sample_indexes)
            next_index = next(sample_indexes)
        while True:
            with open(self.collections[collection_name], 'r', newline='') as f:
                r = csv.reader(f)
                if sample_size is None:
                    for row in r:
                        yield row
                else:
                    for row,index in zip(r, count()):
                        if index == next_index:
                            yield row
                            next_index = next(sample_indexes) # NOTE: will raise StopIteration when indexes exhausted
            epoch += 1
            if num_epochs and (epoch >= num_epochs):
                raise StopIteration

    def _batch_iter(self, collection_name, batch_size, num_epochs, sample_size):
        gen = [self.rows(collection_name, num_epochs, sample_size)] * batch_size
        return itertools.zip_longest(fillvalue=None, *gen)

    def batches(self, collection_name, batch_size, num_epochs=None, sample_size=None):
        for batch in self._batch_iter(collection_name, batch_size, num_epochs, sample_size):
            data = [format_row(row) for row in batch if row]
            y, x, seq_lengths = zip(*data)
            x = pp.pad_sequences(x, padding='post')
            yield np.array(y), x, np.array(seq_lengths)
