import os
import argparse
import json
from collections import namedtuple
import numpy as np
import tensorflow as tf
import model.data as data
import model.model as m
import model.evaluate as e


def evaluate_loss(model, dataset, params, session):
    print('evaluating loss')
    loss = m.loss(
        model,
        dataset.batches('validation', params.batch_size, num_epochs=1),
        session
    )
    print('loss: {}'.format(loss))


def evaluate_retrieval(model, dataset, params, session):
    print('evaluating retrieval')
    print('computing vectors...')

    validation_labels = np.array(
        [[y] for y, _ in dataset.rows('validation', num_epochs=1)]
    )
    training_labels = np.array(
        [[y] for y, _ in dataset.rows('training', num_epochs=1)]
    )
    training_labels = np.concatenate((training_labels, validation_labels), 0)
    test_labels = np.array(
        [[y] for y, _ in dataset.rows('test', num_epochs=1)]
    )

    validation_vectors = m.vectors(
        model,
        dataset.batches('validation', params.batch_size, num_epochs=1),
        session
    )
    training_vectors = m.vectors(
        model,
        dataset.batches('training', params.batch_size, num_epochs=1),
        session
    )
    training_vectors = np.concatenate(
        (training_vectors, validation_vectors),
        0
    )
    test_vectors = m.vectors(
        model,
        dataset.batches('test', params.batch_size, num_epochs=1),
        session
    )

    print('evaluating...')

    recall_values = [0.0001, 0.0002, 0.0005, 0.002, 0.01, 0.05, 0.2]
    results = e.evaluate(
        training_vectors,
        test_vectors,
        training_labels,
        test_labels,
        recall_values
    )
    for i, r in enumerate(recall_values):
        print('precision @ {}: {}'.format(r, results[i]))


def evaluate(model, dataset, params):
    with tf.Session(config=tf.ConfigProto(
        inter_op_parallelism_threads=params.num_cores,
        intra_op_parallelism_threads=params.num_cores,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )) as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(params.model)
        saver.restore(session, ckpt.model_checkpoint_path)

        evaluate_retrieval(model, dataset, params, session)
        evaluate_loss(model, dataset, params, session)


def main(args):
    with open(os.path.join(args.model, 'params.json'), 'r') as f:
        params = json.loads(f.read())
    params.update(vars(args))
    params = namedtuple('Params', params.keys())(*params.values())

    dataset = data.Dataset(args.dataset)
    x = tf.placeholder(tf.int32, shape=(None, None), name='x')
    seq_lengths = tf.placeholder(tf.int32, shape=(None), name='seq_lengths')
    model = m.DocNADE(x, seq_lengths, params)
    evaluate(model, dataset, params)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='path to model output directory')
    parser.add_argument('--dataset', type=str, required=True,
                        help='path to the input dataset')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--num-cores', type=int, default=1,
                        help='the number of CPU cores to use')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
