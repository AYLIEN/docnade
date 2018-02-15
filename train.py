import os
import argparse
import json
import numpy as np
import tensorflow as tf
import model.data as data
import model.model as m
import model.evaluate as e

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


def train(model, dataset, params):
    log_dir = os.path.join(params.model, 'logs')
    model_dir = os.path.join(params.model, 'model')

    with tf.Session(config=tf.ConfigProto(
        inter_op_parallelism_threads=params.num_cores,
        intra_op_parallelism_threads=params.num_cores,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )) as session:
        avg_loss = tf.placeholder(tf.float32, [], 'loss_ph')
        tf.summary.scalar('loss', avg_loss)

        validation = tf.placeholder(tf.float32, [], 'validation_ph')
        tf.summary.scalar('validation', validation)

        summary_writer = tf.summary.FileWriter(log_dir, session.graph)
        summaries = tf.summary.merge_all()
        saver = tf.train.Saver(tf.global_variables())

        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        losses = []

        # This currently streams from disk. You set num_epochs=1 and
        # wrap this call with something like itertools.cycle to keep
        # this data in memory.
        training_data = dataset.batches('training', params.batch_size)

        best_val = 0.0
        training_labels = np.array(
            [[y] for y, _ in dataset.rows('training', num_epochs=1)]
        )
        validation_labels = np.array(
            [[y] for y, _ in dataset.rows('validation', num_epochs=1)]
        )

        for step in range(params.num_steps + 1):
            _, x, seq_lengths = next(training_data)

            _, loss = session.run([model.opt, model.opt_loss], feed_dict={
                model.x: x,
                model.seq_lengths: seq_lengths
            })
            losses.append(loss)

            if step % params.log_every == 0:
                print('{}: {:.6f}'.format(step, loss), flush=True)

            if step and (step % params.save_every) == 0:
                validation_vectors = m.vectors(
                    model,
                    dataset.batches(
                        'validation',
                        params.batch_size,
                        num_epochs=1
                    ),
                    session
                )
                training_vectors = m.vectors(
                    model,
                    dataset.batches(
                        'training',
                        params.batch_size,
                        num_epochs=1,
                        sample_size=params.eval_max_sample
                    ),
                    session
                )
                training_sample = dataset.sample_indexes
                val = e.evaluate(
                    training_vectors,
                    validation_vectors,
                    training_labels[training_sample],
                    validation_labels
                )[0]
                print('validation: {:.3f} (best: {:.3f})'.format(
                    val,
                    best_val or 0.0
                ), flush=True)

                if val > best_val:
                    best_val = val
                    print('saving: {}'.format(model_dir), flush=True)
                    saver.save(session, model_dir, global_step=step)

                summary, = session.run([summaries], feed_dict={
                    model.x: x,
                    model.seq_lengths: seq_lengths,
                    validation: val,
                    avg_loss: np.average(losses)
                })
                summary_writer.add_summary(summary, step)
                summary_writer.flush()
                losses = []


def main(args):
    if not os.path.isdir(args.model):
        os.mkdir(args.model)

    with open(os.path.join(args.model, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))

    dataset = data.Dataset(args.dataset)
    x = tf.placeholder(tf.int32, shape=(None, None), name='x')
    seq_lengths = tf.placeholder(tf.int32, shape=(None), name='seq_lengths')
    model = m.DocNADE(x, seq_lengths, args)
    train(model, dataset, args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='path to model output directory')
    parser.add_argument('--dataset', type=str, required=True,
                        help='path to the input dataset')
    parser.add_argument('--vocab-size', type=int, default=2000,
                        help='the vocab size')
    parser.add_argument('--hidden-size', type=int, default=50,
                        help='size of the hidden layer')
    parser.add_argument('--activation', type=str, default='tanh',
                        help='which activation to use: sigmoid|tanh')
    parser.add_argument('--learning-rate', type=float, default=0.0004,
                        help='initial learning rate')
    parser.add_argument('--num-steps', type=int, default=50000,
                        help='the number of steps to train for')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='softmax samples (default: full softmax)')
    parser.add_argument('--num-cores', type=int, default=2,
                        help='the number of CPU cores to use')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--save-every', type=int, default=500,
                        help='print loss after this many steps')
    parser.add_argument('--eval-max-sample', type=int, default=None,
                        help='max training data sample size when evaluating')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
