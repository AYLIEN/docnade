import unittest
from collections import namedtuple
import numpy as np
import tensorflow as tf
import model as m

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


float_precision = 5  # number of decimal places to check to


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = namedtuple('params', [
            'vocab_size',
            'hidden_size',
            'activation',
            'learning_rate',
            'num_samples'
        ])
        cls.params.vocab_size = 10
        cls.params.hidden_size = 2
        cls.params.activation = 'sigmoid'
        cls.params.learning_rate = 0.0004
        cls.params.num_samples = None
        cls.x = tf.placeholder(tf.int64, [None, None], 'x')
        cls.seq_lengths = tf.placeholder(tf.int64, [None], 'seq_lengths')
        cls.model = m.DocNADE(cls.x, cls.seq_lengths, cls.params)

    def test_pre_act(self):
        '''
        Test that pre-activations are the sum of embedding vectors.
        '''
        batch_size = 3
        seq_lengths = 10
        x = []
        for i in range(batch_size):
            x.append(np.random.randint(0, self.params.vocab_size, seq_lengths))
        x = np.array(x)

        with tf.Session() as session:
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()

            embeddings = session.run(self.model.embeddings, feed_dict={
                self.x: x,
                self.seq_lengths: np.array(
                    [seq_lengths for _ in range(batch_size)]
                )
            })
            embeddings = np.concatenate(
                [
                    np.zeros((batch_size, 1, self.params.hidden_size)),
                    embeddings
                ],
                axis=1
            )
            pre_act = session.run(self.model.pre_act, feed_dict={
                self.x: x,
                self.seq_lengths: np.array(
                    [seq_lengths for _ in range(batch_size)]
                )
            })

            sum = np.zeros((batch_size, 2))
            for i in range(seq_lengths):
                sum += embeddings[:, i]
                assert np.allclose(sum, pre_act[:, i])

    def test_final_state_seq_lengths(self):
        '''
        Test that the sequence length parameter is respected when calculating
        final state (ie, ignores include batch padding).
        '''
        seq_lengths = 10
        x = np.random.randint(0, self.params.vocab_size, seq_lengths)
        x_2 = np.pad(x, (0, seq_lengths), 'constant')

        with tf.Session() as session:
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()

            h_1 = session.run(self.model.h, feed_dict={
                self.x: np.array([x]),
                self.seq_lengths: np.array([seq_lengths])
            })

            h_2 = session.run(self.model.h, feed_dict={
                self.x: np.array([x_2]),
                self.seq_lengths: np.array([seq_lengths])
            })

            assert np.allclose(h_1[0, -1], h_2[0, -1])

            h_3 = session.run(self.model.h, feed_dict={
                self.x: np.array([x_2]),
                self.seq_lengths: np.array([seq_lengths * 2])
            })

            assert not np.allclose(h_1[0, -1], h_3[0, -1])

    def test_autoregressive(self):
        '''
        Test that each output in a sequence does not depend on future input
        values.
        '''
        batch_size = 3
        seq_lengths = 10
        x = []
        for i in range(batch_size):
            x.append(np.random.randint(0, self.params.vocab_size, seq_lengths))
        x = np.array(x)

        with tf.Session() as session:
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()

            for i in range(seq_lengths):
                logits = session.run(self.model.logits, feed_dict={
                    self.x: x,
                    self.seq_lengths: np.array(
                        [seq_lengths for _ in range(batch_size)]
                    )
                })

                logits = logits.reshape(
                    batch_size,
                    seq_lengths,
                    self.params.vocab_size
                )
                logits = logits[:, 0:i + 1, :]

                logits_masked = session.run(self.model.logits, feed_dict={
                    self.x: x * np.hstack(
                        (np.ones(i + 1), np.zeros(seq_lengths - i - 1))
                    ),
                    self.seq_lengths: np.array(
                        [seq_lengths for _ in range(batch_size)]
                    )
                })
                logits_masked = logits_masked.reshape(
                    batch_size,
                    seq_lengths,
                    self.params.vocab_size
                )
                logits_masked = logits_masked[:, 0:i + 1, :]

                assert np.allclose(logits, logits_masked)

    def test_masked_sequence_cross_entropy_loss(self):
        vocab_size = 2
        x = tf.placeholder(tf.int64, [None, None], 'x')
        seq_lengths = tf.placeholder(tf.int64, [None], 'seq_lengths')
        logits = tf.placeholder(tf.float32, [None, vocab_size])
        loss = m.masked_sequence_cross_entropy_loss(x, seq_lengths, logits)

        x_1 = np.array([[0, 1, 1]])
        seq_lengths_1 = np.array([3])
        logits_1 = np.array([[100.0, 0.0], [0.0, 100.0], [0.0, 100.0]])
        logits_2 = np.array([[100.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
        logits_3 = np.array([[100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])

        x_2 = np.array([[0, 1, 1], [0, 1, 1]])
        seq_lengths_2 = np.array([3, 2])  # ignore last element of second seq
        logits_4 = np.array([
            [100.0, 0.0],
            [0.0, 100.0],
            [0.0, 100.0],
            [100.0, 0.0],
            [0.0, 100.0],
            [100.0, 0.0]
        ])

        logits_5 = np.array([
            [100.0, 0.0],
            [0.0, 100.0],
            [100.0, 100.0],
            [100.0, 0.0],
            [0.0, 100.0],
            [100.0, 0.0]
        ])

        with tf.Session() as session:
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()

            loss_1 = session.run(loss, feed_dict={
                x: x_1,
                seq_lengths: seq_lengths_1,
                logits: logits_1
            })
            self.assertAlmostEqual(loss_1, 0.0, places=float_precision)

            loss_2 = session.run(loss, feed_dict={
                x: x_1,
                seq_lengths: seq_lengths_1,
                logits: logits_2
            })
            self.assertAlmostEqual(
                loss_2,
                -np.log(1.0 / np.exp(100.0)) / 3.0,
                places=float_precision
            )

            loss_3 = session.run(loss, feed_dict={
                x: x_1,
                seq_lengths: seq_lengths_1,
                logits: logits_3
            })
            self.assertAlmostEqual(
                loss_3,
                -np.log(0.5) / 3.0,
                places=float_precision
            )

            loss_4 = session.run(loss, feed_dict={
                x: x_2,
                seq_lengths: seq_lengths_2,
                logits: logits_4
            })
            self.assertAlmostEqual(loss_4, 0.0, places=float_precision)

            loss_5 = session.run(loss, feed_dict={
                x: x_2,
                seq_lengths: seq_lengths_2,
                logits: logits_5
            })
            self.assertNotAlmostEqual(loss_5, 0.0, places=float_precision)


if __name__ == '__main__':
    unittest.main()
