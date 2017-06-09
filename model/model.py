import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


def vectors(model, data, session):
    vecs = []
    for _, x, seq_lengths in data:
        vecs.extend(
            session.run([model.h], feed_dict={
                model.x: x,
                model.seq_lengths: seq_lengths
            })[0]
        )
    return np.array(vecs)


def loss(model, data, session):
    loss = []
    for _, x, seq_lengths in data:
        loss.append(
            session.run([model.loss], feed_dict={
                model.x: x,
                model.seq_lengths: seq_lengths
            })[0]
        )
    return sum(loss) / len(loss)


def gradients(opt, loss, vars, step, max_gradient_norm=None, dont_clip=[]):
    gradients = opt.compute_gradients(loss, vars)
    if max_gradient_norm is not None:
        to_clip = [(g, v) for g, v in gradients if v.name not in dont_clip]
        not_clipped = [(g, v) for g, v in gradients if v.name in dont_clip]
        gradients, variables = zip(*to_clip)
        clipped_gradients, _ = clip_ops.clip_by_global_norm(
            gradients,
            max_gradient_norm
        )
        gradients = list(zip(clipped_gradients, variables)) + not_clipped

    # Add histograms for variables, gradients and gradient norms
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        if grad_values is None:
            print('warning: missing gradient: {}'.format(variable.name))
        if grad_values is not None:
            tf.summary.histogram(variable.name, variable)
            tf.summary.histogram(variable.name + '/gradients', grad_values)
            tf.summary.histogram(
                variable.name + '/gradient_norm',
                clip_ops.global_norm([grad_values])
            )

    return opt.apply_gradients(gradients, global_step=step)


def linear(input, output_dim, scope=None, stddev=None):
    if stddev:
        norm = tf.random_normal_initializer(stddev=stddev)
    else:
        norm = tf.random_normal_initializer(
            stddev=np.sqrt(2.0 / input.get_shape()[1].value)
        )
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=norm
        )
        b = tf.get_variable('b', [output_dim], initializer=const)
    return tf.matmul(input, w) + b


def masked_sequence_cross_entropy_loss(
    x,
    seq_lengths,
    logits,
    loss_function=None,
    norm_by_seq_lengths=True
):
    '''
    Compute the cross-entropy loss between all elements in x and logits.
    Masks out the loss for all positions greater than the sequence
    length (as we expect that sequences may be padded).

    Optionally, also either use a different loss function (eg: sampled
    softmax), and/or normalise the loss for each sequence by the
    sequence length.
    '''
    batch_size = tf.shape(x)[0]
    labels = tf.reshape(x, [-1])

    max_doc_length = tf.reduce_max(seq_lengths)
    mask = tf.less(
        tf.range(0, max_doc_length, 1),
        tf.reshape(seq_lengths, [batch_size, 1])
    )
    mask = tf.reshape(mask, [-1])
    mask = tf.to_float(tf.where(
        mask,
        tf.ones_like(labels, dtype=tf.float32),
        tf.zeros_like(labels, dtype=tf.float32)
    ))

    if loss_function is None:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels
        )
    else:
        loss = loss_function(logits, labels)
    loss *= mask
    loss = tf.reshape(loss, [batch_size, -1])
    loss = tf.reduce_sum(loss, axis=1)
    if norm_by_seq_lengths:
        loss = loss / tf.to_float(seq_lengths)
    return tf.reduce_mean(loss)


class DocNADE(object):
    def __init__(self, x, seq_lengths, params):
        self.x = x
        self.seq_lengths = seq_lengths
        batch_size = tf.shape(x)[0]

        # Do an embedding lookup for each word in each sequence
        with tf.device('/cpu:0'):
            # Initialisation scheme taken from the original DocNADE source
            max_embed_init = 1.0 / (params.vocab_size * params.hidden_size)
            W = tf.get_variable(
                'embedding',
                [params.vocab_size, params.hidden_size],
                initializer=tf.random_uniform_initializer(
                    maxval=max_embed_init
                )
            )
            self.embeddings = tf.nn.embedding_lookup(W, x)

        # Compute the hidden layer inputs: each gets summed embeddings of
        # previous words
        def sum_embeddings(previous, current):
            return previous + current

        h = tf.scan(sum_embeddings, tf.transpose(self.embeddings, [1, 2, 0]))
        h = tf.transpose(h, [2, 0, 1])

        bias = tf.get_variable(
            'bias',
            [params.hidden_size],
            initializer=tf.constant_initializer(0)
        )

        # add initial zero vector to each sequence, will then generate the
        # first element using just the bias term
        h = tf.concat([
            tf.zeros([batch_size, 1, params.hidden_size], dtype=tf.float32), h
        ], axis=1)
        h = h[:, :-1, :]
        self.pre_act = h

        # Apply activation
        if params.activation == 'sigmoid':
            h = tf.sigmoid(h + bias)
        elif params.activation == 'tanh':
            h = tf.tanh(h + bias)
        else:
            raise NotImplemented

        # Extract final state for each sequence in the batch
        indices = tf.stack([
            tf.range(batch_size),
            tf.to_int32(seq_lengths) - 1
        ], axis=1)
        self.h = tf.gather_nd(h, indices)

        # Softmax logits
        h = tf.reshape(h, [-1, params.hidden_size])

        if not params.num_samples:
            self.logits = linear(h, params.vocab_size, 'softmax')
            loss_function = None
        else:
            self.logits = linear(h, params.num_samples, 'softmax')
            w_t = tf.get_variable(
                "proj_w_t",
                [params.vocab_size, params.num_samples]
            )
            b = tf.get_variable("proj_b", [params.vocab_size])
            self.proj_w = tf.transpose(w_t)
            self.proj_b = b

            def sampled_loss(logits, labels):
                labels = tf.reshape(labels, [-1, 1])
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.nn.sampled_softmax_loss(
                    weights=local_w_t,
                    biases=local_b,
                    labels=labels,
                    inputs=local_inputs,
                    num_sampled=params.num_samples,
                    num_classes=params.vocab_size,
                    partition_strategy='div'
                )
            loss_function = sampled_loss

        # Compute the loss. If using sampled softmax for training, use full
        # softmax for evaluation and validation
        if not params.num_samples:
            self.loss = masked_sequence_cross_entropy_loss(
                x,
                seq_lengths,
                self.logits
            )
        else:
            projected_logits = \
                tf.matmul(self.logits, self.proj_w) + self.proj_b
            self.loss = masked_sequence_cross_entropy_loss(
                x,
                seq_lengths,
                projected_logits
            )

        self.opt_loss = masked_sequence_cross_entropy_loss(
            x,
            seq_lengths,
            self.logits,
            loss_function=loss_function,
            norm_by_seq_lengths=False
        )

        # Optimiser
        step = tf.Variable(0, trainable=False)
        self.opt = gradients(
            opt=tf.train.AdamOptimizer(learning_rate=params.learning_rate),
            loss=self.opt_loss,
            vars=tf.trainable_variables(),
            step=step
        )
