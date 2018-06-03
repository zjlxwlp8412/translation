# encoding: utf-8

import tensorflow as tf
import numpy as np


def normalize(
    inputs,
    reuse=None,
    scope="normalize"
):
    with tf.variable_scope(scope, reuse=reuse):
        gamma = tf.Variable(tf.ones(tf.shape(inputs[-1:]), dtype=tf.float32))
        beta = tf.Variable(tf.zeros(tf.shape(inputs[-1:]), dtype=tf.float32))
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        normalized = (inputs - mean) / (variance + 1e-8) ** 0.5
        return gamma * normalized + beta


def embedding(
    inputs,
    vocab_size,
    num_units,
    zero_pad=True,
    scale=True,
    reuse=None,
    scope="embedding"
):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable(
            name='lookup_table',
            dtype=tf.float32,
            shape=[vocab_size, num_units],
            initializer=tf.truncated_normal_initializer(stddev=0.1, mean=0.001)
        )
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        if scale:
            outputs *= num_units ** 0.5
        return outputs


def position_encoding(
    batch_size,
    max_len,
    num_units,
    zero_pad=True,
    scale=True,
    reuse=None,
    scope="position_encoding"
):
    with tf.variable_scope(scope, reuse=reuse):
        positions = tf.tile(tf.expand_dims(tf.range(max_len), 0), [batch_size, 1])
        lookup_table = np.array(
            [[pos / (10000.0 ** (2.0 * i / num_units)) for i in range(num_units)] for pos in range(max_len)]
        )
        lookup_table[:, 0::2] = np.sin(lookup_table[:, 0::2])
        lookup_table[:, 1::2] = np.cos(lookup_table[:, 1::2])
        lookup_table = tf.convert_to_tensor(lookup_table, dtype=tf.float32)
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, positions)
        if scale:
            outputs *= num_units ** 0.5
        return outputs


def multi_head_attention(
    queries,
    keys,
    values,
    num_units,
    num_heads,
    dropout_rate,
    is_training=True,
    future_mask=False,
    reuse=None,
    scope="multi_head_attention"
):
    with tf.variable_scope(scope, reuse=reuse):
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # [n, L_q, d]
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # [n, L_k, d]
        V = tf.layers.dense(values, num_units, activation=tf.nn.relu) # [n, L_v, d]
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # [h * n, L_q, d / h]
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # [h * n, L_k, d / h]
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # [h * n, L_v, d / h]
        outputs = tf.matmul(Q_, tf.transpose(K_, perm=[0, 2, 1])) * (num_units / num_heads) ** (-0.5) # [h * n, L_q, L_k]

        key_mask = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        key_mask = tf.tile(tf.expand_dims(key_mask, axis=-1), [num_heads, 1, tf.shape(queries)[1]])
        outputs = tf.where(tf.equal(key_mask, 0), tf.ones_like(outputs) * (-1e9), outputs)  # [h * n, L_q, L_k]

        if future_mask: # future_mask
            future_mask = tf.ones_like(outputs[0, :, :])
            future_mask = tf.contrib.linalg.LinearOperatorTriL(future_mask).to_dense()
            future_mask = tf.expand_dims(future_mask, axis=0)
            future_mask = tf.tile(future_mask, [tf.shape(outputs)[0], 1, 1])
            outputs = tf.where(tf.equal(future_mask, 0), tf.ones_like(outputs) * (-1e9), outputs)  # [h * n, L_q, L_k]

        outputs = tf.nn.softmax(outputs)  # [h * n, L_q, L_k]

        query_mask = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))# [n, L_q, 1]
        query_mask = tf.tile(tf.expand_dims(query_mask, axis=-1), [num_heads, 1, tf.shape(keys)[1]])
        outputs = tf.where(tf.equal(query_mask, 0), tf.zeros_like(outputs), outputs)  # [h * n, L_q, L_k]

        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        outputs = tf.matmul(outputs, V_) # [h * n, L_q, d / h]
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        outputs += queries
        outputs = normalize(outputs)
        return outputs


def feed_forward(
    inputs, # [batch, max_len, num_units]
    num_units=None,
    reuse=None,
    scope="feed_forward"
):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.conv1d(
            inputs=inputs,
            filters=num_units[0],
            kernel_size=1,
            activation=tf.nn.relu
        )
        outputs = tf.layers.conv1d(
            inputs=outputs,
            filters=num_units[1],
            kernel_size=1,
            activation=None
        )
        outputs += inputs
        outputs = normalize(outputs)
        return outputs


def label_smoothing(inputs, epsilon):
    return ((1 - epsilon) * inputs) + (epsilon / (inputs.get_shape().as_list()[-1]))


def save_model(sess, name):
    saver = tf.train.Saver()
    saver.save(sess, name)


def load_model(sess, name):
    saver = tf.train.Saver()
    saver.restore(sess, name)


def get_loss(model_outputs, target_inputs, target_vocab_size):
    target_inputs_smoothed = label_smoothing(tf.one_hot(target_inputs, depth=target_vocab_size), epsilon=0.1)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=model_outputs,
        labels=target_inputs_smoothed
    )
    flag = tf.to_float(tf.not_equal(target_inputs, 0))
    return tf.reduce_sum(loss * flag) / (tf.reduce_sum(flag))


def get_accuracy(preds, target_inputs):
    flag = tf.to_float(tf.not_equal(target_inputs, 0))
    return tf.reduce_sum(tf.to_float(tf.equal(preds, target_inputs)) * flag) / (tf.reduce_sum(flag))


def get_train_op(loss, global_step, learning_rate):
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9).minimize(loss, global_step=global_step)
    return train_op


def encoder(params, inputs, vocab_size, reuse=None, scope="encoder"):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = embedding(inputs, vocab_size, params.num_units)
        outputs += embedding(
            tf.tile(
                tf.expand_dims(tf.range(params.max_len), 0),
                [tf.shape(inputs)[0], 1]
            ),
            vocab_size=params.max_len,
            num_units=params.num_units,
            zero_pad=False,
            scale=False,
            scope="position_encoding"
        )
        outputs = tf.layers.dropout(
            outputs,
            rate=params.dropout_rate,
            training=tf.convert_to_tensor(params.is_training)
        )
        for i in range(params.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                outputs = multi_head_attention(
                    outputs,
                    outputs,
                    outputs,
                    params.num_units,
                    params.num_heads,
                    dropout_rate=params.dropout_rate,
                    is_training=params.is_training
                )
                outputs = feed_forward(outputs, num_units=[4 * params.num_units, params.num_units])
        return outputs

def decoder(params, inputs, encoder_outputs, vocab_size, reuse=None, scope="decoder"):
    with tf.variable_scope(scope, reuse=reuse):
        decoder_inputs = tf.concat((tf.ones_like(inputs[:, :1]) * 2, inputs[:, :-1]), axis=-1)  # 2:<S>
        outputs = embedding(decoder_inputs, vocab_size, params.num_units)
        outputs += embedding(
            tf.tile(
                tf.expand_dims(tf.range(params.max_len), 0),
                [tf.shape(decoder_inputs)[0], 1]
            ),
            vocab_size=params.max_len,
            num_units=params.num_units,
            zero_pad=False,
            scale=False,
            scope="position_encoding"
        )
        outputs = tf.layers.dropout(
            outputs,
            rate=params.dropout_rate,
            training=tf.convert_to_tensor(params.is_training)
        )
        for i in range(params.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                outputs = multi_head_attention(
                    outputs,
                    outputs,
                    outputs,
                    params.num_units,
                    params.num_heads,
                    dropout_rate=params.dropout_rate,
                    is_training=params.is_training,
                    future_mask=True,
                    scope="self_attention"
                )
                outputs = multi_head_attention(
                    outputs,
                    encoder_outputs,
                    encoder_outputs,
                    params.num_units,
                    params.num_heads,
                    dropout_rate=params.dropout_rate,
                    is_training=params.is_training,
                    scope="other_attention"
                )
                outputs = feed_forward(outputs, num_units=[4 * params.num_units, params.num_units])
        outputs = tf.layers.dense(outputs, vocab_size)
        return outputs


