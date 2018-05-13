# encoding: utf-8

import tensorflow as tf
import numpy as np
from transformer import parameters as params

def normalize(inputs, reuse=None, scope="normalize"):
    with tf.variable_scope(scope, reuse=reuse):
        gamma = tf.Variable(tf.ones(tf.shape(inputs[-1:])))
        beta = tf.Variable(tf.zeros(tf.shape(inputs[-1:])))
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        normalized = (inputs - mean) / (variance + 1e-8) ** (0.5)
        return gamma * normalized + beta

def embedding(inputs, vocab_size, num_units, zero_pad=False, scale=False, reuse=None, scope="embedding"):
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

def get_mask(inputs, num_heads, mask_type):
    """
    :param inputs: [n, l]
    :return:[h * n, l, l]
    """
    inputs_shape = tf.shape(inputs)
    if mask_type == "pad_mask":
        mask = tf.to_float(tf.sign(tf.abs(inputs)))
        mask = tf.tile(tf.expand_dims(mask, axis=-1), [num_heads, 1, inputs_shape[1]])
    elif mask_type == "future_mask":
        mask = tf.to_float(tf.ones(shape=[inputs_shape[1], inputs_shape[1]]))
        mask = tf.contrib.linalg.LinearOperatorTriL(mask).to_dense()
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.tile(mask, [inputs_shape[0] * num_heads, 1, 1])
    else:
        mask = None
    return mask

def position_encoding(batch_size, max_len, num_units, zero_pad=True, scale=True, reuse=None, scope="position_encoding"):
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
        pad_mask=None,
        future_mask=None,
        reuse=None,
        scope="multi_head_attention"
):
    with tf.variable_scope(scope, reuse=reuse):
        Q = tf.layers.dense(queries, num_units, activation=None) # [n, L_q, d]
        K = tf.layers.dense(keys, num_units, activation=None) # [n, L_k, d]
        V = tf.layers.dense(values, num_units, activation=None) # [n, L_v, d]
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # [h * n, L_q, d / h]
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # [h * n, L_k, d / h]
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # [h * n, L_v, d / h]
        outputs = tf.matmul(Q_, tf.transpose(K_, perm=[0, 2, 1])) * (num_units / num_heads) ** (-0.5) # [h * n, L_q, L_k]
        if pad_mask is not None: # key_mask
            outputs = tf.where(tf.equal(pad_mask, 0), tf.ones_like(outputs) * (-1e9), outputs) # [h * n, L_q, L_k]
        outputs = tf.nn.softmax(outputs) # [h * n, L_q, L_k]
        if future_mask is not None: # future_mask
            outputs = tf.where(tf.equal(future_mask, 0), tf.ones_like(outputs) * (-1e9), outputs) # [h * n, L_q, L_k]
        outputs = tf.matmul(outputs, V_) # [h * n, L_q, d / h]
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        outputs += queries
        outputs = normalize(outputs)
        return outputs

def feed_forward(inputs, num_units=[2048, 512], reuse=None, scope="feed_forward"):
        with tf.variable_scope(scope, reuse=reuse):
            outputs = tf.layers.conv1d(
                inputs=inputs,
                filters=num_units[0],
                kernel_size=1,
                activation=None
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




# x = embedding(tf.reshape(tf.range(5 * 10), [5, 10]), 50, 20)
# x = position_encoding(10, 8, 30)
# x = feed_forward(tf.to_float(tf.reshape(tf.range(32 * 10 * 8), [32, 10, 8])), [40, 8])
# x = label_smoothing(tf.to_float(tf.reshape(tf.range(32 * 10 * 8), [32, 10, 8])), 0.1)
# inputs2 = tf.to_float(tf.reshape(tf.range(10 * 5 * 6), [10, 5, 6]))
# x = tf.contrib.linalg.LinearOperatorTriL(inputs).to_dense()
# inputs = tf.ones(shape=[10, 5], dtype=tf.int32)
# pad_mask = get_self_atten_mask(inputs1, 3, "future_mask")
# x = multi_head_attention(
#     inputs2,
#     inputs2,
#     inputs2,
#     6,
#     3,
#     pad_mask=pad_mask
# )

# x = encoder(inputs, 10, 20, 5, 8, 4, 1, reuse=None, scope="encoder")
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(inputs).shape)
#     print(sess.run(x))



