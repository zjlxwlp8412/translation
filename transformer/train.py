# encoding: utf-8

import tensorflow as tf
from transformer.modules import embedding, position_encoding, multi_head_attention, feed_forward, label_smoothing, get_mask
from transformer.load_data import load_data, get_batches, load_vocabs
from transformer import parameters as params
import os
import numpy as np

def encoder(inputs, vocab_size, reuse=None, scope="encoder"):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = embedding(inputs, vocab_size, params.num_units)
        outputs += embedding(
            tf.tile(
                tf.expand_dims(tf.range(params.max_len), 0),
                [tf.shape(inputs)[0], 1]
            ),
            vocab_size=params.max_len,
            num_units=params.num_units,
            scope="position_encoding"
        )
        outputs = tf.layers.dropout(
            outputs,
            rate=params.dropout_rate,
            training=tf.convert_to_tensor(params.is_training)
        )
        pad_mask = get_mask(inputs, params.num_heads, "pad_mask")
        for i in range(params.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                outputs = multi_head_attention(
                    outputs,
                    outputs,
                    outputs,
                    params.num_units,
                    params.num_heads,
                    pad_mask=pad_mask
                )
                outputs = feed_forward(outputs, num_units=[4 * params.num_units, params.num_units])
        return outputs

def decoder(inputs, encoder_outputs, vocab_size, reuse=None, scope="decoder"):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = embedding(inputs, vocab_size, params.num_units)
        outputs += embedding(
            tf.tile(
                tf.expand_dims(tf.range(params.max_len), 0),
                [tf.shape(inputs)[0], 1]
            ),
            vocab_size=params.max_len,
            num_units=params.num_units,
            scope="position_encoding"
        )
        outputs = tf.layers.dropout(
            outputs,
            rate=params.dropout_rate,
            training=tf.convert_to_tensor(params.is_training)
        )
        pad_mask = get_mask(inputs, params.num_heads, "pad_mask")
        future_mask = get_mask(inputs, params.num_heads, "future_mask")
        for i in range(params.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                outputs = multi_head_attention(
                    outputs,
                    outputs,
                    outputs,
                    params.num_units,
                    params.num_heads,
                    pad_mask=pad_mask,
                    future_mask=future_mask,
                    scope="self_attention"
                )
                outputs = multi_head_attention(outputs, encoder_outputs, encoder_outputs, params.num_units, params.num_heads, scope="other_attention")
                outputs = feed_forward(outputs, num_units=[4 * params.num_units, params.num_units])
        outputs = tf.layers.dense(outputs, vocab_size)
        return outputs

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
    istarget = tf.to_float(tf.not_equal(target_inputs, 0))
    mean_loss = tf.reduce_sum(loss * istarget) / (tf.reduce_sum(istarget))
    return mean_loss

def get_accuracy(preds, target_inputs):
    istarget = tf.to_float(tf.not_equal(target_inputs, 0))
    acc = tf.reduce_sum(tf.to_float(tf.equal(preds, target_inputs)) * istarget) / (tf.reduce_sum(istarget))
    return acc

def get_train_op(loss, global_step):
    train_op = tf.train.AdamOptimizer(params.lr, beta1=0.9, beta2=0.98, epsilon=1e-9).minimize(loss, global_step=global_step)
    return train_op

def train(has_saved_model):
    train_source_inputs, train_target_inputs, source_vocab_size, target_vocab_size, _, _ = load_data("train")
    test_source_inputs, test_target_inputs, _, _, _, target_idx2word = load_data("test")

    source_input = tf.placeholder(shape=[params.batch_size, params.max_len], dtype=tf.int32, name="source_input")
    target_input = tf.placeholder(shape=[params.batch_size, params.max_len], dtype=tf.int32, name="target_input")

    global_step = tf.train.get_or_create_global_step()

    encoder_output = encoder(source_input, source_vocab_size)
    decoder_output = decoder(target_input, encoder_output, target_vocab_size)
    pred = tf.to_int32(tf.arg_max(decoder_output, dimension=-1))

    loss = get_loss(decoder_output, target_input, target_vocab_size)
    acc = get_accuracy(pred, target_input)
    train_op = get_train_op(loss, global_step)
    with tf.Session() as sess:
        if has_saved_model:
            load_model(sess, os.path.join(params.checkpoint_dir, params.model_name))
        else:
            sess.run(tf.global_variables_initializer())
        def get_test_acc(test_source_inputs, test_target_inputs, block_size):
            test_acc = 0.0
            for i in range(int(test_source_inputs.shape[0] / block_size)):
                test_acc += acc.eval(
                    feed_dict = {
                        source_input: test_source_inputs[i * block_size : (i + 1) * block_size],
                        target_input: test_target_inputs[i * block_size : (i + 1) * block_size]
                    }
                )
            return block_size * test_acc / test_source_inputs.shape[0]
        best_acc = get_test_acc(test_source_inputs, test_target_inputs, params.batch_size)
        for epoch in range(params.num_epochs):
            batches = get_batches(train_source_inputs, train_target_inputs)
            for batch in batches:
                _, global_step_val, loss_val, acc_val = sess.run(
                    [
                        train_op,
                        global_step,
                        loss,
                        acc
                    ],
                    feed_dict={
                        source_input: batch[0],
                        target_input: batch[1]
                    }
                )
                if global_step_val % 50 == 0:
                    print(
                        'epoch: %d  step: %d   train_loss: %.3f    train_acc: %.3f'
                        %(
                            epoch + 1,
                            global_step_val,
                            loss_val,
                            acc_val
                        )
                    )
                if global_step_val % 100 == 0:
                    inference(pred, source_input, target_input, batch[0], batch[1], target_idx2word)
            test_acc = get_test_acc(test_source_inputs, test_target_inputs, params.batch_size)
            if test_acc > best_acc:
                save_model(sess, os.path.join(params.checkpoint_dir, params.model_name))
                print(
                    'epoch: %d. Congratulations! The model accuracy is improved from %.3f to %.3f. Saved!!!'
                    %(epoch + 1, best_acc, test_acc)
                )
                best_acc = test_acc
            else:
                print(
                    'epoch: %d  test_acc: %.3f No improvement.'
                    %(epoch + 1, test_acc)
                )

def inference(
    pred,
    source_input,
    target_input,
    test_source_inputs,
    test_target_inputs,
    target_idx2word
):
    params.is_training = False
    preds = np.zeros((params.batch_size, params.max_len), np.int32)
    preds[:,0] = 2
    for i in range(1, params.max_len):
        _preds = pred.eval(feed_dict={source_input: test_source_inputs, target_input: preds})
        preds[:, i] = _preds[:, i]

    ct = 0
    for i in range(params.batch_size):
        if ct >= 10:break
        pred = " ".join([target_idx2word[idx] for idx in preds[i, :]])
        true = " ".join([target_idx2word[idx] for idx in test_target_inputs[i, :]])
        print("预测：" + pred)
        print("答案：" + true)
        print("-------------")
        ct += 1
    params.is_training = True

def main(argv=None):
    if not os.path.exists(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    train(os.path.exists(os.path.join(params.checkpoint_dir, "checkpoint")))

if __name__ == '__main__':
    tf.app.run()
