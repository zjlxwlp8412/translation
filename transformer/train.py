# encoding: utf-8

import tensorflow as tf
from transformer.modules import encoder, decoder, get_loss, get_accuracy, get_train_op, load_model, save_model
from transformer.load_data import load_data, get_batches
from transformer import parameters as params
import os
import numpy as np


def train(has_saved_model):
    train_source_inputs, train_target_inputs, source_vocab_size, target_vocab_size, _, _, _ = load_data("train")
    test_source_inputs, test_target_inputs, _, _, _, target_idx2word, source_idx2word = load_data("test")
    source_input = tf.placeholder(shape=[params.batch_size, params.max_len], dtype=tf.int32, name="source_input")
    target_input = tf.placeholder(shape=[params.batch_size, params.max_len], dtype=tf.int32, name="target_input")
    global_step = tf.train.get_or_create_global_step()
    encoder_output = encoder(params, source_input, source_vocab_size)
    decoder_output = decoder(params, target_input, encoder_output, target_vocab_size)
    pred = tf.to_int32(tf.arg_max(decoder_output, dimension=-1))
    loss = get_loss(decoder_output, target_input, target_vocab_size)
    acc = get_accuracy(pred, target_input)
    learning_rate = tf.multiply(tf.minimum(1.0 / tf.sqrt(tf.to_float(global_step)), tf.multiply(tf.to_float(global_step), 4000.0 ** (-1.5))), (params.num_units ** (-0.5)))
    train_op = get_train_op(loss, global_step, learning_rate=learning_rate)
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333))) as sess:
        if has_saved_model:
            load_model(sess, os.path.join(params.checkpoint_dir, params.model_name))
        else:
            sess.run(tf.global_variables_initializer())

        def get_test_loss(test_source_inputs, test_target_inputs, block_size):
            test_loss = 0.0
            for i in range(int(test_source_inputs.shape[0] / block_size)):
                test_loss += loss.eval(
                    feed_dict = {
                        source_input: test_source_inputs[i * block_size: (i + 1) * block_size],
                        target_input: test_target_inputs[i * block_size: (i + 1) * block_size]
                    }
                )
            return test_loss
        best_loss = get_test_loss(test_source_inputs, test_target_inputs, params.batch_size)
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
                if global_step_val % 500 == 0:
                    inference(pred, source_input, target_input, batch[0], batch[1], target_idx2word, source_idx2word)
            test_loss = get_test_loss(test_source_inputs, test_target_inputs, params.batch_size)
            if test_loss < best_loss:
                save_model(sess, os.path.join(params.checkpoint_dir, params.model_name))
                print(
                    'epoch: %d. Congratulations! The model accuracy is improved from %.3f to %.3f. Saved!!!'
                    %(epoch + 1, best_loss, test_loss)
                )
                best_loss = test_loss
            else:
                print(
                    'epoch: %d  test_acc: %.3f No improvement.'
                    %(epoch + 1, test_loss)
                )

def inference(
    pred,
    source_input,
    target_input,
    test_source_inputs,
    test_target_inputs,
    target_idx2word,
    source_idx2word
):
    params.is_training = False
    preds = np.zeros((params.batch_size, params.max_len), np.int32)
    for i in range(0, params.max_len):
        _preds = pred.eval(feed_dict={source_input: test_source_inputs, target_input: preds})
        preds[:, i] = _preds[:, i]

    ct = 0
    for i in range(params.batch_size):
        if ct >= 10: break
        source_true = " ".join([source_idx2word[idx] for idx in test_source_inputs[i, :]])
        target_true = " ".join([target_idx2word[idx] for idx in test_target_inputs[i, :]])
        target_pred = " ".join([target_idx2word[idx] for idx in preds[i, :]])
        print("中文：" + source_true)
        print("英文：" + target_true)
        print("机器翻译：" + target_pred)
        print("-------------")
        ct += 1
    params.is_training = True

def main(argv=None):
    if not os.path.exists(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    train(os.path.exists(os.path.join(params.checkpoint_dir, "checkpoint")))

if __name__ == '__main__':
    tf.app.run()
