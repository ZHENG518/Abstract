import numpy as np
import tensorflow as tf


def linear_distribution(num_epoch):
    return np.linspace(start=1.0, stop=0.0, num=num_epoch, dtype=np.float32)


def loss_function(real, pred, pad_id):
    mask = tf.math.logical_not(tf.math.equal(real[:, 1:], pad_id))  # <PAD>位置为FALSE
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
    # logits表示网络的直接输出,没经过sigmoid或者softmax的概率化。from_logits=False就表示把已经概率化了的输出
    loss = loss_object(real[:, 1:], pred[:, 1:, :])  # loss[batch_size, seq_len-1] 忽略<BOS>

    mask = tf.cast(mask, loss.dtype)  # <PAD>位置为0
    loss *= mask  # loss[batch_size, seq_len]<PAD>位置loss为0

    return tf.reduce_mean(loss)  # scaler


def coverage_loss(att_list, cov_list):
    coverage = tf.zeros_like(att_list[0])
    cover_losses = []
    for a in att_list:  # a[batch_size, x_seq_len] * y_seq_len
        cov_loss = tf.reduce_sum(tf.minimum(a, coverage), axis=1)  # [batch_size,1]
        cover_losses.append(cov_loss)  # [batch_size,1] * y_seq_len
        coverage += a
    # change from[y_seq_len, batch_sz] to [batch_sz, y_seq_len]
    cover_losses = tf.stack(cover_losses, 1)
    loss = tf.reduce_mean(tf.reduce_mean(cover_losses, axis=0))  # mean loss of each time step and then sum up

    return loss
