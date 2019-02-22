import os
import sys
import cv2 as cv
import time
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import src
from src.data_manager import EmojifierDataManager

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def weight_variable(shape):
    initial = tf.truncated_normal(
        shape=shape, stddev=0.1
    )
    
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(
        x, W, strides=[1,1,1,1], padding='SAME'
    )


def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'
    )


def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])

    return tf.nn.relu(tf.layers.batch_normalization(conv2d(input, W) + b))


def full_layer(input, size):
    insize = int(input.get_shape()[1])
    W = weight_variable([insize, size])
    b = bias_variable([size])

    return tf.matmul(input, W) + b


def model(x, keep_prob):

    C1, C2, C3 = 10, 30, 40
    F1 = 512
    C1_1 = conv_layer(x, shape=[3, 3, 1, C1])
    P1_1 = max_pool_2x2(C1_1)
    
    C2_1 = conv_layer(P1_1, shape=[3, 3, C1, C2])
    P2_1 = max_pool_2x2(C2_1)
    
    drop_C12 = tf.nn.dropout(P2_1, keep_prob=keep_prob) 
    
    C3_1 = conv_layer(drop_C12, shape=[3, 3, C2, C3])
    P3_1 = max_pool_2x2(C3_1)

    flat_C3 = tf.reshape(P3_1, [-1, 6*6*C3])
    drop_C3 = tf.nn.dropout(flat_C3, keep_prob=keep_prob)

    FC1 = tf.nn.relu(full_layer(drop_C3, F1))
    drop_FC1 = tf.nn.dropout(FC1, keep_prob=keep_prob)
    
    y_conv = full_layer(drop_FC1, 6)

    return y_conv


def test(emoji_data, sess):
    x = emoji_data.test.images.reshape(120, 60, 48, 48, 1)
    y = emoji_data.test.labels.reshape(120, 60, 6)

    acc = np.mean([
        sess.run(accuracy, feed_dict={X:x[i], Y:y[i], keep_prob:1.0}) \
        for i in range(6)
    ])   
    logger.critical('test-accuracy: {:.4}%'.format(acc*100))


if __name__ == '__main__':

    logger = logging.getLogger('emojifier.model')

    CHECKPOINT_SAVE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'model_checkpoints')

    if not os.path.exists(CHECKPOINT_SAVE_PATH):
        os.makedirs(CHECKPOINT_SAVE_PATH)

    BATCH_SIZE = 128
    STEPS = 1200

    X = tf.placeholder(
        tf.float32, shape=[None, 48, 48, 1]
    )
    
    Y = tf.placeholder(tf.float32, shape=[None,6])
    
    keep_prob = tf.placeholder(tf.float32)

    emoset = EmojifierDataManager()

    logger.info("Number of train images: {}".format(
        len(emoset.train.images)
    ))
    logger.info("Number of train labels: {}".format(
        len(emoset.train.labels)
    ))
    logger.info("Number of test images: {}".format(
        len(emoset.test.images)
    ))
    logger.info("Number of test labels: {}".format(
        len(emoset.test.labels)
    ))

    y_conv = model(X, keep_prob)

    #Cost function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=Y,
            logits=y_conv
        )
    )
    
    #Optimizer
    train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_predictions = tf.equal(
        tf.argmax(y_conv, 1), tf.argmax(Y, 1)
    )
    accuracy = tf.reduce_mean(
        tf.cast(correct_predictions, tf.float32)
    )
    
    saver = tf.train.Saver()
    
    with tf.Session(config=config) as sess:
        
        sess.run(tf.global_variables_initializer())

        for i in range(STEPS):
            x_data, y_data = emoset.train.next_batch(BATCH_SIZE)

            acc, loss, _ = sess.run(
                [accuracy, cross_entropy, train], 
                feed_dict={X:x_data, Y:y_data, keep_prob: 0.7}
            )

            if i % 20 == 0:
                logger.info('step: {}, accuracy: {:.4}%, loss: {:.4}'.format(
                    i, acc*100, loss
                ))

        test(emoset, sess)
        
        save_path = saver.save(sess, os.path.join(CHECKPOINT_SAVE_PATH, 'model.ckpt'))
        
        logger.info("Model saved in path: {}".format(save_path))
    