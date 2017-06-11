import tensorflow as tf
from tensorflow.contrib.layers import flatten


def LeNet(x):
    mu = 0
    sigma = 0.08

    def fc(input, input_size, output_size):
        fc_W = tf.Variable(tf.truncated_normal(shape=(input_size, output_size), mean=mu, stddev=sigma), name="w")
        fc_b = tf.Variable(tf.zeros(output_size), name="b")
        return tf.matmul(input, fc_W) + fc_b

    # Convolutional Layer 1. Input = 32x32x3. Output = 28x28x6.
    with tf.name_scope("conv1"):
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma), name="w")
        conv1_b = tf.Variable(tf.zeros(6), name="b")
        conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
        conv1 = tf.nn.relu(conv1)

    # Pooling Layer 1. Input = 28x28x6. Output = 14x14x6.
    with tf.name_scope("pool1"):
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Convolutional Layer 2. Output = 10x10x16.
    with tf.name_scope("conv2"):
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma), name="w")
        conv2_b = tf.Variable(tf.zeros(16), name="b")
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    with tf.name_scope("pool2"):
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    with tf.name_scope("flatten"):
        fc0 = flatten(conv2)

    with tf.name_scope("fc1"):
        fc1 = fc(fc0, 400, 120)
        fc1 = tf.nn.relu(fc1)

    with tf.name_scope("fc2"):
        fc2 = fc(fc1, 120, 84)
        fc2 = tf.nn.relu(fc2)

    with tf.name_scope("fc3"):
        logits = fc(fc2, 84, 43)

    return logits
