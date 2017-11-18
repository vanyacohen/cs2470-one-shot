import numpy as np
import tensorflow as tf
import data_processing as dp

bSz = 128
imgSz = 105
learning_rate = 1e-4

imgBatchA = tf.placeholder(tf.float32, [bSz, imgSz, imgSz, 1])
imgBatchB = tf.placeholder(tf.float32, [bSz, imgSz, imgSz, 1])
labels = tf.placeholder(tf.float32, [bSz, 1])

def cnn(imgBatch):
    conv1 = tf.layers.conv2d(
        inputs=imgBatch,
        filters=64,
        kernel_size=[10, 10],
        padding="valid",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[7, 7],
        padding="valid",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[4, 4],
        padding="valid",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=256,
        kernel_size=[4, 4],
        padding="valid",
        activation=tf.nn.relu)

    conv4_flat = tf.reshape(conv4, [bSz, 6 * 6 * 256])
    feature_vector = tf.layers.dense(inputs=conv4_flat, units=4096, activation=tf.nn.sigmoid)
    return feature_vector

L1_distance_vector = tf.abs(tf.subtract(cnn(imgBatchA), cnn(imgBatchB)))
logits = tf.layers.dense(inputs=L1_distance_vector, units=1)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_pairs, train_labels = dp.get_data(3000, 8, 'train')
#test_pairs, test_labels = dp.get_data(10000, 0, 'test')
#vali_pairs, vali_labels = dp.get_data(10000, 0, 'validate')

print "doing nn stuff"
for i in xrange(3000 // bSz):
    imgs1 = []
    imgs2 = []
    for j in range(i*bSz, (i+1)*bSz):
        pair = train_pairs[j]
        imgs1 += [pair[0]]
        imgs2 += [pair[1]]
    imgs1 = np.array(imgs1).reshape(bSz, imgSz, imgSz, 1)
    imgs2 = np.array(imgs2).reshape(bSz, imgSz, imgSz, 1)
    y = np.array(train_labels[i*bSz:(i+1)*bSz]).reshape(bSz, 1)
    _, l = sess.run([train, loss], feed_dict={imgBatchA : imgs1, imgBatchB : imgs2, labels: y})
    print l
