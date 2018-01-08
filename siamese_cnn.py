import numpy as np
import tensorflow as tf
import data_processing as dp
import cv2
import random as rand

#128 for training
bSz = 20
imgSz = 105
learning_rate = 1e-4
trainingSz = 30000

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
conf = tf.sigmoid(logits)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

print('Building Data')
#train_pairs, train_labels = dp.get_data_paths(trainingSz, 8, 'train')
train_pairs, train_labels = dp.get_data_paths2(trainingSz, 8, 'train', 0.5)
#train_pairs, train_labels = dp.get_data(trainingSz, 8, 'train')
test_pairs, test_labels = dp.get_test_data()
#vali_pairs, vali_labels = dp.get_data(10000, 0, 'validate')

print("doing nn stuff")
saver.restore(sess, "../tmp/30k_higher_new_2.ckpt")
#50 to 80 epochs
#this does the training

# for e in xrange(120):
#     sumL = 0.0
#     for i in xrange(trainingSz // bSz):
#         imgs1 = []
#         imgs2 = []
#         for j in xrange(i*bSz, (i+1)*bSz):
#             pair = dp.get_image_pair(train_pairs[j])
#             #print(np.sum(pair[0]))
#             #pair = train_pairs[j]
#             imgs1 += [pair[0]]
#             imgs2 += [pair[1]]
#         imgs1 = np.array(imgs1).reshape(bSz, imgSz, imgSz, 1)
#         imgs2 = np.array(imgs2).reshape(bSz, imgSz, imgSz, 1)
#         y = np.array(train_labels[i*bSz:(i+1)*bSz]).reshape(bSz, 1)
#         _, l = sess.run([train, loss], feed_dict={imgBatchA : imgs1, imgBatchB : imgs2, labels: y})
#         sumL += l
#     save_path = saver.save(sess, "../tmp/30k_higher_new_2.ckpt")
#     print(e, sumL)
#     zipped = list(zip(train_pairs, train_labels))
#     rand.shuffle(zipped)
#     train_pairs, train_labels = zip(*zipped)

right = 0
trials = 0
for i in xrange(8000 // bSz):
    imgs1 = []
    imgs2 = []
    for j in range(i*bSz, (i+1)*bSz):
        pair = test_pairs[j]
        imgs1 += [pair[0]]
        imgs2 += [pair[1]]
    imgs1 = np.array(imgs1).reshape(bSz, imgSz, imgSz, 1)
    imgs2 = np.array(imgs2).reshape(bSz, imgSz, imgSz, 1)
    y = np.array(test_labels[i*bSz:(i+1)*bSz]).reshape(bSz, 1)
    c = sess.run(conf, feed_dict={imgBatchA : imgs1, imgBatchB : imgs2})
    l = np.argmax(c)
    trials += 1
    #print(y)
    if l == y[0]:
        print("hurray!!")
        right += 1
    print(c.shape, l)
print right / float(trials)