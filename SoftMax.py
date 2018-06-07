from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
# pylint: enable=unused-import
import tensorflow.examples.tutorials.mnist.input_data as input_data

#  ************ 读取数据 ***************
# 使用 input_data 的 read_data_sets 方法返回数据集，该方法会自动下载
# MNIST_data 文件夹下的训练和测试数据，并将其解压后返回带有 field name 的
# 训练数据，测试数据，和验证数据
# *************************************
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# *********** 为张量创建一个占位符 ************
# 数据类型是 float32 大小为 None 说明为任意大小，784 维
# 该方法允许没有数据的情况下来构造计算图和操作，待后续分配数据(feed data)
# placeholder 必须被 fed 采用 Session.run  的 feed_dict 可选参数
# *****************************************
x = tf.placeholder(tf.float32, [None, 784])

# *********** 创建变量 *********************
# 创建一个所有元素都是 0 的变量， W 为 10 维，784 个向量
# b 是 10 个 1 维向量
# ****************************************
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#define model
y = tf.nn.softmax(tf.matmul(x,W) + b)
#
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
writer = tf.summary.FileWriter("output",sess.graph)
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

writer.close()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

