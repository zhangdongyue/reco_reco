#!/usr/local/bin/python2.7
# -*- coding:utf-8 -*-
# coding:utf-8

import tensorflow as tf
import numpy as np
import sys,os

import input_data

def mnist_train():
  train_rate = 0.005

  try :
    print "----------";
    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder("float", [None, 10])
    print "----------";

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(train_rate).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    sess = tf.Session()

    sess.run(init)

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print "----------";

    for i in range(2000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

  except Exception, e:
    print "Catch exception:",e

def main(argv):
  mnist_train()  

if __name__ == '__main__' : main(sys.argv)
