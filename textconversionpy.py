from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import matplotlib.pyplot as plt

learning_rate=0.001
training_epochs=15
batch_size=100
display_step=1

n_hidden_1=256
n_hidden_2=256
n_input=784
n_classes=10


x=tf.placeholder("float", [None,n_input])
y=tf.placeholder("float", [None,n_classes])


def multilayer_perceptron(x,weights,biases):
    
    layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1=tf.nn.relu(layer_1)
    
    
    layer_2=tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2=tf.nn.relu(layer_2)
    
    out_layer=tf.matmul(layer_2,weights['out']) + biases['out']
    return out_layer

weights= {
     'h1': tf.Variable(tf.random_normal([n_input,n_hidden_1])),
     'h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
     'out': tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}


biases= {
     'b1': tf.Variable(tf.random_normal([n_hidden_1])),
     'b2': tf.Variable(tf.random_normal([n_hidden_2])),
     'out': tf.Variable(tf.random_normal([n_classes]))
}

pred=multilayer_perceptron(x,weights,biases)
