'''
Tensorflow implementation of PV-DM algorithm as a scikit-learn like model
with fit, transform methods.

@author: Zichen Wang (wangzc921@gmail.com)
@author: Sergey Ivanov (sergei.ivanov@skolkovotech.ru -- Adaptation for graph2vec via AW


@references:

https://github.com/wangz10/tensorflow-playground/blob/master/doc2vec.py
http://arxiv.org/abs/1405.4053
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import math
import random
import json
import argparse
import sys
import time
import re
import threading
import multiprocessing
import pandas

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin

from preprocessing import *

SEED = 2018

class Doc2Vec(BaseEstimator, TransformerMixin):

    def __init__(self,
                 comments,
                 reactions,

                 batch_size=64,
                 window_size=4,
                 concat=False,
                 embedding_size_w=64,
                 embedding_size_d=64,
                 loss_type='sampled_softmax',
                 num_samples=50,
                 optimize='Adagrad',
                 learning_rate=1.0,
                 epochs = 1,
                 candidate_func = None):

        # bind params to class
        self.comments = comments
        self.reactions = reactions
        self.batch_size = batch_size
        self.window_size = window_size
        self.concat = concat
        self.embedding_size_w = embedding_size_w
        self.embedding_size_d = embedding_size_d
        self.loss_type = loss_type
        self.num_samples = num_samples
        self.optimize = optimize
        self.learning_rate = learning_rate
        self.candidate_func = candidate_func
        self.epochs = epochs

        # determine dimensions
        self.wdict, self.reverse_wdict, N = build_dataset(comments)
        with open('output/wdict.json', 'w+') as f:
            json.dump(self.wdict, f)
        self.vocabulary_size = N + 1
        print('# words: {}'.format(N))
        self.document_size = reactions.shape[1] + 1
        print('# documents: {}'.format(self.document_size))

        # init all variables in a tensorflow graph
        self._init_graph()



        # create a session
        self.sess = tf.Session(graph=self.graph)


    def _init_graph(self):
        '''
        Init a tensorflow Graph containing:
        input data, variables, model, loss function, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(SEED)

            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.window_size*2+1])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            # Variables.
            # embeddings for words, W in paper
            self.word_embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size_w], -1.0, 1.0))

            # embedding for documents (can be sentences or paragraph), D in paper
            self.doc_embeddings = tf.Variable(
                tf.random_uniform([self.document_size, self.embedding_size_d], -1.0, 1.0))

            if self.concat: # concatenating word vectors and doc vector
                combined_embed_vector_length = self.embedding_size_w * self.window_size * 2+ self.embedding_size_d
            else: # concatenating the average of word vectors and the doc vector
                combined_embed_vector_length = self.embedding_size_w + self.embedding_size_d

            # softmax weights, W and D vectors should be concatenated before applying softmax
            self.weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, combined_embed_vector_length],
                    stddev=1.0 / math.sqrt(combined_embed_vector_length)))
            # softmax biases
            self.biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Model.
            # Look up embeddings for inputs.
            # shape: (batch_size, embeddings_size)
            embed = [] # collect embedding matrices with shape=(batch_size, embedding_size)
            if self.concat:
                for j in range(2*self.window_size):
                    embed_w = tf.nn.embedding_lookup(self.word_embeddings, self.train_dataset[:, j])
                    embed.append(embed_w)
            else:
                # averaging word vectors
                embed_w = tf.zeros([self.batch_size, self.embedding_size_w])
                for j in range(2*self.window_size):
                    embed_w += tf.nn.embedding_lookup(self.word_embeddings, self.train_dataset[:, j])
                embed.append(embed_w)

            embed_d = tf.nn.embedding_lookup(self.doc_embeddings, self.train_dataset[:, 2*self.window_size])
            embed.append(embed_d)
            # concat word and doc vectors
            self.embed = tf.concat(embed, 1)

            # choosing negative sampling function
            sampled_values = None # log uniform by default
            if self.candidate_func == 'uniform': # change to uniform
                sampled_values = tf.nn.uniform_candidate_sampler(
                    true_classes=tf.to_int64(self.train_labels),
                    num_true=1,
                    num_sampled=self.num_samples,
                    unique=True,
                    range_max=self.vocabulary_size)

            # Compute the loss, using a sample of the negative labels each time.
            if self.loss_type == 'sampled_softmax':
                loss = tf.nn.sampled_softmax_loss(self.weights, self.biases, self.train_labels,
                                                  self.embed,
                                                  self.num_samples,
                                                  self.vocabulary_size,
                                                  sampled_values = sampled_values)
            elif self.loss_type == 'nce':
                loss = tf.nn.nce_loss(self.weights, self.biases, self.train_labels,
                                     self.embed, self.num_samples, self.vocabulary_size,
                                     sampled_values=sampled_values)

            self.loss = tf.reduce_mean(loss)

            # Optimizer.
            if self.optimize == 'Adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
            elif self.optimize == 'SGD':
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

            # Compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance:
            norm_w = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings), 1, keep_dims=True))
            self.normalized_word_embeddings = self.word_embeddings / norm_w

            norm_d = tf.sqrt(tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True))
            self.normalized_doc_embeddings = self.doc_embeddings / norm_d

            # init op
            self.init_op = tf.global_variables_initializer()
            # create a saver
            self.saver = tf.train.Saver()

    def _train_thread_body(self):
        gen = generate_batch(self.comments, self.reactions, self.vocabulary_size - 1, self.wdict, self.batch_size, self.window_size)
        for (batch_data, batch_labels) in gen:
            feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
            op, l = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
            self.batch_number += 1
            self.global_step += 1

            # print('Thread: {}, Doc-id: {}, Samples: {}/{}'.format(threading.currentThread().getName(), self.doc_id, self.sample, self.samples))

            self.average_loss += l
            if self.global_step % 1000 == 0:
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (self.global_step, self.average_loss))
                self.average_loss = 0

    def train(self):
        # with self.sess as session:
        session = self.sess

        session.run(self.init_op)

        self.average_loss = 0
        self.global_step = 0
        self.batch_number = 0
        print('Initialized')
        for _ in range(self.epochs):
            print('Epoch: {}'.format(_))
            self._train_thread_body()
            D, W, weights, biases = session.run([self.normalized_doc_embeddings, self.normalized_word_embeddings, self.weights, self.biases])
            np.save('output/doc_embeddings', D)
            np.save('output/word_embeddings', W)
            np.save('output/weights', weights)
            np.save('output/biases', biases)

        return self



if __name__ == '__main__':

    # Set random seeds
    SEED = 2018
    random.seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv('data/train_clean.csv')
    comments = df['comment_text']
    reactions = df.iloc[:, 2:].as_matrix()

    d2v = Doc2Vec(comments, reactions)

    d2v.train()

    console = []