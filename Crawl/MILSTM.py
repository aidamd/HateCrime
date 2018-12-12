import numpy as np
import tensorflow as tf
import os, copy
from tensorflow.contrib.layers import fully_connected
from utils import  *
from tensorflow.python.ops import array_ops
from sklearn.metrics import f1_score

class MILSTM():
    def __init__(self, params, vocab, my_embeddings=None):
        self.params = params
        self.vocab = vocab
        for key in params:
            setattr(self, key, params[key])
        if self.pretrain:
            self.my_embeddings = my_embeddings

    def build_embedding(self):
        if self.pretrain:
            embeddings = tf.Variable(tf.constant(0.0, shape=[len(self.vocab), self.embedding_size]),
                                     trainable=False, name="W")

            embedding_placeholder = tf.placeholder(tf.float32, [len(self.vocab), self.embedding_size])
            embedding_init = embeddings.assign(embedding_placeholder)
        else:
            embedding_placeholder = tf.get_variable("embedding",
                                                    initializer=tf.random_uniform(
                                                        [len(self.vocab), self.embedding_size], -1, 1),
                                                    dtype=tf.float32)
        return embedding_placeholder

    def build(self):
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
        self.embedding_placeholder = self.build_embedding()
        self.sequence_length = tf.placeholder(tf.int64, [None])
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)
        self.output = tf.placeholder(tf.int64, [None])
        self.batch_size = tf.placeholder(tf.int64)

        cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)

        self.keep_prob = tf.placeholder(tf.float32)

        cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
        self.network = tf.contrib.rnn.MultiRNNCell([cell_drop] * self.num_layers)

        bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(self.network, self.network, self.embed,
                                                              dtype=tf.float32, sequence_length=self.sequence_length)
        fw_outputs, bw_outputs = bi_outputs
        fw_states, bw_states = bi_states

        self.state = tf.reshape(tf.concat([fw_states, bw_states], 2), [-1, 2 * self.hidden_size])

        fc_drop = fully_connected(self.state, 1)
        self.predictions = tf.reshape(fc_drop, [-1])

        zero_padding = tf.zeros([1000 - tf.shape(self.predictions)[0]])
        pred_padded = tf.reshape(tf.concat([self.predictions, zero_padding], 0), [1, 1000])
        W2 = tf.get_variable(name="W", shape=[1000, 2], initializer=tf.random_normal_initializer(0, 0.01))

        self.predictions = tf.matmul(pred_padded, W2)

        self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output,
                                                                       logits=self.predictions)
        self.loss = tf.reduce_mean(self.xentropy)

        self.predicted_label = tf.argmax(self.predictions, 1)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predicted_label, self.output), tf.float32))

        self.training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def run_model(self, batches, dev_batches, test_batches):
        init = tf.global_variables_initializer()
        with tf.Session() as self.sess:
            init.run()
            epoch = 1
            while True:
                ## Train
                epoch_loss = float(0)
                epoch += 1
                train_correct = 0
                for batch in batches:
                    X_batch, X_len, y_batch = batch
                    feed_dict = {self.train_inputs: X_batch,
                                self.sequence_length: X_len,
                                self.keep_prob: self.keep_ratio,
                                self.output: [y_batch]
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    _, loss_val, predictions_= self.sess.run([self.training_op, self.loss, self.predicted_label], feed_dict= feed_dict)
                    if predictions_ == y_batch:
                        train_correct += 1
                    epoch_loss += loss_val
                ## Test
                test_correct = 0
                y_true = list()
                y_pred = list()
                for X_batch, X_len, y_batch in test_batches:
                    feed_dict = {self.train_inputs: X_batch,
                                 self.sequence_length: X_len,
                                 self.keep_prob: 1,
                                 self.output: [y_batch]
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    predictions_ = self.predicted_label.eval(feed_dict=feed_dict)
                    y_pred.append(predictions_)
                    y_true.append(y_batch)
                    if predictions_ == y_batch:
                        test_correct += 1
                print(epoch, "Train accuracy:", train_correct / len(batches),
                      "Loss: ", epoch_loss / float(len(batches)),
                      "Test accuracy: ", test_correct / len(test_batches),
                      "F1: ", f1_score(y_true, y_pred))

                if epoch == self.epochs:
                    break
        return