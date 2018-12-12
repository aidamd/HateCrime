import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from sklearn.metrics import f1_score
import numpy as np


class Entity():
    def __init__(self, params, vocab, my_embeddings=None):
        self.params = params
        self.vocab = vocab
        for key in params:
            setattr(self, key, params[key])
        if self.pretrain:
            self.my_embeddings = my_embeddings

    def build_embedding(self):
        if self.pretrain:
            embedding_placeholder = tf.Variable(tf.constant(0.0, shape=[len(self.vocab), self.embedding_size]),
                                     trainable=False, name="W")
        else:
            embedding_placeholder = tf.get_variable("embedding",
                                                    initializer=tf.random_uniform(
                                                        [len(self.vocab), self.embedding_size], -1, 1),
                                                    dtype=tf.float32)
        return embedding_placeholder

    def build(self):
        # input data is in form of [batch_size, article_len, sentence_len]
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None, None], name="inputs")
        self.embedding_placeholder = self.build_embedding()

        # length of each sentence in the whole batch
        self.sequence_length = tf.placeholder(tf.int64, [None])

        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)

        self.keep_ratio = tf.placeholder(tf.float32)

        # target labels corresponding to each article. Shape: [batch_size]
        self.target_group = tf.placeholder(tf.int64, [None])
        # the weight of each target label is 1 - (label frequency) / (all articles)
        self.target_weight = tf.placeholder(tf.float64, [None])

        # action labels corresponding to each article. Shape: [batch_size]
        self.hate_act = tf.placeholder(tf.int64, [None])
        # the weight of each action label is 1 - (label frequency) / (all articles)
        self.act_weight = tf.placeholder(tf.float64, [None])

        cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)

        cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_ratio)
        self.network = tf.contrib.rnn.MultiRNNCell([cell_drop] * self.num_layers)

        shape = tf.shape(self.embed)

        # the inputs are reshaped to [all sentences, sentence_len] to be passed to LSTM
        embed = tf.reshape(self.embed, [shape[0] * shape[1], shape[2], self.embedding_size])
        self.sequence_length = tf.reshape(self.sequence_length, [tf.shape(embed)[0]])

        # Bi-directional LSTM to capture the sentence representation
        bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(self.network, self.network, embed,
                                                                dtype=tf.float32,
                                                                sequence_length=self.sequence_length)
        fw_states, bw_states = bi_states

        state = tf.concat([fw_states, bw_states], 2)

        # vectors are reshaped to form the articles
        state = tf.reshape(state, [shape[0], shape[1], 2 * self.hidden_size])

        state = tf.nn.dropout(state, keep_prob=self.keep_ratio)

        fc_target = fully_connected(state, 8)
        fc_act = fully_connected(state, 4)

        self.high_target = tf.reduce_max(fc_target, axis=[1])
        self.high_act = tf.reduce_max(fc_act, axis=[1])

        t_weight = tf.gather(self.target_weight, self.target_group)
        a_weight = tf.gather(self.act_weight, self.hate_act)

        # weighted losses are calculated
        self.target_xentropy = tf.losses.sparse_softmax_cross_entropy(labels=self.target_group,
                                                                      logits=self.high_target,
                                                                      weights=t_weight)
        self.act_xentropy = tf.losses.sparse_softmax_cross_entropy(labels=self.hate_act,
                                                                   logits=self.high_act,
                                                                   weights=a_weight)

        self.loss = tf.add(self.target_xentropy, self.act_xentropy)

        self.predicted_target = tf.argmax(self.high_target, 1)
        self.predicted_act = tf.argmax(self.high_act, 1)

        self.accuracy_target = tf.reduce_mean(
              tf.cast(tf.equal(self.predicted_target, self.target_group), tf.float32))
        self.accuracy_act = tf.reduce_mean(
            tf.cast(tf.equal(self.predicted_act, self.hate_act), tf.float32))

        self.accuracy = (self.accuracy_target + self.accuracy_act) / 2
        self.training_op = tf.train.AdamOptimizer(learning_rate=self.entity_learning_rate).minimize(self.loss)

    def run_model(self, batches, dev_batches, test_batches, weights):
        init = tf.global_variables_initializer()
        with tf.Session() as self.sess:
            self.sess.run(init)
            epoch = 1
            target_weight, act_weight = weights
            while True:
                ## Train
                epoch_loss = float(0)
                epoch += 1
                train_accuracy = 0
                for batch in batches:
                    feed_dict = {self.train_inputs: np.array([b[0] for b in batch]),
                                self.sequence_length: np.array([l for b in batch for l in b[1]]),
                                self.keep_ratio: self.entity_keep_ratio,
                                self.target_group: np.array([b[3] for b in batch]),
                                self.hate_act: np.array([b[4] for b in batch]),
                                self.target_weight: target_weight,
                                self.act_weight: act_weight
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    loss_val, _= self.sess.run([self.loss, self.training_op], feed_dict= feed_dict)
                    train_accuracy += self.accuracy.eval(feed_dict=feed_dict)
                    epoch_loss += loss_val
                ## Test
                test_accuracy = 0
                t_pred, a_pred, t_true, a_true = list(), list(), list(), list()
                for batch in dev_batches:
                    feed_dict = {self.train_inputs: np.array([b[0] for b in batch]),
                                 self.sequence_length: np.array([l for b in batch for l in b[1]]),
                                 self.keep_ratio: 1,
                                 self.target_group: np.array([b[3] for b in batch]),
                                 self.hate_act: np.array([b[4] for b in batch])
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    test_accuracy += self.accuracy.eval(feed_dict=feed_dict)
                    try:
                        target_, act_  = self.sess.run([self.predicted_target, self.predicted_act], feed_dict=feed_dict)
                        t_pred.extend(list(target_))
                        a_pred.extend(list(act_))
                        t_true.extend([b[3] for b in batch])
                        a_true.extend([b[4] for b in batch])
                    except Exception:
                        print()
                print(epoch, "Train accuracy:", train_accuracy / len(batches),
                      "Loss: ", epoch_loss / float(len(batches)),
                      "Test accuracy: ", test_accuracy / len(dev_batches),
                      "Act F1: ", f1_score(a_true, a_pred, average="macro"),
                      "Target F1: ", f1_score(t_true, t_pred, average="macro"))
                if epoch == self.epochs:
                    break
        return