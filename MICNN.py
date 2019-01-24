import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from sklearn.metrics import f1_score, precision_score, recall_score
from preprocess import *

class MICNN():
    def __init__(self, params, vocabs, my_embeddings=None):
        self.params = params
        self.vocabs = vocabs
        for key in params:
            setattr(self, key, params[key])
        if self.pretrain:
            self.my_embeddings = my_embeddings

    def build_embedding(self):
        if self.pretrain:
            embeddings = tf.Variable(tf.constant(0.0, shape=[len(self.vocabs), self.embedding_size]),
                                     trainable=False, name="W")

            embedding_placeholder = tf.placeholder(tf.float32, [len(self.vocabs), self.embedding_size])
            embedding_init = embeddings.assign(embedding_placeholder)
        else:
            embedding_placeholder = tf.get_variable("embedding",
                                                    initializer=tf.random_uniform(
                                                        [len(self.vocabs), self.embedding_size], -1, 1),
                                                    dtype=tf.float32)
        return embedding_placeholder


    def build(self):
        tf.reset_default_graph()
        # length of each sentence in the whole batch
        self.sequence_length = tf.placeholder(tf.int64, [None])
        self.article_lens = tf.placeholder(tf.int64, [None])

        # input data is in form of [batch_size, article_len, sentence_len]
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None, None])
        self.embedding_placeholder = self.build_embedding()
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)
        self.output = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        shape = tf.shape(self.embed)
        embed = tf.reshape(self.embed, [shape[0] * shape[1], shape[2], self.embedding_size])
        self.sequence_length = tf.reshape(self.sequence_length, [tf.shape(embed)[0]])

        cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
        cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_ratio)
        self.network = tf.contrib.rnn.MultiRNNCell([cell_drop] * self.num_layers)

        # the inputs are reshaped to [all sentences, sentence_len] to be passed to LSTM
        bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(self.network, self.network,
                                                                embed,dtype=tf.float32,
                                                                sequence_length=self.sequence_length)
        fw_states, bw_states = bi_states
        state = tf.concat([fw_states, bw_states], 2)
        state = tf.reshape(state, [shape[0], shape[1], 2 * self.hidden_size, 1])

        # CNN on top of Bi-LSTM to get the article features
        art_pooled_outputs = list()

        for i, art_filter_size in enumerate(self.art_filter_sizes):
            filter_shape = [art_filter_size, 2 * self.hidden_size, 1, self.art_num_filters]
            art_b = tf.Variable(tf.constant(0.1, shape=[self.art_num_filters]))
            art_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="art_W")

            conv = tf.nn.conv2d(state, art_W, strides=[1, 1, 1, 1], padding="VALID")
            relu = tf.nn.relu(tf.nn.bias_add(conv, art_b))

            pooled = tf.reduce_max(relu, axis=1, keep_dims=True)
            art_pooled_outputs.append(pooled)

        #art_num_filters_total = 2 * self.hidden_size * self.art_num_filters * len(self.art_filter_sizes)
        art_num_filters_total = self.art_num_filters * len(self.art_filter_sizes)

        # [batch_size, art_num_filters]
        self.art_pool = tf.reshape(tf.concat(art_pooled_outputs, 3), [shape[0], 1, art_num_filters_total])

        # [batch_size, sentences, num_filters]
        self.local = tf.reshape(state, [shape[0], -1, 2 * self.hidden_size])

        # context vector for each sentence
        # [batch_size, sentences, art_num_filters]
        self.context = tf.tile(self.art_pool, [1, shape[1], 1])

        # [batch_size, num_filters + art_num_filters]
        self.sentence = tf.concat([self.local, self.context], 2)

        self.drop = tf.reshape(self.sentence, [shape[0], shape[1] , 2 * self.hidden_size + art_num_filters_total])
        self.drop = tf.nn.dropout(self.drop, keep_prob=self.keep_ratio)

        self.fc_drop = fully_connected(self.drop, 1, activation_fn=tf.sigmoid)

        # High rank sentences
        high_count = tf.ceil(tf.scalar_mul(0.2, tf.to_float(shape[1])))

        #a = tf.gather(tf.reshape(self.fc_drop, [shape[0], shape[1]]), self.article_lens, axis=1)
        a = tf.reshape(self.fc_drop, [shape[0], shape[1]])
        b = tf.cast(3, tf.int32)

        #[batch_size, high_count]
        self.highests = tf.nn.top_k(a, b)

        self.logits1 = tf.reduce_mean(self.highests.values, axis=1)
        self.logits0 = tf.ones_like(self.logits1) - self.logits1
        self.logits = tf.concat([tf.expand_dims(self.logits0, 1), tf.expand_dims(self.logits1, 1)], 1)

        self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output,
                                                                       logits=self.logits)
        self.loss = tf.reduce_mean(self.xentropy)
        self.predicted_label = tf.argmax(self.logits, 1)

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predicted_label, self.output), tf.float32))
        self.training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def run_model(self, batches, dev_batches, test_batches, unlabeled_batches):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            # init.run()
            self.sess.run(init)
            epoch = 1
            while True:
                ## Train
                epoch_loss = float(0)
                epoch += 1
                train_accuracy = 0
                for batch in batches:
                    #print(np.array([b[0] for b in batch]).shape)
                    #print(np.array([b[5] for b in batch]))
                    feed_dict = {self.train_inputs: np.array([b[0] for b in batch]),
                                 self.sequence_length: np.array([l for b in batch for l in b[1]]),
                                 self.keep_prob: self.keep_ratio,
                                 self.output: np.array([b[2] for b in batch]),
                                 self.article_lens: np.array([b[5] for b in batch])
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings

                    loss_val, _, log = self.sess.run([self.loss, self.training_op, self.logits], feed_dict=feed_dict)
                    train_accuracy += self.accuracy.eval(feed_dict=feed_dict)
                    epoch_loss += loss_val
                ## Dev
                test_accuracy = 0
                hate_pred, hate_true = list(), list()
                for batch in dev_batches:
                    feed_dict = {self.train_inputs: np.array([b[0] for b in batch]),
                                 self.sequence_length: np.array([l for b in batch for l in b[1]]),
                                 self.keep_prob: 1,
                                 self.output: np.array([b[2] for b in batch]),
                                 self.article_lens: np.array([b[5] for b in batch])
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    test_accuracy += self.accuracy.eval(feed_dict=feed_dict)
                    try:
                        hate = self.predicted_label.eval(feed_dict=feed_dict)
                        hate_pred.extend(list(hate))
                        hate_true.extend([b[2] for b in batch])
                    except Exception:
                        print()
                print(sum(hate_pred))
                print(epoch, "Train accuracy:", train_accuracy / len(batches),
                      "Loss:", epoch_loss / float(len(batches)),
                      "Test accuracy:", test_accuracy / len(dev_batches),
                      "Hate F1:", f1_score(hate_true, hate_pred, average="binary"),
                      "Precision", precision_score(hate_true, hate_pred),
                      "Recall", recall_score(hate_true, hate_pred))
                if epoch == self.epochs:
                    save_path = saver.save(self.sess, "/tmp/model.ckpt")
                    break
            test_accuracy = 0
            for batch in test_batches:
                feed_dict = {self.train_inputs: np.array([b[0] for b in batch]),
                             self.sequence_length: np.array([l for b in batch for l in b[1]]),
                             self.keep_prob: 1,
                             self.output: np.array([b[2] for b in batch]),
                             self.article_lens: np.array([b[5] for b in batch])
                             }
                if self.pretrain:
                    feed_dict[self.embedding_placeholder] = self.my_embeddings
                test_accuracy += self.accuracy.eval(feed_dict=feed_dict)
                try:
                    hate = self.predicted_label.eval(feed_dict=feed_dict)
                    hate_pred.extend(list(hate))
                    hate_true.extend([b[2] for b in batch])
                except Exception:
                    print()
            print("Test report",
                  "Test accuracy:", test_accuracy / len(test_batches),
                  "Hate F1:", f1_score(hate_true, hate_pred, average="binary"),
                  "Precision", precision_score(hate_true, hate_pred),
                  "Recall", recall_score(hate_true, hate_pred))
            hate_pred = list()

            if not self.train:
                for i in range(len(unlabeled_batches) // 5000 + 1):
                    print("Gathering labels for 5000 datapoints, batch #", i)
                    sub = unlabeled_batches[i * 5000: min((i + 1) * 5000, len(unlabeled_batches))]
                    unlabeled_batches = BatchIt(sub, self.batch_size, self.vocabs)
                    for batch in unlabeled_batches:
                        print(np.array([b[0] for b in batch]).shape)
                        feed_dict = {self.train_inputs: np.array([b[0] for b in batch]),
                                     self.sequence_length: np.array([l for b in batch for l in b[1]]),
                                     self.keep_prob: 1,
                                     self.article_lens: np.array([b[2] for b in batch])
                                     }
                        if self.pretrain:
                            feed_dict[self.embedding_placeholder] = self.my_embeddings
                        hate = self.predicted_label.eval(feed_dict=feed_dict)
                        hate_pred.extend(list(hate))
                        #pickle.dump(hate_pred, open("patch_tmp.pkl", "wb"))
        return hate_pred
