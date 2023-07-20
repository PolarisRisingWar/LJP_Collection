import os
import time
import random
import optuna
import numpy as np
import keras.layers
import tensorflow as tf
from optuna.trial import Trial
from tensorflow.contrib import layers
from optuna.samplers import TPESampler
from utils.progress import WorkSplitter
from keras.layers import Dense, Lambda, multiply
from evaluation.metrics import evaluation_multitask
from utils.model import LSTMDecoder

from utils.parser import ConfigParser

'''
1. Calculate the attention vector using positive examples and negative examples respectively
2. The positive examples and negative examples is based on the law and the accu
3. Use attention vector to construct triplet loss
4. Use two decoders to further distinguish the information of the head and tail cases
'''

configFilePath = 'config/GCN.config'
config_lstm = ConfigParser(configFilePath)


class Objective:

    def __init__(self, train, valid, test, word2id_dict, tail_law_index, tail_accu_index, word_embedding, iters, seed,
                 gpu_on) -> None:
        """Initialize Class"""
        self.train = train
        self.valid = valid
        self.test = test
        self.word2id_dict = word2id_dict
        self.tail_law_index = tail_law_index
        self.tail_accu_index = tail_accu_index
        self.word_embedding = word_embedding
        self.iters = iters
        self.seed = seed
        self.gpu_on = gpu_on

    def __call__(self, trial: Trial) -> float:
        """Calculate an objective value."""

        # sample a set of hyperparameters.
        lstm_size = trial.suggest_categorical('lstm_size', [64])

        setup_seed(self.seed)

        model = CTM(lstm_size=np.int(lstm_size), word2id_dict=self.word2id_dict, tail_law_index=self.tail_law_index,
                    tail_accu_index=self.tail_accu_index, word_embedding=self.word_embedding, gpu_on=self.gpu_on)

        score, _, _ = model.train_model(self.train, self.valid, self.test, self.iters, self.seed)

        model.sess.close()
        tf.reset_default_graph()

        return score


class Tuner:
    """Class for tuning hyperparameter of Legal-AI models."""

    def __init__(self):
        """Initialize Class."""

    def tune(self, n_trials, train, valid, test, word2id_dict, tail_law_index, tail_accu_index, word_embedding, epoch,
             seed, gpu_on):
        """Hyperparameter Tuning by TPE."""
        objective = Objective(train=train, valid=valid, test=test, word2id_dict=word2id_dict,
                              tail_law_index=tail_law_index, tail_accu_index=tail_accu_index,
                              word_embedding=word_embedding, iters=epoch, seed=seed, gpu_on=gpu_on)

        search_space = {"lstm_size": [64]}
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='maximize')
        # study = optuna.create_study(sampler=TPESampler(seed=seed), direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return study.trials_dataframe(), study.best_params


class CTM(object):
    def __init__(self, lstm_size,
                 batch_size=64,
                 doc_len_fact=15,
                 sent_len_fact=100,
                 word2id_dict=None,
                 tail_law_index=None,
                 tail_accu_index=None,
                 word_embedding=None,
                 num_law=118,
                 num_accu=130,
                 num_term=11,
                 learning_rate=1e-3,
                 shuffle=True,
                 gpu_on=False,
                 **unused):
        self._lstm_size = lstm_size
        self._batch_size = batch_size
        self._doc_len_fact = doc_len_fact
        self._sent_len_fact = sent_len_fact
        self._word2id_dict = word2id_dict
        self._tail_law_index = tail_law_index
        self._tail_accu_index = tail_accu_index
        self._word_embedding = word_embedding
        self._word_dict_len = len(self._word2id_dict)
        self._num_law = num_law
        self._num_accu = num_accu
        self._num_term = num_term
        self._embed_dim = self._lstm_size * 2
        self._learning_rate = learning_rate
        self._shuffle = shuffle
        self._gpu_on = gpu_on
        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope('CTM'):
            # Placehoder
            self.fact_input = tf.placeholder(tf.int32, [self._batch_size, self._doc_len_fact, self._sent_len_fact],
                                             name='fact')
            self.law_labels = tf.placeholder(tf.int32, [self._batch_size], name='law')
            self.accu_labels = tf.placeholder(tf.int32, [self._batch_size], name='accu')
            self.time_labels = tf.placeholder(tf.int32, [self._batch_size], name='time')
            self.category_labels = tf.placeholder(tf.int32, [self._batch_size], name='category')

            self.sim_law_fact = tf.placeholder(tf.int32, [self._batch_size, self._doc_len_fact, self._sent_len_fact],
                                               name='sim_law_fact')
            self.sim_accu_fact = tf.placeholder(tf.int32, [self._batch_size, self._doc_len_fact, self._sent_len_fact],
                                                name='sim_accu_fact')
            self.dis_fact = tf.placeholder(tf.int32, [self._batch_size, self._doc_len_fact, self._sent_len_fact],
                                           name='dis_fact')

            with tf.variable_scope('Fact_Encoder'):
                # Parameter
                self.u_aw = tf.get_variable('u_aw', shape=[1, self._lstm_size * 2],
                                            initializer=layers.xavier_initializer())
                self.u_as = tf.get_variable('u_as', shape=[1, self._lstm_size * 2],
                                            initializer=layers.xavier_initializer())

                self.fully_atten_sent_1 = keras.layers.Dense(self._lstm_size * 2, name='Fully_atten_sent_1')
                self.fully_atten_doc_1 = keras.layers.Dense(self._lstm_size * 2, name='Fully_atten_doc_1')

                self.model = keras.Sequential(
                    [keras.layers.Bidirectional(keras.layers.GRU(self._lstm_size, return_sequences=True),
                                                merge_mode='concat')])
                self.model_1 = keras.Sequential(
                    [keras.layers.Bidirectional(keras.layers.GRU(self._lstm_size, return_sequences=True),
                                                merge_mode='concat')])
                # Get the representation of each case
                rep_fact_1 = self.get_rep_fact(self.fact_input)
                gold_matrix_law = tf.one_hot(self.law_labels, self._num_law, dtype=tf.float32)
                gold_matrix_accu = tf.one_hot(self.accu_labels, self._num_accu, dtype=tf.float32)
                gold_matrix_time = tf.one_hot(self.time_labels, self._num_term, dtype=tf.float32)
                gold_matrix_category = tf.one_hot(self.category_labels, 2, dtype=tf.float32)

                sim_law_rep_fact_1 = self.get_rep_fact(self.sim_law_fact)
                sim_accu_rep_fact_1 = self.get_rep_fact(self.sim_accu_fact)
                dis_law_accu_rep_fact_1 = self.get_rep_fact(self.dis_fact)

            with tf.variable_scope('Category_Prediction'):
                feature_category = Dense(self._embed_dim, use_bias=False, activation='relu')(rep_fact_1)
                category_preds = Dense(2, activation='softmax')(feature_category)

            with tf.variable_scope('Relational_Attention'):
                law_embed_1 = Dense(self._embed_dim, bias=False, name='law_embed_1')
                law_embed_2 = Dense(self._embed_dim, name='law_embed_2')
                law_embed_3 = Dense(self._embed_dim, bias=False, name='law_embed_3')

                sim_law_attention = law_embed_3(
                    tf.sigmoid(law_embed_1(rep_fact_1) + law_embed_2(sim_law_rep_fact_1)))
                dis_law_attention = law_embed_3(
                    tf.sigmoid(law_embed_1(rep_fact_1) + law_embed_2(dis_law_accu_rep_fact_1)))
                self_law_attention = law_embed_3(tf.sigmoid(law_embed_1(rep_fact_1) + law_embed_2(rep_fact_1)))

                accu_embed_1 = Dense(self._embed_dim, bias=False, name='accu_embed_1')
                accu_embed_2 = Dense(self._embed_dim, name='accu_embed_2')
                accu_embed_3 = Dense(self._embed_dim, bias=False, name='accu_embed_3')

                sim_accu_attention = accu_embed_3(
                    tf.sigmoid(accu_embed_1(rep_fact_1) + accu_embed_2(sim_accu_rep_fact_1)))
                dis_accu_attention = accu_embed_3(
                    tf.sigmoid(accu_embed_1(rep_fact_1) + accu_embed_2(dis_law_accu_rep_fact_1)))
                self_accu_attention = accu_embed_3(tf.sigmoid(accu_embed_1(rep_fact_1) + accu_embed_2(rep_fact_1)))

            with tf.variable_scope('TopJudge'):
                feature = rep_fact_1
                # head decoder
                head_decoder = LSTMDecoder(config_lstm, self._num_law, self._num_accu)
                output_task1_head, output_task2_head, output_task3_head = head_decoder(feature)
                # tail decoder
                tail_decoder = LSTMDecoder(config_lstm, self._num_law, self._num_accu)
                output_task1_tail, output_task2_tail, output_task3_tail = tail_decoder(feature)

                output_task6 = category_preds

            with tf.variable_scope('Loss'):
                # head loss
                law_prob = tf.nn.softmax(output_task1_head, -1)
                accu_prob = tf.nn.softmax(output_task2_head, -1)
                time_prob = tf.nn.softmax(output_task3_head, -1)

                self.law_predictions_head = tf.argmax(law_prob, 1)
                self.accu_predictions_head = tf.argmax(accu_prob, 1)
                self.time_predictions_head = tf.argmax(time_prob, 1)

                loss_1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.get_masked_tensor(output_task1_head),
                                                                 labels=self.get_masked_tensor(gold_matrix_law))
                loss_2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.get_masked_tensor(output_task2_head),
                                                                 labels=self.get_masked_tensor(gold_matrix_accu))
                loss_3 = tf.nn.softmax_cross_entropy_with_logits(logits=self.get_masked_tensor(output_task3_head),
                                                                 labels=self.get_masked_tensor(gold_matrix_time))

                self.law_loss_head = tf.reduce_sum(loss_1)
                self.accu_loss_head = tf.reduce_sum(loss_2)
                self.time_loss_head = tf.reduce_sum(loss_3)

                # tail loss
                law_prob = tf.nn.softmax(output_task1_tail, -1)
                accu_prob = tf.nn.softmax(output_task2_tail, -1)
                time_prob = tf.nn.softmax(output_task3_tail, -1)

                self.law_predictions_tail = tf.argmax(law_prob, 1)
                self.accu_predictions_tail = tf.argmax(accu_prob, 1)
                self.time_predictions_tail = tf.argmax(time_prob, 1)

                loss_1 = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.get_masked_tensor(output_task1_tail, 'tail'),
                    labels=self.get_masked_tensor(gold_matrix_law, 'tail'))
                loss_2 = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.get_masked_tensor(output_task2_tail, 'tail'),
                    labels=self.get_masked_tensor(gold_matrix_accu, 'tail'))
                loss_3 = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.get_masked_tensor(output_task3_tail, 'tail'),
                    labels=self.get_masked_tensor(gold_matrix_time, 'tail'))

                self.law_loss_tail = tf.reduce_sum(loss_1)
                self.accu_loss_tail = tf.reduce_sum(loss_2)
                self.time_loss_tail = tf.reduce_sum(loss_3)

                # category loss
                self.category_predictions = tf.argmax(output_task6, 1)
                self.category_loss = tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits(labels=gold_matrix_category, logits=output_task6))

                # triplet loss
                d_pos = tf.reduce_sum(tf.square(self_law_attention - sim_law_attention), 1)
                d_neg = tf.reduce_sum(tf.square(self_law_attention - dis_law_attention), 1)
                self.triplet_loss1 = tf.reduce_mean(tf.maximum(0.0, 0.5 + d_pos - d_neg))

                d_pos = tf.reduce_sum(tf.square(self_accu_attention - sim_accu_attention), 1)
                d_neg = tf.reduce_sum(tf.square(self_accu_attention - dis_accu_attention), 1)
                self.triplet_loss2 = tf.reduce_mean(tf.maximum(0.0, 0.3 + d_pos - d_neg))

                loss = self.law_loss_head + self.accu_loss_head + self.time_loss_head + self.law_loss_tail + self.accu_loss_tail + self.time_loss_tail + self.triplet_loss1 + self.triplet_loss2 + self.category_loss
                # loss = self.law_loss + self.accu_loss + self.time_loss + self.triplet_loss1 + self.triplet_loss2
                tf.add_to_collection('losses_1', tf.contrib.layers.l2_regularizer(0.0001)(self.law_loss_head))
                tf.add_to_collection('losses_2', tf.contrib.layers.l2_regularizer(0.0001)(self.accu_loss_head))
                tf.add_to_collection('losses_3', tf.contrib.layers.l2_regularizer(0.0001)(self.time_loss_head))
                tf.add_to_collection('losses_4', tf.contrib.layers.l2_regularizer(0.0001)(self.law_loss_tail))
                tf.add_to_collection('losses_5', tf.contrib.layers.l2_regularizer(0.0001)(self.accu_loss_tail))
                tf.add_to_collection('losses_6', tf.contrib.layers.l2_regularizer(0.0001)(self.time_loss_tail))
                tf.add_to_collection('losses_7', tf.contrib.layers.l2_regularizer(0.0001)(self.triplet_loss1))
                tf.add_to_collection('losses_8', tf.contrib.layers.l2_regularizer(0.0001)(self.triplet_loss2))
                tf.add_to_collection('losses_9', tf.contrib.layers.l2_regularizer(0.0001)(self.category_loss))

                self.loss_total = loss + tf.add_n(tf.get_collection('losses_1')) + tf.add_n(
                    tf.get_collection('losses_2')) + tf.add_n(tf.get_collection('losses_3')) + tf.add_n(
                    tf.get_collection('losses_4')) + tf.add_n(tf.get_collection('losses_5')) + tf.add_n(
                    tf.get_collection('losses_6')) + tf.add_n(tf.get_collection('losses_7')) + tf.add_n(
                    tf.get_collection('losses_8')) + tf.add_n(tf.get_collection('losses_9'))

                global_step = tf.Variable(0, trainable=False)
                optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate, beta1=.9, beta2=.999,
                                                   epsilon=1e-7)
                self.train_op = optimizer.minimize(self.loss_total, global_step=global_step)

                self.correct_law_head = tf.nn.in_top_k(output_task1_head, self.law_labels, 1)
                self.correct_accu_head = tf.nn.in_top_k(output_task2_head, self.accu_labels, 1)
                self.correct_time_head = tf.nn.in_top_k(output_task3_head, self.time_labels, 1)
                self.correct_law_tail = tf.nn.in_top_k(output_task1_tail, self.law_labels, 1)
                self.correct_accu_tail = tf.nn.in_top_k(output_task2_tail, self.accu_labels, 1)
                self.correct_time_tail = tf.nn.in_top_k(output_task3_tail, self.time_labels, 1)
                self.correct_category = tf.nn.in_top_k(output_task6, self.category_labels, 1)

            if self._gpu_on:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
            else:
                config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def get_rep_fact(self, fact_input):
        fact_mask = tf.cast(tf.cast(fact_input - self._word2id_dict['BLANK'], tf.bool), tf.float32)
        fact_sent_len = tf.reduce_sum(fact_mask, -1)
        fact_doc_mask = tf.cast(tf.cast(fact_sent_len, tf.bool), tf.float32)

        fact_description = tf.nn.embedding_lookup(self._word_embedding, fact_input)

        rep_fact = self.run_model(fact_description, fact_mask, self.model, self._lstm_size)
        rep_fact_, _ = self.atten_encoder_mask(self.u_aw, rep_fact, self.fully_atten_sent_1, fact_mask, K_ori=True)

        rep_fact_1 = self.run_model(rep_fact_, fact_doc_mask, self.model_1, self._lstm_size)
        rep_fact_1, _ = self.atten_encoder_mask(self.u_as, rep_fact_1, self.fully_atten_doc_1, fact_doc_mask, K_ori=True)
        return rep_fact_1

    @staticmethod
    def normalize(x, epsilon=1e-9):
        x /= (tf.reduce_sum(x, axis=-1, keepdims=True) + epsilon)
        return x

    @staticmethod
    def run_model(rep, mask, model, lstm_size):
        input_shape = rep.get_shape().as_list()
        mask = tf.expand_dims(mask, -1)
        mask_shape = mask.get_shape().as_list()
        rep = tf.reshape(rep, [int(np.prod(input_shape[:-2]))] + input_shape[-2:])
        mask = tf.reshape(mask, [int(np.prod(mask_shape[:-2]))] + mask_shape[-2:])
        mask = tf.squeeze(mask)
        out = model(rep, mask=mask)
        _rep = tf.reshape(out, input_shape[:-1] + [lstm_size * 2])
        return _rep

    @staticmethod
    def get_safe_shift(logits, mask):
        """
        :param logits: A tf.Tensor of shape [B, TQ, TK] of dtype tf.float32
        :param mask: A tf.Tensor of shape [B, TQ, TK] of dtype tf.float32
        where TQ, TK are the maximum lengths of the queries resp. the keys in the batch
        """

        # Determine minimum
        K_shape = logits.get_shape().as_list()
        mask_shape = mask.get_shape().as_list()
        if mask_shape != K_shape:
            mask = tf.tile(mask, [1] + [K_shape[1] // mask_shape[1]] + [1] * (len(K_shape) - 2))

        logits_min = tf.reduce_min(logits, axis=-1, keepdims=True)  # [B, TQ, 1]
        logits_min = tf.tile(logits_min, multiples=[1] * (len(K_shape) - 1) + [K_shape[-1]])  # [B, TQ, TK]

        logits = tf.where(condition=mask > .5, x=logits, y=logits_min)

        # Determine maximum
        logits_max = tf.reduce_max(logits, axis=-1, keepdims=True, name="logits_max")  # [B, TQ, 1]
        logits_shifted = tf.subtract(logits, logits_max, name="logits_shifted")  # [B, TQ, TK]

        return logits_shifted

    def padding_aware_softmax(self, logits, key_mask, query_mask=None, epsilon=1e-9):

        logits_shifted = self.get_safe_shift(logits, key_mask)

        # Apply exponential
        weights_unscaled = tf.exp(logits_shifted)

        # Apply mask
        weights_unscaled = tf.multiply(key_mask, weights_unscaled)  # [B, TQ, TK]

        # Derive total mass
        weights_total_mass = tf.reduce_sum(weights_unscaled, axis=-1, keepdims=True)  # [B, TQ, 1]

        # Avoid division by zero
        if query_mask:
            weights_total_mass = tf.where(condition=tf.equal(query_mask, 1),
                                          x=weights_total_mass,
                                          y=tf.ones_like(weights_total_mass))

        # Normalize weights
        weights = tf.divide(weights_unscaled, weights_total_mass + epsilon)  # [B, TQ, TK]

        return weights

    def atten_encoder_mask(self, Q, K, fc_layer=None, mask=None, weights_regularizer=None, K_ori=False, div_norm=True):
        '''
        :param Q: [..., seq_len_q, F] : the attention vector u
        :param K: [..., seq_len_k, F]
        :param mask:
        :return: a tensor whose size is [..., F]
        '''
        V = K
        K_shape = K.get_shape().as_list()  # size[x, y, z]
        if fc_layer is not None:
            K = fc_layer(K)
        else:
            K = layers.fully_connected(K, K_shape[-1], activation_fn=tf.nn.tanh,
                                       weights_regularizer=weights_regularizer)  # size: [x, y, z]

        if not K_ori:
            V = K

        scores = tf.reduce_sum(K * Q, -1)  # size: [x, y]
        if div_norm:
            scores = scores / tf.sqrt(tf.cast(K_shape[-1], tf.float32))  # size: [x, y]

        if mask is not None:
            scores = self.padding_aware_softmax(scores, mask)
        else:
            scores = tf.nn.softmax(scores, -1)
        return tf.reduce_sum(tf.expand_dims(scores, -1) * V, -2), scores  # size: [x, z]

    def gen_dict(self, inputs_, law_labels_input_, accu_labels_input_, time_labels_input_, category_labels_input_,
                 similar_law_imput_, similar_accu_imput_, dissimilar_input_):
        if similar_law_imput_ is None:
            feed_dict_ = {self.fact_input: inputs_, self.law_labels: law_labels_input_,
                          self.accu_labels: accu_labels_input_, self.time_labels: time_labels_input_,
                          self.category_labels: category_labels_input_}

        else:
            feed_dict_ = {self.fact_input: inputs_, self.law_labels: law_labels_input_,
                          self.accu_labels: accu_labels_input_, self.time_labels: time_labels_input_,
                          self.category_labels: category_labels_input_, self.sim_law_fact: similar_law_imput_,
                          self.sim_accu_fact: similar_accu_imput_, self.dis_fact: dissimilar_input_}

        return feed_dict_

    def get_labels_group(self, law_labels, accu_labels):
        _law_labels, _accu_labels = np.array(law_labels), np.array(accu_labels)
        _law_labels, _accu_labels = np.reshape(_law_labels, (len(_law_labels), 1)), np.reshape(_accu_labels,
                                                                                               (len(_accu_labels), 1))

        _law_labels = np.hstack((_law_labels, np.reshape(np.arange(_law_labels.shape[0]), (len(_law_labels), 1))))

        _law_labels = _law_labels[_law_labels[:, 0].argsort()]
        _law_labels_group = np.split(_law_labels[:, 1], np.unique(_law_labels[:, 0], return_index=True)[1][1:])
        _law_labels_dict = dict(enumerate(np.unique(_law_labels[:, 0])))
        _law_labels_dict = {v: k for k, v in _law_labels_dict.items()}

        _accu_labels = np.hstack((_accu_labels, np.reshape(np.arange(_accu_labels.shape[0]), (len(_accu_labels), 1))))
        _accu_labels = _accu_labels[_accu_labels[:, 0].argsort()]
        _accu_labels_group = np.split(_accu_labels[:, 1], np.unique(_accu_labels[:, 0], return_index=True)[1][1:])
        _accu_labels_dict = dict(enumerate(np.unique(_accu_labels[:, 0])))
        _accu_labels_dict = {v: k for k, v in _accu_labels_dict.items()}

        _dis_group = list()
        for k, _ in _accu_labels_dict.items():
            index = np.where((_accu_labels[:, 0] != k) & (
                np.isin(_law_labels[:, 0], self._tail_law_index, invert=True)) & (
                                 np.isin(_accu_labels[:, 0], self._tail_accu_index, invert=True)))[0]
            _dis_group.append(index)
        return _law_labels_group, _accu_labels_group, _dis_group, _law_labels_dict, _accu_labels_dict

    @staticmethod
    def get_comparable_cases(inputs, law_labels_input, accu_labels_input, fact, law_labels_dict, accu_labels_dict,
                             mode1_group, mode2_group, mode3_group, mode=0, seed=0):
        setup_seed(seed)
        comparable_cases = []

        for i in range(len(inputs)):
            if mode == 0:
                # mode 0: similar_law_cases
                index = mode1_group[law_labels_dict[law_labels_input[i]]]
                sample_index = np.random.choice(index, 1)[0]
                comparable_cases.append(fact[sample_index])
            elif mode == 1:
                # mode 1: similar_accu_cases
                index = mode2_group[accu_labels_dict[accu_labels_input[i]]]
                sample_index = np.random.choice(index, 1)[0]
                comparable_cases.append(fact[sample_index])
            elif mode == 2:
                # mode 2: dissimilar_law_cases and dissimilar_accu_cases
                index = mode3_group[accu_labels_dict[accu_labels_input[i]]]
                sample_index = np.random.choice(index, 1)[0]
                comparable_cases.append(fact[sample_index])
            # print('Step: %d/%d' % (i, len(inputs)))
        return comparable_cases

    def get_category_label(self, law_labels_input, accu_labels_input):
        length = len(law_labels_input)
        category_label = np.ones(length)
        count_tail = 0
        for i in range(length):
            if law_labels_input[i] in self._tail_law_index or accu_labels_input[i] in self._tail_accu_index:
                category_label[i] = 0
                count_tail += 1
        print('The total number of long tail cases is: ', count_tail)
        return category_label.tolist()

    def get_masked_tensor(self, ori_tensor, decoder_type='head'):
        if decoder_type == 'head':
            label_mask = self.category_labels
        else:
            label_mask = 1 - self.category_labels
        length = tf.shape(ori_tensor)[1]
        label_mask = length*label_mask
        tensor_mask = tf.sequence_mask(lengths=label_mask, maxlen=length, dtype=tf.float32)
        res_tensor = tf.multiply(ori_tensor, tensor_mask)

        return res_tensor

    def train_model(self, f_train, f_valid, f_test, max_epoch=20, seed=0):
        setup_seed(seed)

        task = ['law', 'accu', 'time']

        total_loss, law_loss_head, accu_loss_head, term_loss_head, triplet_loss1, triplet_loss2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        law_loss_tail, accu_loss_tail, term_loss_tail, category_loss = 0.0, 0.0, 0.0, 0.0
        start_time = time.time()

        train_step = int(len(f_train['fact_list']) / self._batch_size) + 1
        lose_num_train = train_step * self._batch_size - len(f_train['fact_list'])

        valid_step = int(len(f_valid['fact_list']) / self._batch_size) + 1
        lose_num_valid = valid_step * self._batch_size - len(f_valid['fact_list'])

        test_step = int(len(f_test['fact_list']) / self._batch_size) + 1
        lose_num_test = test_step * self._batch_size - len(f_test['fact_list'])

        fact_train = f_train['fact_list']
        law_labels_train = f_train['law_label_lists']
        accu_label_train = f_train['accu_label_lists']
        term_train = f_train['term_lists']
        category_train = self.get_category_label(law_labels_train, accu_label_train)

        best_valid_result, best_test_accuracy, best_test_metric, result_early_stop = 0, 0, 0, 0
        temp_y, temp_prediction = [], []

        if self._shuffle:
            index = [i for i in range(len(f_train['term_lists']))]
            random.shuffle(index)
            fact_train = [fact_train[i] for i in index]
            law_labels_train = [law_labels_train[i] for i in index]
            accu_label_train = [accu_label_train[i] for i in index]
            term_train = [term_train[i] for i in index]
            category_train = [category_train[i] for i in index]

            mode1_group, mode2_group, mode3_group, law_labels_dict, accu_labels_dict = self.get_labels_group(
                law_labels_train, accu_label_train)
            for epoch in range(max_epoch):
                # Sample the introduced cases
                similar_law_cases = self.get_comparable_cases(fact_train, law_labels_train, accu_label_train,
                                                              fact_train, law_labels_dict, accu_labels_dict,
                                                              mode1_group, mode2_group, mode3_group, 0,
                                                              seed)
                similar_accu_cases = self.get_comparable_cases(fact_train, law_labels_train, accu_label_train,
                                                               fact_train, law_labels_dict, accu_labels_dict,
                                                               mode1_group, mode2_group, mode3_group,
                                                               1,
                                                               seed)
                dissimilar_cases = self.get_comparable_cases(fact_train, law_labels_train, accu_label_train,
                                                             fact_train, law_labels_dict, accu_labels_dict,
                                                             mode1_group, mode2_group, mode3_group,
                                                             2, seed)

                for i in range(train_step):
                    if i == train_step - 1:
                        inputs = np.array(fact_train[i * self._batch_size:] + fact_train[:lose_num_train],
                                          dtype='int32')
                        law_labels_input = np.array(
                            law_labels_train[i * self._batch_size:] + law_labels_train[:lose_num_train],
                            dtype='int32')
                        accu_labels_input = np.array(
                            accu_label_train[i * self._batch_size:] + accu_label_train[:lose_num_train],
                            dtype='int32')
                        time_labels_input = np.array(term_train[i * self._batch_size:] + term_train[:lose_num_train],
                                                     dtype='int32')
                        category_labels_input = np.array(category_train[i * self._batch_size:] + category_train[:lose_num_train],
                                                         dtype='int32')

                        similar_law_input = np.array(
                            similar_law_cases[i * self._batch_size:] + similar_law_cases[:lose_num_train],
                            dtype='int32')
                        similar_accu_input = np.array(
                            similar_accu_cases[i * self._batch_size:] + similar_accu_cases[:lose_num_train],
                            dtype='int32')
                        dissimilar_input = np.array(
                            dissimilar_cases[i * self._batch_size:] + dissimilar_cases[:lose_num_train],
                            dtype='int32')
                    else:
                        inputs = np.array(fact_train[i * self._batch_size: (i + 1) * self._batch_size],
                                          dtype='int32')
                        law_labels_input = np.array(law_labels_train[i * self._batch_size: (i + 1) * self._batch_size],
                                                    dtype='int32')
                        accu_labels_input = np.array(accu_label_train[i * self._batch_size: (i + 1) * self._batch_size],
                                                     dtype='int32')
                        time_labels_input = np.array(term_train[i * self._batch_size: (i + 1) * self._batch_size],
                                                     dtype='int32')
                        category_labels_input = np.array(category_train[i * self._batch_size: (i + 1) * self._batch_size],
                                                         dtype='int32')

                        similar_law_input = np.array(
                            similar_law_cases[i * self._batch_size: (i + 1) * self._batch_size],
                            dtype='int32')
                        similar_accu_input = np.array(
                            similar_accu_cases[i * self._batch_size: (i + 1) * self._batch_size],
                            dtype='int32')
                        dissimilar_input = np.array(
                            dissimilar_cases[i * self._batch_size: (i + 1) * self._batch_size],
                            dtype='int32')

                    # print(inputs.shape)
                    feed_dict = self.gen_dict(inputs, law_labels_input, accu_labels_input, time_labels_input, category_labels_input,
                                              similar_law_input, similar_accu_input, dissimilar_input)
                    loss_value, law_loss_value_head, accu_loss_value_head, term_loss_value_head, \
                    law_loss_value_tail, accu_loss_value_tail, term_loss_value_tail, \
                    triplet_loss1_value, triplet_loss2_value, category_loss_value, _ = self.sess.run(
                        [self.loss_total,
                         self.law_loss_head,
                         self.accu_loss_head,
                         self.time_loss_head,
                         self.law_loss_tail,
                         self.accu_loss_tail,
                         self.time_loss_tail,
                         self.triplet_loss1,
                         self.triplet_loss2,
                         self.category_loss,
                         self.train_op], feed_dict=feed_dict)
                    total_loss += loss_value
                    law_loss_head += law_loss_value_head
                    accu_loss_head += accu_loss_value_head
                    term_loss_head += term_loss_value_head
                    law_loss_tail += law_loss_value_tail
                    accu_loss_tail += accu_loss_value_tail
                    term_loss_tail += term_loss_value_tail
                    triplet_loss1 += triplet_loss1_value
                    triplet_loss2 += triplet_loss2_value
                    category_loss += category_loss_value
                    print('Step %d: loss = %.2f, law_head = %.2f, accu_head = %.2f, term_head = %.2f, law_tail = %.2f, accu_tail = %.2f, term_tail = %.2f, triplet1 = %.2f, triplet2 = %.2f, category = %.2f' % (
                        i, loss_value, law_loss_value_head, accu_loss_value_head, term_loss_value_head, law_loss_value_tail, accu_loss_value_tail, term_loss_value_tail, triplet_loss1_value, triplet_loss2_value, category_loss_value))
                    if (i + 1) == train_step:
                        duration = time.time() - start_time
                        start_time = time.time()
                        print(
                            'Step %d: loss = %.2f, law_head = %.2f, accu_head = %.2f, term_head = %.2f, law_tail = %.2f, accu_tail = %.2f, term_tail = %.2f, triplet1 = %.2f, triplet2 = %.2f, category = %.2f (%.3f sec)' % (
                                i, total_loss, law_loss_head, accu_loss_head, term_loss_head, law_loss_tail, accu_loss_tail, term_loss_tail, triplet_loss1, triplet_loss2, category_loss, duration))
                        total_loss, law_loss_head, accu_loss_head, term_loss_head, law_loss_tail, accu_loss_tail, term_loss_tail, triplet_loss1, triplet_loss2, category_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

                # ----------the following is valid prediction---------- #
                predic_law, predic_accu, predic_time, predic_category = [], [], [], []
                y_law, y_accu, y_time, y_category = [], [], [], []
                time_correct = []
                total_tags = 0.0
                correct_tags_law = 0
                correct_tags_accu = 0
                correct_tags_time = 0
                correct_tags_category = 0
                category_valid = self.get_category_label(f_valid['law_label_lists'], f_valid['accu_label_lists'])

                for i in range(valid_step):
                    if i == valid_step - 1:
                        inputs = np.array(
                            f_valid['fact_list'][i * self._batch_size:] + f_valid['fact_list'][:lose_num_valid],
                            dtype='int32')
                        law_labels_input = np.array(
                            f_valid['law_label_lists'][i * self._batch_size:] + f_valid['law_label_lists'][
                                                                                :lose_num_valid],
                            dtype='int32')
                        accu_labels_input = np.array(
                            f_valid['accu_label_lists'][i * self._batch_size:] + f_valid['accu_label_lists'][
                                                                                 :lose_num_valid],
                            dtype='int32')
                        time_labels_input = np.array(
                            f_valid['term_lists'][i * self._batch_size:] + f_valid['term_lists'][:lose_num_valid],
                            dtype='int32')
                        category_labels_input = np.array(
                            category_valid[i * self._batch_size:] + category_valid[:lose_num_valid],
                            dtype='int32')
                    else:
                        inputs = np.array(f_valid['fact_list'][i * self._batch_size: (i + 1) * self._batch_size],
                                          dtype='int32')
                        law_labels_input = np.array(
                            f_valid['law_label_lists'][i * self._batch_size: (i + 1) * self._batch_size],
                            dtype='int32')
                        accu_labels_input = np.array(
                            f_valid['accu_label_lists'][i * self._batch_size: (i + 1) * self._batch_size],
                            dtype='int32')
                        time_labels_input = np.array(
                            f_valid['term_lists'][i * self._batch_size: (i + 1) * self._batch_size],
                            dtype='int32')
                        category_labels_input = np.array(
                            category_valid[i * self._batch_size: (i + 1) * self._batch_size],
                            dtype='int32')
                    feed_dict_valid = self.gen_dict(inputs, law_labels_input, accu_labels_input, time_labels_input, category_labels_input,
                                                    None, None, None)
                    num_y = self._batch_size
                    if i + 1 == valid_step:
                        num_y = self._batch_size - lose_num_valid

                    total_tags += num_y

                    # aggregate results
                    correct_law_head_, correct_accu_head_, correct_time_head_, correct_law_tail_, correct_accu_tail_, correct_time_tail_, correct_category_, \
                    predic_law_head_, predic_accu_head_, predic_time_head_, predic_law_tail_, predic_accu_tail_, predic_time_tail_, predic_category_, \
                    y_law_, y_accu_, y_time_, y_category_ = self.sess.run(
                        (self.correct_law_head, self.correct_accu_head, self.correct_time_head, self.correct_law_tail,
                         self.correct_accu_tail, self.correct_time_tail, self.correct_category,
                         self.law_predictions_head, self.accu_predictions_head, self.time_predictions_head,
                         self.law_predictions_tail, self.accu_predictions_tail, self.time_predictions_tail,
                         self.category_predictions,
                         self.law_labels, self.accu_labels, self.time_labels, self.category_labels),
                        feed_dict=feed_dict_valid)

                    predic_law_ = predic_law_head_ * y_category_ + predic_law_tail_ * (1 - y_category_)
                    predic_accu_ = predic_accu_head_ * y_category_ + predic_accu_tail_ * (1 - y_category_)
                    predic_time_ = predic_time_head_ * y_category_ + predic_time_tail_ * (1 - y_category_)

                    correct_law_ = correct_law_head_ * y_category_ + correct_law_tail_ * (1 - y_category_)
                    correct_accu_ = correct_accu_head_ * y_category_ + correct_accu_tail_ * (1 - y_category_)
                    correct_time_ = correct_time_head_ * y_category_ + correct_time_tail_ * (1 - y_category_)

                    predic_law += list(predic_law_[:num_y])
                    predic_accu += list(predic_accu_[:num_y])
                    predic_time += list(predic_time_[:num_y])
                    predic_category += list(predic_category_[:num_y])

                    y_law += list(y_law_[:num_y])
                    y_accu += list(y_accu_[:num_y])
                    y_time += list(y_time_[:num_y])
                    y_category += list(y_category_[:num_y])
                    time_correct += list(correct_time_head_[:num_y])

                    correct_tags_law += np.sum(np.cast[np.int32](correct_law_[:num_y]))
                    correct_tags_accu += np.sum(np.cast[np.int32](correct_accu_[:num_y]))
                    correct_tags_time += np.sum(np.cast[np.int32](correct_time_[:num_y]))
                    correct_tags_category += np.sum(np.cast[np.int32](correct_category_[:num_y]))

                prediction = [predic_law, predic_accu, predic_time, predic_category]
                y = [y_law, y_accu, y_time, y_category]
                correct_tags = [correct_tags_law, correct_tags_accu, correct_tags_time, correct_tags_category]
                print(len(time_correct))
                accuracy, metric = evaluation_multitask(y, prediction, 4, correct_tags, total_tags)
                print('Now_epoch is: {}'.format(epoch))
                for i in range(len(task)):
                    print('Accuracy for {} prediction is: '.format(task[i]), accuracy[i])
                    print('Macro-precision for {} prediction is: '.format(task[i]), metric[i][3])
                    print('Macro-recall for {} prediction is: '.format(task[i]), metric[i][1])
                    print('Macro-F1 for {} prediction is: '.format(task[i]), metric[i][5])
                    print('Other metrics for {} prediction is: '.format(task[i]), metric[i])

                print('\n')

                # ----------the following is test prediction---------- #
                predic_law, predic_accu, predic_time, predic_category = [], [], [], []
                y_law, y_accu, y_time, y_category = [], [], [], []
                time_correct = []
                total_tags = 0.0
                correct_tags_law = 0
                correct_tags_accu = 0
                correct_tags_time = 0
                correct_tags_category = 0
                category_test = self.get_category_label(f_test['law_label_lists'], f_test['accu_label_lists'])

                for i in range(test_step):
                    if i == test_step - 1:
                        inputs = np.array(
                            f_test['fact_list'][i * self._batch_size:] + f_test['fact_list'][:lose_num_test],
                            dtype='int32')
                        law_labels_input = np.array(
                            f_test['law_label_lists'][i * self._batch_size:] + f_test['law_label_lists'][
                                                                               :lose_num_test],
                            dtype='int32')
                        accu_labels_input = np.array(
                            f_test['accu_label_lists'][i * self._batch_size:] + f_test['accu_label_lists'][
                                                                                :lose_num_test],
                            dtype='int32')
                        time_labels_input = np.array(
                            f_test['term_lists'][i * self._batch_size:] + f_test['term_lists'][:lose_num_test],
                            dtype='int32')
                        category_labels_input = np.array(
                            category_test[i * self._batch_size:] + category_test[:lose_num_test],
                            dtype='int32')
                    else:
                        inputs = np.array(f_test['fact_list'][i * self._batch_size: (i + 1) * self._batch_size],
                                          dtype='int32')
                        law_labels_input = np.array(
                            f_test['law_label_lists'][i * self._batch_size: (i + 1) * self._batch_size],
                            dtype='int32')
                        accu_labels_input = np.array(
                            f_test['accu_label_lists'][i * self._batch_size: (i + 1) * self._batch_size],
                            dtype='int32')
                        time_labels_input = np.array(
                            f_test['term_lists'][i * self._batch_size: (i + 1) * self._batch_size],
                            dtype='int32')
                        category_labels_input = np.array(
                            category_test[i * self._batch_size: (i + 1) * self._batch_size],
                            dtype='int32')

                    feed_dict_test = self.gen_dict(inputs, law_labels_input, accu_labels_input, time_labels_input, category_labels_input,
                                                   None, None, None)
                    num_y = self._batch_size
                    if i + 1 == test_step:
                        num_y = self._batch_size - lose_num_test

                    total_tags += num_y
                    # aggregate results
                    correct_law_head_, correct_accu_head_, correct_time_head_, correct_law_tail_, correct_accu_tail_, correct_time_tail_, correct_category_, \
                    predic_law_head_, predic_accu_head_, predic_time_head_, predic_law_tail_, predic_accu_tail_, predic_time_tail_, predic_category_, \
                    y_law_, y_accu_, y_time_, y_category_ = self.sess.run(
                        (self.correct_law_head, self.correct_accu_head, self.correct_time_head, self.correct_law_tail,
                         self.correct_accu_tail, self.correct_time_tail, self.correct_category,
                         self.law_predictions_head, self.accu_predictions_head, self.time_predictions_head,
                         self.law_predictions_tail, self.accu_predictions_tail, self.time_predictions_tail,
                         self.category_predictions,
                         self.law_labels, self.accu_labels, self.time_labels, self.category_labels),
                        feed_dict=feed_dict_test)

                    predic_law_ = predic_law_head_ * y_category_ + predic_law_tail_ * (1 - y_category_)
                    predic_accu_ = predic_accu_head_ * y_category_ + predic_accu_tail_ * (1 - y_category_)
                    predic_time_ = predic_time_head_ * y_category_ + predic_time_tail_ * (1 - y_category_)

                    correct_law_ = correct_law_head_ * y_category_ + correct_law_tail_ * (1 - y_category_)
                    correct_accu_ = correct_accu_head_ * y_category_ + correct_accu_tail_ * (1 - y_category_)
                    correct_time_ = correct_time_head_ * y_category_ + correct_time_tail_ * (1 - y_category_)

                    predic_law += list(predic_law_[:num_y])
                    predic_accu += list(predic_accu_[:num_y])
                    predic_time += list(predic_time_[:num_y])
                    predic_category += list(predic_category_[:num_y])

                    y_law += list(y_law_[:num_y])
                    y_accu += list(y_accu_[:num_y])
                    y_time += list(y_time_[:num_y])
                    y_category += list(y_category_[:num_y])
                    time_correct += list(correct_time_head_[:num_y])

                    correct_tags_law += np.sum(np.cast[np.int32](correct_law_[:num_y]))
                    correct_tags_accu += np.sum(np.cast[np.int32](correct_accu_[:num_y]))
                    correct_tags_time += np.sum(np.cast[np.int32](correct_time_[:num_y]))
                    correct_tags_category += np.sum(np.cast[np.int32](correct_category_[:num_y]))

                prediction = [predic_law, predic_accu, predic_time, predic_category]
                y = [y_law, y_accu, y_time, y_category]
                correct_tags = [correct_tags_law, correct_tags_accu, correct_tags_time, correct_tags_category]

                _accuracy, _metric = evaluation_multitask(y, prediction, 4, correct_tags, total_tags)

                print('Now_testing')
                for i in range(len(task)):
                    print('Accuracy for {} prediction is: '.format(task[i]), _accuracy[i])
                    print('Macro-precision for {} prediction is: '.format(task[i]), _metric[i][3])
                    print('Macro-recall for {} prediction is: '.format(task[i]), _metric[i][1])
                    print('Macro-F1 for {} prediction is: '.format(task[i]), _metric[i][5])
                    print('Other metrics for {} prediction is: '.format(task[i]), _metric[i])

                print('\n')

                if accuracy[0] > best_valid_result:
                    best_valid_result = accuracy[0]
                    best_test_accuracy = _accuracy
                    best_test_metric = _metric
                    temp_y, temp_prediction = y, prediction
                    result_early_stop = 0
                else:
                    result_early_stop += 1
                    if result_early_stop > 5:
                        break
        # Save tags and predictions of each task for analyzing the source of errors
        if not os.path.exists('analysis/'):
            os.makedirs('analysis/')

        np.savetxt("analysis/CTM_y_" + str(seed) + ".csv", np.array(temp_y).astype(int), fmt='%i', delimiter=",")
        np.savetxt("analysis/CTM_pred_" + str(seed) + ".csv", np.array(temp_prediction).astype(int), fmt='%i',
                   delimiter=",")

        return best_valid_result, best_test_accuracy, best_test_metric


def get_tail_index(train, threshold=0.4):
    law_labels_train = train['law_label_lists']
    accu_labels_train = train['accu_label_lists']

    num_law = 118
    num_accu = 130
    # Calculating the frequency of the law
    law_idx = np.arange(num_law)
    law_frequency = np.zeros_like(law_idx)
    for i in range(len(law_labels_train)):
        law_frequency[law_labels_train[i]] += 1
    law_frequency_mat = np.hstack((np.expand_dims(law_idx, 1), np.expand_dims(law_frequency, 1)))
    index = np.argsort(-law_frequency_mat[:, -1])
    law_frequency_mat = law_frequency_mat[index]
    law_cut_point = np.ceil(threshold * law_frequency_mat.shape[0]).astype(int)
    tail_law_index = law_frequency_mat[law_cut_point:, 0]

    # Calculating the frequency of the accusation
    accu_idx = np.arange(num_accu)
    accu_frequency = np.zeros_like(accu_idx)
    for i in range(len(accu_labels_train)):
        accu_frequency[accu_labels_train[i]] += 1
    accu_frequency_mat = np.hstack((np.expand_dims(accu_idx, 1), np.expand_dims(accu_frequency, 1)))
    index = np.argsort(-accu_frequency_mat[:, -1])
    accu_frequency_mat = accu_frequency_mat[index]

    accu_cut_point = np.ceil(threshold * accu_frequency_mat.shape[0]).astype(int)
    tail_accu_index = accu_frequency_mat[accu_cut_point:, 0]

    return tail_law_index, tail_accu_index


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def ctm(f_train, f_valid, f_test, word2id_dict, word_embedding, iteration=20, seed=0, gpu_on=False, n_trials=1,
        **unused):
    progress = WorkSplitter()

    progress.section("CTM: Pre-calculate long-tail categories of laws and crimes")
    tail_law_index, tail_accu_index = get_tail_index(f_train)

    progress.section("CTM: Set the random seed")
    setup_seed(seed)

    progress.section("CTM: Training")
    tuner = Tuner()
    trials, best_params = tuner.tune(n_trials=n_trials, train=f_train, valid=f_valid, test=f_test,
                                     word2id_dict=word2id_dict, tail_law_index=tail_law_index,
                                     tail_accu_index=tail_accu_index, word_embedding=word_embedding, epoch=iteration,
                                     seed=seed, gpu_on=gpu_on)
    return trials, best_params