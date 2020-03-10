#coding=utf-8
'''
Created on 2020年3月4日
@author: Administrator
@email: 1113471782@qq.com
'''
"""
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""
import os
import time
import logging
import json
import numpy as np
import tensorflow as tf
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import CrossAttention
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder
from layers.match_layer import SelfAttLayer
from layers.match_layer import GatedLayer
from layers.loss_func import cul_weighted_avg_loss
from layers.match_layer import MultiHeadBIDAF
class Model(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, vocab,args):

        # logging
        self.logger = logging.getLogger("brc")
        self.algo = args.algo
        self.config=args
        self.hidden_size = args.hidden_size#150
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1
        # length limit
        self.max_p_num = args.max_p_num#5
        self.max_p_len = args.max_p_len#500
        self.max_q_len = args.max_q_len#60
        self.max_a_len = args.max_a_len#200
        self.vocab = vocab
        self._build_graph()
    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._fuse()
        self._selfatt() 
        self._decode()
        self._compute_loss()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        #param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        #self.logger.info('There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32, [None, None])#passage encdoing
        self.q = tf.placeholder(tf.int32, [None, None])#question_encoding
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None, None], name='start_label')
        self.end_label = tf.placeholder(tf.int32, [None, None], name='end_label')
        self.match_score = tf.placeholder(tf.float32, [None, None], name='match_score')
        self.normed_match_score = tf.div_no_nan(self.match_score, tf.reduce_sum(self.match_score, axis=1, keepdims=True))
        self.dropout_keep_prob = tf.placeholder(tf.float32)
    
    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(np.asarray(self.vocab.embedding_matrix, dtype=np.float32)),
                trainable=True
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.p_length, self.hidden_size)

        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """
        if self.algo == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)#150
        elif self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)#150
        elif self.algo=='DCA':
            match_layer = CrossAttention(self.hidden_size)
        elif self.algo=="MBIDAF":
            match_layer = MultiHeadBIDAF(self.hidden_size)
        else:
            raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
        '''
        BIDAF:return G
        MATLSTM:
        '''
        self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes, self.sep_q_encodes,
                                                    self.p_length, self.q_length)
        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        with tf.variable_scope('fusion'):
            self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes, self.p_length,self.hidden_size, layer_num=1)
            gated_layer = GatedLayer(2 * self.hidden_size)
            gated = gated_layer.gate(self.fuse_p_encodes)
            self.gated_p_encodes = gated * self.fuse_p_encodes


    def _selfatt(self):
        attention_layer = SelfAttLayer(self.hidden_size)
        with tf.variable_scope("para_attention"):
            self_att = attention_layer.bi_linear_att(self.gated_p_encodes, self.gated_p_encodes)
            self.para_p_encodes, _ = rnn('bi-gru', self_att, self.p_length, self.hidden_size)

        with tf.variable_scope("document_attention"):
            batch_size = tf.shape(self.start_label)[0]
            doc_encodes = tf.reshape(self.para_p_encodes, [batch_size, -1, 2 * self.hidden_size])
            doc_att = attention_layer.bi_linear_att(doc_encodes, doc_encodes)
            self.document_encodes, _ = rnn('bi-lstm', doc_att, None, self.hidden_size)
    def _decode(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same document.
        And since the encodes of queries in the same document is same, we select the first one.
        """
        with tf.variable_scope('same_question_concat'):
            batch_size = tf.shape(self.start_label)[0]
            no_dup_question_encodes = tf.reshape(self.sep_q_encodes,[batch_size, -1, tf.shape(self.sep_q_encodes)[1], 2 * self.hidden_size])[0:, 0, 0:, 0:]
        decoder = PointerNetDecoder(self.hidden_size)
        self.start_probs, self.end_probs = decoder.decode(self.document_encodes,no_dup_question_encodes)

    def _compute_loss(self):
        """
        The loss function
        """

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses

        self.loss = 0
        self.weighted_avg_loss = cul_weighted_avg_loss(self.start_probs, self.end_probs, self.start_label,self.end_label,self.normed_match_score)
        self.loss += self.weighted_avg_loss
        self.all_params = tf.trainable_variables()
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss
