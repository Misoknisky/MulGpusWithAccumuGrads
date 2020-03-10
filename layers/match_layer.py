# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module implements the core layer of Match-LSTM and BiDAF
"""

import tensorflow as tf
import tensorflow.contrib as tc


class MatchLSTMAttnCell(tc.rnn.LSTMCell):
    """
    Implements the Match-LSTM attention cell
    """
    def __init__(self, num_units, context_to_attend):
        super(MatchLSTMAttnCell, self).__init__(num_units, state_is_tuple=True)
        self.context_to_attend = context_to_attend
        self.fc_context = tc.layers.fully_connected(self.context_to_attend,
                                                    num_outputs=self._num_units,
                                                    activation_fn=None)

    def __call__(self, inputs, state, scope=None):
        (c_prev, h_prev) = state
        with tf.variable_scope(scope or type(self).__name__):
            ref_vector = tf.concat([inputs, h_prev], -1)
            G = tf.tanh(self.fc_context
                        + tf.expand_dims(tc.layers.fully_connected(ref_vector,
                                                                   num_outputs=self._num_units,
                                                                   activation_fn=None), 1))
            logits = tc.layers.fully_connected(G, num_outputs=1, activation_fn=None)
            scores = tf.nn.softmax(logits, 1)
            attended_context = tf.reduce_sum(self.context_to_attend * scores, axis=1)
            new_inputs = tf.concat([inputs, attended_context,
                                    inputs - attended_context, inputs * attended_context],
                                   -1)
            return super(MatchLSTMAttnCell, self).__call__(new_inputs, state, scope)

class MatchLSTMLayer(object):
    """
    Implements the Match-LSTM layer, which attend to the question dynamically in a LSTM fashion.
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        Match the passage_encodes with question_encodes using Match-LSTM algorithm
        """
        with tf.variable_scope('match_lstm'):
            cell_fw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            cell_bw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                             inputs=passage_encodes,
                                                             sequence_length=p_length,
                                                             dtype=tf.float32)
            match_outputs = tf.concat(outputs, 2)
            state_fw, state_bw = state
            c_fw, h_fw = state_fw
            c_bw, h_bw = state_bw
            match_state = tf.concat([h_fw, h_bw], 1)
        return match_outputs, match_state
class AttentionFlowMatchLayer(object):
    """
    Implements the Attention Flow layer,
    which computes Context-to-question Attention and question-to-context Attention
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        Match the passage_encodes with question_encodes using Attention Flow Match algorithm
        """
        with tf.variable_scope('bidaf'):
            sim_matrix = tf.matmul(passage_encodes, question_encodes, transpose_b=True)
            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), question_encodes)
            b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)
            question2context_attn = tf.tile(tf.matmul(b, passage_encodes),
                                         [1, tf.shape(passage_encodes)[1], 1])
            concat_outputs = tf.concat([passage_encodes, context2question_attn,
                                        passage_encodes * context2question_attn,
                                        passage_encodes * question2context_attn], -1)
            return concat_outputs, None
        
class CrossAttention(object):
    def __init__(self,hidden_size):
        self.hidden_size = hidden_size
    def match(self, passage_encodes, question_encodes, p_length, q_length):
        with tf.variable_scope("DCA"):
            sim_matrix = tf.matmul(passage_encodes, question_encodes, transpose_b=True)#pq
            A=tf.nn.softmax(sim_matrix,axis=-1)#bpq
            B=tf.nn.softmax(tf.transpose(sim_matrix,perm=[0,2,1]),axis=-1)#bqp
            contextp=tf.matmul(A,question_encodes)
            contextq=tf.matmul(B,passage_encodes) 
            R=tf.matmul(contextp,contextq, transpose_b=True)
            ra=tf.nn.softmax(R,axis=-1)#bpq
            D=tf.reduce_sum(tf.einsum("bpq,bqd->bpqd",ra,contextq),axis=2)#bpd
            G=tf.concat([passage_encodes,contextp,D],axis=-1)
            return G,None
class SelfAttLayer(object):

    def __init__(self, hidden):
        self.hidden = hidden

    def bi_linear_att(self, inputs, memory):#pwq
        with tf.variable_scope("self_attention"):
            i_dim = inputs.get_shape().as_list()[-1]#tf.einsum()
            flat_inputs = tf.reshape(inputs, [-1, i_dim])
            m_dim = inputs.get_shape().as_list()[-1]
            weight = tf.get_variable("W", [i_dim, m_dim])

            shape = tf.shape(inputs)
            out_shape = [shape[idx] for idx in range(len(inputs.get_shape().as_list()) - 1)] + [m_dim]
            result_input = tf.reshape(tf.matmul(flat_inputs, weight), out_shape)

            outputs = tf.nn.relu(tf.matmul(result_input, tf.transpose(memory, [0, 2, 1]))) / (self.hidden ** 0.5)
            logits = tf.nn.softmax(outputs)
            outputs = tf.matmul(logits, memory)
            return tf.concat([inputs, outputs], axis=2)


class GatedLayer(object):

    def __init__(self, hidden):
        self.hidden = hidden

    def gate(self, res):
        with tf.variable_scope("gated_layer"):
            gate = tf.nn.sigmoid(self.fully_connect(res))
            return gate

    def fully_connect(self, inputs, scope="dense"):
        """
            3-Dimension fully_connect
        """
        with tf.variable_scope(scope):
            shape = tf.shape(inputs)
            dim = inputs.get_shape().as_list()[-1]
            out_shape = [shape[idx] for idx in range(len(inputs.get_shape().as_list()) - 1)] + [self.hidden]
            flat_inputs = tf.reshape(inputs, [-1, dim])
            weight = tf.get_variable("W", [dim, self.hidden])
            result = tf.matmul(flat_inputs, weight)
            result = tf.reshape(result, out_shape)
            return result


class MultiHeadBIDAF(object):
    def __init__(self,hidden_size):
        self.hidden_size = hidden_size
    def multihead_attention(self,queries, keys, values,num_heads=6,scope="multihead_attention"):
        d_model = queries.get_shape().as_list()[-1]
        assert d_model % num_heads == 0
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Linear projections
            Q = tf.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
            K = tf.layers.dense(keys, d_model, use_bias=True)  # (N, T_k, d_model)
            V = tf.layers.dense(values, d_model, use_bias=True)  # (N, T_k, d_model)
            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
            # Attention
            outputs = self.scaled_dot_product_attention(Q_, K_, V_)
            print(outputs)
            # Restore shape
            #outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)
            # Residual connection
            #outputs += queries
            # Normalize
            #outputs = self.ln(outputs)
        return outputs

    def ln(self,inputs, epsilon=1e-8, scope="ln"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta
        return outputs

    def scaled_dot_product_attention(self,Q, K, V, scope="scaled_dot_product_attention"):
        '''See 3.2.1.
        Q: Packed queries. 3d tensor. [N, T_q, d_k].
        K: Packed keys. 3d tensor. [N, T_k, d_k].
        V: Packed values. 3d tensor. [N, T_k, d_v].
        key_masks: A 2d tensor with shape of [N, key_seqlen]
        causality: If True, applies masking for future blinding
        dropout_rate: A floating point number of [0, 1].
        training: boolean for controlling droput
        scope: Optional scope for `variable_scope`.
        '''
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            d_k = Q.get_shape().as_list()[-1]
            # dot product
            outputs = tf.matmul(Q,tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)
            # scale
            #outputs /= d_k ** 0.5
            # softmax
            #outputs = tf.nn.softmax(outputs)
            # weighted sum (context vectors)
            #outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)
        return outputs

    def match(self,passage_encodes, question_encodes, p_length, q_length,num_heads=6):
        d_model = passage_encodes.get_shape().as_list()[-1]
        assert d_model % num_heads == 0
        matt_dim = d_model // num_heads
        with tf.variable_scope("multihead_attention", reuse=tf.AUTO_REUSE):
            Q = tf.layers.dense(passage_encodes, d_model, use_bias=True)  # (b, p, d_model)
            K = tf.layers.dense(question_encodes, d_model, use_bias=True)  # (N, q, d_model)
            Q = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            K = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            Q = tf.reshape(Q,[-1,tf.shape(passage_encodes)[1],num_heads,matt_dim]) #bpnd
            K = tf.reshape(K,[-1, tf.shape(question_encodes)[1], num_heads, matt_dim])#bqnd
            Q = tf.transpose(Q,perm=[0,2,1,3])#bnpd
            K = tf.transpose(K,perm=[0,2,1,3])#bnqd
            sim_matrix = tf.einsum("bnpd,bnqd->bnpq",Q,K)
            context2question_attn=tf.einsum("bnpq,bnqd->bnpd",tf.nn.softmax(sim_matrix,axis=-1),K)#bnpd
            b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, -1), 2), -1)#bn1p
            question2context_attn=tf.tile(tf.einsum("bntp,bnpd->bntd",b,Q),multiples=[1,1,tf.shape(passage_encodes)[1],1])
            concat_outputs = tf.concat([Q,context2question_attn,
                                      Q*context2question_attn,Q*question2context_attn],axis=-1)
            concat_outputs = tf.reshape(concat_outputs,shape=[-1,tf.shape(passage_encodes)[1],d_model*4])
        return concat_outputs, None































































 
  
   
    
     
      
       




















