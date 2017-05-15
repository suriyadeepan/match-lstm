import tensorflow as tf
import numpy as np


class mLSTM(object):

    def __init__(self, sentence_size, num_class, embedding_size, batch_size, vocab_size, lr):

        self.sentence_size = sentence_size
        self.num_class = num_class
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.lr = lr

        def __graph__():

            # clear graph
            tf.reset_default_graph()

            # placeholders
            premises = tf.placeholder(shape=[None, sentence_size], dtype=tf.int32, name='P')
            hypotheses = tf.placeholder(shape=[None, sentence_size], dtype=tf.int32, name='H')
            labels = tf.placeholder(shape=[None, num_class ] , dtype=tf.int32, name='labels')

            # embedding
            def embed_line(i, inputs, embeddings, emb_ta):
                emb_list = []
                for j in range(sentence_size):
                    word = inputs[i][j]
                    unk_word = tf.constant(-1)
                    f1 = lambda : tf.nn.embedding_lookup(params=emb, ids=word)
                    f2 = lambda : tf.zeros(shape=[embedding_size])
                    word_emb = tf.case([(tf.not_equal(unk_word, word), f1)], default=f2)
                    emb_list.append(word_emb)
                emb_tensor = tf.stack(emb_list)
                emb_ta = emb_ta.write(i, emb_tensor)
                i = tf.add(i,1)
                return i, inputs, embeddings, emb_ta

            def embed_sentences(sentences, embeddings):
                emb_ta = tf.TensorArray(dtype=tf.float32, size=batch_size)
                i = tf.constant(0)
                c = lambda x,y,z,n : tf.less(x, batch_size)
                b = lambda x,y,z,n : embed_line(x,y,z,n)
                emb_res = tf.while_loop(cond=c, body=b, loop_vars=(i, sentences, emb, emb_ta) )
                emb_tensor = emb_res[-1].stack()
                return tf.reshape(emb_tensor, [-1, sentence_size, embedding_size])


            emb = tf.get_variable(name='emb', shape=[vocab_size, embedding_size])
            p_embs = embed_sentences(premises, emb)
            h_embs = embed_sentences(hypotheses, emb)


            # - sequence length helper function
            def seq_len(seq):
                seq_bool = tf.sign(tf.abs(seq))
                return tf.reduce_sum(seq_bool, axis=-1)


            # Representation
            with tf.variable_scope('lstm_premises'):
                lstm_p_cell = tf.contrib.rnn.BasicLSTMCell(num_units=embedding_size)
                # get actual length of premises
                premises_len = seq_len(premises)
                h_s, _ = tf.nn.dynamic_rnn(cell=lstm_p_cell, inputs=p_embs, sequence_length=premises_len,
                                 dtype=tf.float32)

            with tf.variable_scope('lstm_hypotheses'):
                lstm_h_cell = tf.contrib.rnn.BasicLSTMCell(num_units=embedding_size)
                # get actual length of premises
                hyp_len = seq_len(hypotheses)
                h_t, _ = tf.nn.dynamic_rnn(cell=lstm_h_cell, inputs=h_embs, sequence_length=hyp_len,
                                 dtype=tf.float32)

            # final LSTM
            lstm_m_cell = tf.contrib.rnn.BasicLSTMCell(num_units=embedding_size)


            # Attention
            def match_attention(k, p_emb, h_emb, len_p, state):
                h_emb_k = tf.reshape(h_emb[k], [1, -1])
                p_emb_k = tf.slice(p_emb, begin=[0,0], size=[len_p, embedding_size])
                
                with tf.variable_scope('attn_weights'):
                    w_s = tf.get_variable(shape=[embedding_size, embedding_size],
                                           name='w_s')
                    w_t = tf.get_variable(shape=[embedding_size, embedding_size],
                                           name='w_t')
                    w_m = tf.get_variable(shape=[embedding_size, embedding_size],
                                           name='w_m')
                    w_e = tf.get_variable(shape=[embedding_size, 1],
                                          name='w_e')
                m_lstm_state = state.h
                sum_m = tf.matmul(p_emb_k, w_s) + tf.matmul(h_emb_k, w_t) + tf.matmul(m_lstm_state, w_m)
                alpha_k = tf.nn.softmax(tf.matmul(tf.tanh(sum_m), w_e))
                a_k = tf.matmul(alpha_k, p_emb_k, transpose_a=True)
                a_k.set_shape([1, embedding_size])
                
                m_k = tf.concat([a_k, h_emb_k], axis=1)
                with tf.variable_scope('lstm_m_step'):
                    _, next_state = lstm_m_cell(inputs=m_k, state=state)
                
                k = tf.add(k,1)
                
                return k, p_emb, h_emb, len_p, next_state

            def match_sentence(i, h_m_ta):
                p_emb_i, h_emb_i = p_embs[i], h_embs[i]
                len_p_i, len_h_i = seq_len(premises[i]), seq_len(hypotheses[i])
                state = lstm_m_cell.zero_state(batch_size=1, dtype=tf.float32)
                
                # inner loop
                k = tf.constant(0)
                c = lambda a, x, y, z, s : tf.less(a, len_h_i)
                b = lambda a,x,y,z,s : match_attention(a,x,y,z,s)
                res = tf.while_loop(cond=c, body=b, 
                                   loop_vars=(k, p_emb_i, h_emb_i, len_p_i, state ))
                
                h_m_ta = h_m_ta.write(i, res[-1].h)
                
                i = tf.add(i,1)
                return i, h_m_ta


            # final representation
            with tf.variable_scope('lstm_m'):
                h_m_ta = tf.TensorArray(dtype=tf.float32, size=batch_size)
                c = lambda x, y : tf.less(x, batch_size)
                b = lambda x, y : match_sentence(x,y)
                i = tf.constant(0)
                h_m_res = tf.while_loop(cond=c, body=b,
                                       loop_vars = (i, h_m_ta))
                
                h_m_tensor = tf.squeeze(h_m_res[-1].stack(), axis=1)

            # softmax layer
            with tf.variable_scope('fully_connected'):
              w_fc = tf.get_variable(shape=[embedding_size,num_class], name='w_fc')
              b_fc = tf.get_variable(shape=[num_class], name='b_fc')
              logits = tf.matmul(h_m_tensor, w_fc) + b_fc


            # Optimization
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_sum(cross_entropy)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999)
            train_op = optimizer.minimize(loss)

            # attach graph nodes to class
            self.train_op = train_op
            self.loss = loss
            self.logits = logits
            #   placeholders
            self.premises = premises
            self.hypotheses = hypotheses
            self.labels = labels

        # build graph
        __graph__()
