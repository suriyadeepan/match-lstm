import tensorflow as tf
import numpy as np

from lib.vectorize import vectorize
from lib.data_utils import unpack_data

import model


if __name__ == '__main__':

    # model parameters
    window_size = 4
    sentence_size = 80
    embedding_size = 300
    batch_size = 256
    num_class = 3
    nepochs = 10000
    lr = 0.001

    # get vectorized data 
    ddict = vectorize(sent_size=sentence_size)
    vocab_size = len(list(ddict['word2index']))
    train_data = ddict['train_data']

    # build model
    mlstm = model.mLSTM(sentence_size, 
            num_class, 
            embedding_size, 
            batch_size, 
            vocab_size,
            lr)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # training loop
        except_cnt = 0
        for j in range(nepochs):
            avg_loss = 0
            for i in range(len(train_data)//batch_size):
                try:
                    # fetch data
                    data = unpack_data(train_data[i*batch_size : (i+1)*batch_size], num_class=3)
                    loss_v = sess.run(mlstm.loss, feed_dict= {
                        mlstm.premises : np.array(data[0]),
                        mlstm.hypotheses : np.array(data[1]),
                        mlstm.labels : np.array(data[3])
                    })
                    avg_loss += loss_v
                except:
                    except_cnt += 1

                if i%1000 == 0 and i:
                    print('[j={}; i={}] : {}'.format(j, i, avg_loss/(1000-except_cnt)))
                    print('#exceptions : ', except_cnt)
                    avg_loss = 0
                    except_cnt = 0
