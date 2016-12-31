import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LSTM, Lambda, Merge, Reshape
from keras.optimizers import SGD
import numpy as np
from keras import backend as K

sess = tf.Session
K.set_session(sess)

def cnn_lstm(max_num_sents, num_filters, ngrams, wordvec_dim, max_sent_len, cnn_output_dim, sentimen_out_dim):
    sess = tf.Session
    K.set_session(sess)

    for sent in xrange(max_num_sents):
        for ngram in ngrams:
            model =
    return model

def pearsonr(list1, list2):
    return np.corrcoef(list1, list2)[0, 1]
