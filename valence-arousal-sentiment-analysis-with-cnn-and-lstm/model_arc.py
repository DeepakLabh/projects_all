from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LSTM, Lambda, Merge, Reshape
from keras.optimizers import SGD
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

def cnn_lstm(max_num_sents, num_filters, ngrams, wordvec_dim, max_sent_len, cnn_output_dim, sentimen_out_dim):
    cnn_sent_model = []
    for sent in xrange(max_num_sents):
        conv_list = []
        for ngram in ngrams:
            temp_model = Sequential()
            temp_model.add(Convolution2D(num_filters,ngram,wordvec_dim, border_mode = 'valid', input_shape = (1, max_sent_len,wordvec_dim)))
            temp_model.add(Activation('relu'))
            temp_model.add(MaxPooling2D(pool_size=((max_sent_len - ngram + 1), 1)))
            temp_model.add(Flatten())

            temp_model.add(Dropout(0.30))
            temp_model.add(Dense(128))
            # print temp_model.output_shape,11111111111111
            conv_list.append(temp_model)
        model = Sequential()
        model.add(Merge(conv_list, mode = 'concat',concat_axis = -1))
        # model.add(Dropout(0.25))
        # model.add(Dense(output_dim=128))
        cnn_sent_model.append(model)
    model = Sequential()
    model.add(Merge(cnn_sent_model, mode = 'concat', concat_axis = -1))
    input_shape = model.output_shape
    # print model.output_shape, 33333333333333333333333
    model.add(Reshape((len(ngrams)*max_num_sents, cnn_output_dim), input_shape=input_shape))
    #model.add(LSTM(128, return_sequences=False, go_backwards=False, activation='tanh', inner_activation='hard_sigmoid'))
    model.add(LSTM(len(ngrams)*max_num_sents*cnn_output_dim, return_sequences=False, go_backwards=False, activation='tanh', inner_activation='hard_sigmoid'))
    model.add(Reshape((len(ngrams)*max_num_sents, cnn_output_dim), input_shape=model.input_shape))
    model.add(LSTM(128, return_sequences=False, go_backwards=True, activation='tanh', inner_activation='hard_sigmoid'))
    # print model.input_shape,2222222222222
    # print model.output_shape,444444444444

  # model.add(Dense(128))
    # model.add(Activation('tanh'))

    #model.add(Dense(64))
    #model.add(Activation('sigmoid'))
    # print model.output_shape,5555555555
    model.add(Dense(sentimen_out_dim))
    # model.add(Activation('softmax'))
    model.add(Activation('relu'))
    # print model.output_shape,666666666666
    # model.compile(loss = 'categorical_crossentropy', optimizer='adagrad', metrics = ['accuracy'])
    model.compile(loss = 'mean_squared_error', optimizer='adam')
    #model.compile(loss = 'mean_squared_error', optimizer='adam', metrics = ['accuracy'])
    ##################################################################################################3
    return model

def pearsonr(list1, list2):
    return np.corrcoef(list1, list2)[0, 1]

