
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Convolution1D, MaxPooling1D
import tensorflow as tf
import json
import re
from collections import Counter

tf.python.control_flow_ops = tf

max_features = 5000
maxlen = 40  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
nb_filter = 64
filter_length =5
hidden_dims = 64
nb_epoch = 25
embedding_dims = 128

x, y1, y2 = [[], [], []] 

data = json.load(open('restraunt_data.json', 'r'))

sents = [i['sentence'] for i in data]
attr = [i['attribute'] for i in data]
pol = [i['polarity'] for i in data]

sentences = []
for i in sents:
	sentences += i.split()

c_sents = Counter(sentences)
embeed = {}
for  i in xrange(len(c_sents.keys())):
	embeed[c_sents.keys()[i]] = i+1
c_attrs = Counter(attr)
attributes_dict = {}
for  i in xrange(len(c_attrs.keys())):
	attributes_dict[c_attrs.keys()[i]] = i+1
polarity = {'positive':1, 'negative':0, 'neutral':2}
nb_classes = len(c_attrs)

x = [[embeed[j] for j in i.split()]for i in sents]
y1 = [attributes_dict[i] for i in attr]
y2 = [polarity[i] for i in pol]

x = sequence.pad_sequences(x, maxlen=maxlen)
x_train, x_test, y1_train, y1_test, y2_train, y2_test = [x[:1500], x[1500:], y1[:1500], y1[1500:], y2[:1500], y2[1500:]]


model = Sequential()
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen,
                    dropout=0.2))# model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use max pooling:
model.add(MaxPooling1D(pool_length=5))

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
model.add(Flatten())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y2_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(x_test, y2_test))
# print('Test score:', score)
# print('Test accuracy:', acc)
