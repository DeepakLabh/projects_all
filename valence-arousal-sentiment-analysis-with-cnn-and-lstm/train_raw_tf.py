# -*- coding: utf-8 -*-
import csv
from nltk import sent_tokenize
import model_arc as ma
import numpy as np
from pymongo import MongoClient
from keras import backend as K
import model_arc as ma
import re
import matplotlib.pyplot as plt
import pickle as pk
c = MongoClient('192.168.86.164')
coll_vec = c.wordvec.general

max_sent_len = 20
max_num_sents = 7
nb_epoch = 100
wordvec_dim = 300
num_filters = 32
cnn_output_dim = 128
ngrams = [3,4]
sentimen_out_dim = 4
zero_vec = np.zeros(wordvec_dim)
zero_vec_sent = np.zeros((max_sent_len,wordvec_dim))
rand_vec = np.random.random(wordvec_dim)
#print '111111111111111111111111111111111111111'
r = csv.reader(open('dataset-fb-valence-arousal-anon.csv'))

x = []
y = []
x_train = []
y_train = []
############ Create data ####################
try:
    x_d,y = pk.load(open('../data/training_data.pkl', 'r'))
except Exception as pickle_load_error:
    print pickle_load_error, 'Creating training data'
    for i in r:
        try:
            float(i[1])
        except:
            continue
        sents = sent_tokenize(i[0].decode('utf-8'))
        para_vec = []
        for sent in sents:
            words = sent.strip().split()
            sent_vec = []
            for word in words:
                try:
                    vec = coll_vec.find_one({'_id':word})['vec']
                except:
                    vec = zero_vec
                if len(re.findall(r'[0-1]+',word))>0:
                    vec = zero_vec
                sent_vec.append(vec)
            sent_vec = sent_vec[:max_sent_len] if len(sent_vec)>max_sent_len else sent_vec+[zero_vec]*(max_sent_len-len(sent_vec))

            para_vec.append(sent_vec)
        para_vec = para_vec[:max_num_sents] if len(para_vec)>max_num_sents else para_vec+[zero_vec_sent]*(max_num_sents-len(para_vec))
        x.append(para_vec)
        y.append([i[1:5]])
    x_d = []
    for i in xrange(max_num_sents):
        x_d.append([])
    for i in xrange(len(x)):
        for j in xrange(len(x[i])):
            x_d[j].append([x[i][j]])
            # x_d[j].append(x[i][j])
    x_d = np.array(x_d)
    x_d = list(np.array(x_d))*len(ngrams)
    try:
        pk.dump([x_d,y], open('../data/training_data.pkl', 'w'))
    except Exception as e:
        print e, 'pickling error'
        pass
    print 'data created'

train_size = int(len(y)*0.95)
print train_size
x_d_train = [i[:train_size] for i in x_d]
y_train = np.array(y[:train_size])
y_train = y_train.reshape(len(y_train), sentimen_out_dim)
x_d_test = [i[train_size:] for i in x_d]
y_test = np.array(y[train_size:])
y_test = y_test.reshape(len(y_test), sentimen_out_dim)
# y = np.array(y)
# y = y.reshape((len(y),sentimen_out_dim))

model = ma.cnn_lstm(max_num_sents,num_filters,ngrams,wordvec_dim,max_sent_len,cnn_output_dim,sentimen_out_dim)

print "Model Created"
print "--------------------"
print "Training Started"
# print len(list(np.array(x_d))*len(ngrams)),len(ngrams)
# print y.shape,11111111111111
history = model.fit(x_d_train, y_train, validation_split=0.10, nb_epoch=nb_epoch, batch_size=32, verbose=1)
#########################3 PLOT loss and accuracy ##############
print(history.history.keys())
# summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
#plt.savefig('accuracy.jpg')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('loss.jpg')
#########################3 PLOT loss and accuracy ##############
print "Training Completed"
model.save_weights('../data/cnn_lstm.weights',overwrite=True)
loss_and_metrics = model.evaluate(x_d_test, y_test, batch_size=32)
predicted_test_result = model.predict_on_batch(x_d_test)
pearsonr = ma.pearsonr(predicted_test_result.astype(np.float), y_test.astype(np.float))
print loss_and_metrics
print 'pearson coefficient: ', pearsonr
