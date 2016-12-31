# -*- coding: utf-8 -*-
import numpy as np
from scipy import spatial
# from pymongo import MongoClient
# from nltk import sent_tokenize

def an_index(tree, collection, wordvec_dim):
    try:
        tree.load('../data/word_vec_tree.ann')
    except:
        data = collection.find()
        for i in data:
            try:
                tree.add_item(i['id'], list(i['vec']))
            except Exception as e:
                print e,'adding error'
        tree.build(10)
        tree.save('../data/word_vec_tree.ann')

def an_search(tree, collection, query_word, wordvec_dim):
    #v = np.zeros(wordvec_dim)
    #count = 1
    v = collection.find_one({'_id':query_word})['vec']
    result_id = tree.get_nns_by_vector(list(v), 5)
    # print result_id,1111111111
    result = [i['_id'] for i in collection.find({'id':{'$in':result_id}})]
    return result

def cosine_distance(sent1, sent2, collection):
    sum1, sum2=[np.zeros(300), np.zeros(300)]
    count = 0
    for i in sent1.split():
        try:
            sum1+=collection.find_one({'_id':i})['vec']
            count +=1
        except:
            pass
    if count == 0: count = 1
    sum1 = sum1/count

    count = 0
    for i in sent2.split():
        try:
            sum2+=collection.find_one({'_id':i})['vec']
            count +=1
        except:
            pass
    if count == 0: count = 1
    sum2 = sum2/count
    result = 1 - spatial.distance.cosine(sum1, sum2)
    # print result, 'cosine similarity with::- ', sent2
    return result

def new_data(tree, data, collection, threshold_g, threshold_l):
    result_data = [data]
    dict = {}
    for i in list(set(data.strip().split())):
        try:
            for new_word in an_search(tree, collection, i, 300):
                if cosine_distance(i, new_word, collection) > threshold_g and cosine_distance(i, new_word, collection) < threshold_l:
                    # print new_word, i,11111111111
                    if i != new_word:
                        result_data.append(data.replace(i, new_word))
                        try:
                            dict[i].append(new_word)
                        except:
                            dict[i] = []
        except: pass
    return result_data
