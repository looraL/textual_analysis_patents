#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 21:45:47 2018

@author: lizhuoran
"""

import pandas as pd
import numpy as np
import re
import itertools
from collections import Counter
import os

from bs4 import BeautifulSoup
import nltk
#nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, concatenate, Lambda, BatchNormalization, Activation
from keras.layers import GlobalMaxPooling1D
from keras.models import Model, Sequential
from keras import backend as K
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

import matplotlib.pyplot as plt

from hyperopt import hp, fmin, tpe,  STATUS_OK, Trials
from hyperopt.mongoexp import MongoTrials

import random
import string
import json

from deepexplain.tensorflow import DeepExplain
import tensorflow as tf
from tensorflow import Tensor


prepare_data = True
interpret = False
prepare_json = False

CLEAN_TEXT = True
NUM_CATEGORY = 4
MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 20000 # number of words in vocabulary
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.333

def clean_doc(string, CLEAN_TEXT = True):
    if CLEAN_TEXT: 
        #string = BeautifulSoup(string).get_text()
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]"," ", string)
        string = re.sub(r"\\", "", string)      
        string = re.sub(r"\"", "", string)
    
    return string.encode('utf-8')


def clean_str(sentence, stop_words, CLEAN_TEXT = True):
    """
    string cleaning for dataset
    All in lowercase
    """
    # Remove HTML syntax
    
    words = sentence.split()
    
    if CLEAN_TEXT:
        # remove remaining tokens that are not alphabetic
        words = [word for word in words if word.isalpha()]
        # set to lowercase
        words = [x.lower() for x in words]
        # filter out stop words
        words = [w for w in words if not w in stop_words]
        # filter out short tokens
        #words = [word for word in words if len(word) > 1]
    
    return words


def prepare_train_test(data, train_size=300):
    """
    Randomly shuffle data, tokenize and do word-level one-hot embedding on texts.
    Split the data into training set and validation set.
    Return x_train, y_train, texts_train, x_test, y_test, texts_test.
    """
    
    # clean string, format texts, labels into lists
    texts = []
    labels = []
    for ind in range(data.appline.shape[0]):
        # Remove HTML syntax
        #text = BeautifulSoup(data.review[ind])
        text = data.appline[ind]
        # encountering UnicodeDecodeError: unexpected end of data
        # https://stackoverflow.com/questions/24004278/unicodedecodeerror-utf8-codec-cant-decode-byte-0xc3-in-position-34-unexpect
        # if we remove all the conflicts
        #texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        texts.append(clean_doc(text, CLEAN_TEXT=True))
        labels.append(data.category[ind])
        
    # pairwise shuffle on texts and labels 
    pack_texts_labels = list(zip(texts, labels))
    random.shuffle(pack_texts_labels)
    texts, labels = zip(*pack_texts_labels)
    texts = texts[:train_size]
    labels = labels[:train_size]
    
    stop_words = set(stopwords.words('english'))
    token_index = {}
    for appline in texts:
        # strip punctuation and special characters from the review sentences.
        # then tokenize the reviews via the `split` method.
        appline_cleaned = clean_str(appline, stop_words, CLEAN_TEXT=True)
        for word in appline_cleaned:
            if word not in token_index:
                # Assign a unique index to each unique word
                token_index[word] = len(token_index) + 1
                # Note that 0 is not attributed to anything.
      
    # vectorization
    # only consider the first MAX_SEQUENCE_LENGTH words in each review.      
    # this is where we store the one-hot embedding results:
    oneHot_result = np.zeros((len(texts), MAX_SEQUENCE_LENGTH))
    
    for i, appline in enumerate(texts):
        appline = clean_str(appline, stop_words, CLEAN_TEXT=True)
        for j, word in list(enumerate(appline))[:MAX_SEQUENCE_LENGTH]:
            index = token_index.get(word)
            oneHot_result[i, j] = index
###################################################    
            
     #format label
    labels = to_categorical(np.asarray(labels))
    labels = np.delete(labels, 0, 1)
    
    # split train, test dataset using VALIDATION_SPLIT ratio
    #reverse_word_map = dict(map(sequences[0], word_index.items()))
    nb_validation_samples = int(VALIDATION_SPLIT * len(texts))
    
    x_train = oneHot_result[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    texts_train = texts[:-nb_validation_samples]
    x_test = oneHot_result[-nb_validation_samples:]
    y_test = labels[-nb_validation_samples:]
    texts_test = texts[-nb_validation_samples:]

    return x_train, y_train, texts_train, x_test, y_test, texts_test, token_index

if prepare_data:
    df = pd.read_excel('../20180419SampleText.xlsx', sheet_name='Sheet1').iloc[:300]
    category_dict = {'Pure Background': 1, 'Tool/Technique/Formula/Input': 2,
                     'Motivation for Research/Difference from Existing Inventions': 3, 
                     'Similar concept being patented, idea that can be used with invention, or potential use of invention': 4,
                     'Impossible to Tell': 5}
    if NUM_CATEGORY == 4: 
        category_dict = {'Pure Background': 1, 'Tool/Technique/Formula/Input': 2,
                     'Motivation for Research/Difference from Existing Inventions': 1, 
                     'Similar concept being patented, idea that can be used with invention, or potential use of invention': 3,
                     'Impossible to Tell': 4}
    for cate in category_dict: 
        df.loc[df[cate] == 1, 'category'] = category_dict[cate]
        
    df = df[['first_author', 'appline', 'category']]
   
    x_train, y_train, texts_train, x_test, y_test, texts_test, token_index = prepare_train_test(df)

    # compute an index mapping words to Glove embeddings
    #GLOVE_DIR = "../Glove"
    embeddings_index = {}
    # (Zhang et al.) suggested 300d gives best performance
    f = open('Glove/glove.6B.50d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    # leverage our embedding index and word index to compute embedding matrix
    embedding_matrix = np.zeros((len(token_index) + 1, EMBEDDING_DIM))
    for word, i in token_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            # size (4343, 50)
            embedding_matrix[i] = embedding_vector
if interpret:
    
    with DeepExplain(session=K.get_session()) as de:  
        # Need to reconstruct the graph in DeepExplain context, using the same weights.
        # explain(method_name, target_tensor, input_tensor, samples, ...args)
        # samples: np-array required
        
        model = load_model('model-036.h5')   
        # y_pred: prob. for each class    
        y_pred = model.predict(x_test)
        
        # reference to tensors
        input_tensor = model.get_layer("input_x").input
        embedding = model.get_layer("embedding").output
        pre_softmax = model.get_layer("dense2").output
        
        x_interpret = x_test[9:10]        
        # perform embedding lookup
        get_embedding_output = K.function([input_tensor],[embedding])
        embedding_out = get_embedding_output([x_interpret])[0]        
        
        # target the output of the last dense layer (pre-softmax)
        # To do so, create a new model sharing the same layers untill the last dense (index -2)
        fModel = Model(inputs=input_tensor, outputs = pre_softmax)
        target_tensor = fModel(input_tensor)
        
        # to target a specific neuron(class), we apply a binary map
        ys = [0, 1, 0, 0]
        
        attributions = de.explain('elrp', pre_softmax*ys, embedding, embedding_out)

        text_dec = []
        for x in np.nditer(x_interpret):
            if x in token_index.values():
                text_dec.append(token_index.keys()[token_index.values().index(x)])
                
        p_gens = np.sum(attributions,axis=2).tolist()
        new_pgens = []
        for gen in p_gens[0]:
            new_pgens.append([gen])
            
        abstract_str = texts_test[9]
        true_label = y_test[9:10].argmax(axis=1).tolist()
        pred_prob = y_pred[9:10].tolist()
        pred_label = y_pred[9:10].argmax(axis=1).tolist()
        
        format_data = {'abstract_str': abstract_str, 'decoded_lst': text_dec, 
                       'p_gens': new_pgens, 'true_label': true_label, 'pred_label': pred_label,
                       'pred_prob': pred_prob}

if prepare_json:
    with open('sig_vis_data.json', 'w') as outfile:
        json.dump(format_data, outfile)
        
        
