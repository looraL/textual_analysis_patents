#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
an implementation of CNN model to predict pos/neg movie reviews

"""
import pandas as pd
import numpy as np
import re
import itertools
from collections import Counter
import os
import sys

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


#sys.path.append('/Users/lizhuoran/Desktop/UOFT/research/patents/src/deepexplain')

#note: import DeepExplain only once
# I've encountered conflicts if running this line a second time
#from deepexplain.tensorflow import DeepExplain
import tensorflow as tf
from tensorflow import Tensor


# settings and controls

# controls: each corresponds to a section, initialized as False

# prepare training/test sets(x, y, text)
prepare_data = False
# hyper tune range of choices
load_exp_space = False
# hyper tune run trials
hyper_tune = False
# construct model: needs manual input from "hyper_tune" output(variable suggestions)
construct_model_tuned = False
train = False
# accuracy and loss
plot = False
# generate predicted probability for each class, load output into excel
check_prediction = False
# run deepExplain
interpret = False
# prepare a json file for web visualization, load output from deepExplain into json
prepare_json = False

CLEAN_TEXT = True # used in function clean_doc, clean_str 
NUM_CATEGORY = 5 # set to 4 or 5
MAX_SEQUENCE_LENGTH = 150 # max number of words in one sample
MAX_NB_WORDS = 25000 # number of words in vocabulary
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.333


def clean_doc(string, author = None, CLEAN_TEXT= True):
    if CLEAN_TEXT: 
        # better to reserve HTML tags
        #string = BeautifulSoup(string).get_text()
        
        # remove some symbols, common cleaning techniques used for IMBD dataset
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]"," ", string)
        string = re.sub(r"\\", "", string)      
        string = re.sub(r"\"", "", string)
    if not author == None:
        string = string.replace(author, 'name')
        
    return string.encode('utf-8')


def clean_str(sentence, stop_words, CLEAN_TEXT = True):
    """
    string cleaning for dataset
    All in lowercase
    """
    words = sentence.split()
    
    if CLEAN_TEXT:
        # remove remaining tokens that are not alphabetic
        words = [word for word in words if word.isalpha()]
        # set to lowercase
        words = [x.lower() for x in words]
        # filter out stop words
        # imported stop words from nltk
        # words = [w for w in words if not w in stop_words]
        
        # filter out short tokens
        # observation: single letter word was colored in the final visualization, such as 'p'
        #words = [word for word in words if len(word) > 1]
    
    return words


# train_size means total size for labeled data
# split into 200 training and 100 test
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
        
        # replace first author with "name"
#        author = data.first_author[ind]
#        texts.append(clean_doc(text, author, CLEAN_TEXT=True))
        # original last name for authors, if not substitute with "name", authors' last names will be replaced with "SUB" since GLOVE cannot 
        # find matched word for such last names
        
        # save text for visualizations later
        texts.append(clean_doc(text, CLEAN_TEXT=True))
        
        # y_label        
        labels.append(data.category[ind])
        
    # pairwise shuffle on texts and labels 
    pack_texts_labels = list(zip(texts, labels))
    random.shuffle(pack_texts_labels)
    texts, labels = zip(*pack_texts_labels)
    texts = texts[:train_size]
    labels = labels[:train_size]
     
    # word-level one-hot embedding without Keras
    # output np array(# of review, length of review), with word index as value
    # build an index of all tokens in the data.
     
    # filter out stop_words
    # token_index equivalent to word_index in the commented keras one-hot embedding block
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
    nb_validation_samples = int(VALIDATION_SPLIT * len(texts))
    
    x_train = oneHot_result[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    texts_train = texts[:-nb_validation_samples]
    x_test = oneHot_result[-nb_validation_samples:]
    y_test = labels[-nb_validation_samples:]
    texts_test = texts[-nb_validation_samples:]

    return x_train, y_train, texts_train, x_test, y_test, texts_test, token_index



if prepare_data:
    # Kaggle IMDB dataset: https://www.kaggle.com/c/word2vec-nlp-tutorial/data
    # pd dataframe of size(300, 3), columns: id, first_author, appline, category
    df = pd.read_excel('20180419SampleText.xlsx', sheet_name='Sheet1').iloc[:300]
    category_dict = {'Pure Background': 1, 'Tool/Technique/Formula/Input': 2,
                     'Motivation for Research/Difference from Existing Inventions': 3, 
                     'Similar concept being patented, idea that can be used with invention, or potential use of invention': 4,
                     'Impossible to Tell': 5}
    if NUM_CATEGORY == 4: 
        category_dict = {'Pure Background': 1, 'Tool/Technique/Formula/Input': 2,
                     'Motivation for Research/Difference from Existing Inventions': 1, 
                     'Similar concept being patented, idea that can be used with invention, or potential use of invention': 3,
                     'Impossible to Tell': 4}
    # enumerate one-hot encoding
    for cate in category_dict: 
        df.loc[df[cate] == 1, 'category'] = category_dict[cate]
        
    df = df[['first_author', 'appline', 'category']]
    
    #train_len = df.appline.str.split().str.len()
    #train_len.describe()
    #count    300.000000
    #mean      91.640000
    #std       47.539778
    #min        8.000000
    #25%       45.000000
    #50%      106.000000
    #75%      133.000000
    #max      172.000000
    #Name: appline, dtype: float64
    
    # set MAX_SEQUENCE_LENGTH = 1000
    
    # option: prepare_train_val(train, train_size = 500)
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
            
    
if load_exp_space:
    #This is the parameter space to explore with hyperopt

    #Simply offers several discrete choices for numnber of hidden units and drop out rates for
    #a 2 or 3 layer MLP and also batch size
    
    #This can be expanded over other parameters and to sample from a distribution instead of a discrete choice
    #for # of units etc.
        
    space = {        
        'fsz1': hp.choice('fsz1', [2, 3, 4, 5]),
        'fsz2': hp.choice('fsz2', [2, 3, 4]),
        
        'num_filters': hp.choice('num_filters', [50, 100, 300]),
        
        'conv_l2': hp.choice('conv_l2', [0.001, 0.01, 0.05, 0.1]),
        'dense1_l2': hp.choice('dense1_l2', [0.001, 0.01, 0.05, 0.1]),
        'pred_l2': hp.choice('pred_l2', [0.001, 0.005, 0.01, 0.1]),
        
        'maxPooling': hp.choice('maxPooling', [2, 5, 10, 30]),
        
        'actv': hp.choice('actv', ['sigmoid','relu']),
             
        'dropout1': hp.uniform('dropout1', 0, 1),        
        'dropout2': hp.uniform('dropout2', 0, 1),
        
        'dense_units': hp.choice('dense_units', [50, 100, 300, 500]),
        
        'batch_size' : hp.choice('batch_size', [16, 32, 64, 128]),
    
        'optimizer': hp.choice('optimizer', ['rmsprop','adam', 'nadam']), 
        'patience': hp.choice('patience', [2, 5, 10, 20])
        }



#objective function for hyper tune      
def objective(params):
    # load embedding matrix to an embedding layer
    # outputs a 3D tensor of shape (samples, sequence_length, embedding_dim)
    embedding_layer = Embedding(len(token_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
  
    # trained on 101 reviews, validated on 99 reviews
    # dense layer immediate after concatenated convolutional layers
    l_conv = Conv1D(nb_filter=params['num_filters'],filter_length=params['fsz1'], kernel_regularizer=l2(params['conv_l2']))(embedded_sequences)
    # normalization decrease accuracy significantly
    #l_norm = BatchNormalization()(l_conv)
    l_actv = Activation(params['actv'])(l_conv)
    l_dropout = Dropout(params['dropout1'])(l_actv) 
    # (Zhang et al.) proposed that globalMaxPooling gives the best performance
    l_pool = MaxPooling1D(params['maxPooling'])(l_dropout)    
    # reduce the three-dimensional output to two dimensional for concatenation
    l_flat = Flatten()(l_pool)
        
    l_dense = Dense(params['dense_units'], W_regularizer=l2(params['dense1_l2']))(l_flat)
    #l_norm1 = BatchNormalization()(l_dense)
    l_actv1 = Activation(params['actv'])(l_dense)
    # this dropout layer reduce "loss" from test set, observed from plots
    l_dropout2 = Dropout(params['dropout2'])(l_actv1) 
    l_dense2 = Dense(NUM_CATEGORY, W_regularizer=l2(params['pred_l2']))(l_dropout2)
    # multi-class classification
    pred = Activation('softmax')(l_dense2)
    
    # tutorial on optimizer: http://ruder.io/optimizing-gradient-descent/index.html#rmsprop
    model = Model(inputs= sequence_input, outputs=pred)
    model.compile(loss='categorical_crossentropy',
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
        
    # select optimal stopping automatically
    # save improved model after each epoch
    # patience: number of epochs after observing no improvement
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=params['patience'], verbose=0),
        ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto') 
    ]
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
              batch_size=params['batch_size'], epochs = 100, callbacks=callbacks)
    score, acc = model.evaluate(x_test, y_test, batch_size=params['batch_size'])
    #minimizing' -val_acc' to attain the best set of parameters
    return {'loss': -acc, 'status': STATUS_OK, 'model': model }

# run hyper tune, minimize objective function
if hyper_tune:
    # to apply hyperopt, we need to specify:
    # 1. the objective function to minimize
    # 2. the space over which to search
    # 3. a trials database [optional]
    # 4.  the search algorithm to use [optional] (Bergstra et al.2013)
    trials = Trials()
    # func = objective(), minimize -acc
    # input parameters on trial: space{}
    # pass in a Trials() object so that we can retain all the assessed points during the search besides the 'best'one, access by Trials' attributes
    best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=10)
    print (best)
    print (trials.best_trial)


if construct_model_tuned:
    embedding_layer = Embedding(len(token_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True, name='embedding')
    
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32', name='input_x')
    embedded_sequences = embedding_layer(sequence_input)
    
    if NUM_CATEGORY == 4:
    #hyperopt tuned for 4 categories(combining 1&3 into 1)
        l_conv = Conv1D(nb_filter=100,filter_length=3, kernel_regularizer=l2(0.001))(embedded_sequences)
        #l_norm = BatchNormalization()(l_conv)
        l_actv = Activation('relu')(l_conv)
        
        l_dropout = Dropout(0.5)(l_actv) 
        l_pool = MaxPooling1D(5)(l_dropout)
        l_flat = Flatten()(l_pool)
        
        l_dense = Dense(50, W_regularizer=l2(0.05))(l_flat)
        #l_norm1 = BatchNormalization()(l_dense)
        l_actv1 = Activation('relu')(l_dense)
        l_dropout2 = Dropout(0.2)(l_actv1) 
    
        l_dense2 = Dense(4, W_regularizer=l2(0.05), name='dense2')(l_dropout2)
        pred = Activation('softmax')(l_dense2)
        
        sys.path.append('/4cate_models')
        
        model = Model(inputs= sequence_input, outputs=pred)    
        model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])    
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=0),
            ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto') 
            ]
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  batch_size=32, epochs = 100, callbacks=callbacks)
        score, acc = model.evaluate(x_test, y_test, batch_size=32)
    #############################################   
    # Outputs for 4 categories:
        
    #Output for 4 categories: 
    #Test acc score: 0.818181818182
    #Test Confusion Matrix: 
    # [[42  7  0  1]
    # [ 5 15  0  0]
    # [ 3  0  0  0]
    # [ 0  2  0 24]]
    #Test F1 score: 0.615748663102
    #############################################
    
#    Output for 4 categories: 
#    Test acc score: 0.737373737374
#    Test Confusion Matrix: 
#     [[39  5  0  3]
#     [10 16  0  1]
#     [ 6  1  0  0]
#     [ 0  0  0 18]]
#    Test F1 score: 0.579441776711
   
    if NUM_CATEGORY == 5:
#         hyperparameters chosen by hyperopt
#         for 5 categories
#         filter size does not affect much
        l_conv = Conv1D(nb_filter=100,filter_length=3, kernel_regularizer=l2(0.001))(embedded_sequences)
        #l_norm = BatchNormalization()(l_conv)
        l_actv = Activation('relu')(l_conv)
        
        l_dropout = Dropout(0.88)(l_actv) 
        l_pool = MaxPooling1D(10)(l_dropout)
        l_flat = Flatten()(l_pool)
        
        l_dense = Dense(100, W_regularizer=l2(0.05))(l_flat)
        #l_norm1 = BatchNormalization()(l_dense)
        l_actv1 = Activation('relu')(l_dense)
        l_dropout2 = Dropout(0.5)(l_actv1) 
    
        l_dense2 = Dense(5, W_regularizer=l2(0.1), name='dense2')(l_dropout2)
        pred = Activation('softmax')(l_dense2)
        
        sys.path.append('/5cate_models')
        
        model = Model(inputs= sequence_input, outputs=pred)   
        model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=0),
            ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto') 
            ]
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  batch_size=16, epochs = 100, callbacks=callbacks)
        score, acc = model.evaluate(x_test, y_test, batch_size=16)
        
    ###########################################
    # Outputs for 5 categories:    
    # if using the same model as in IMDB without change in hyperparameters
    # test_acc = 28%, train_acc = 90% in 100 epochs(with early stopping)
    # this is a bit better than random guess
    # it seems the main reason involves with BatchNormalization() layer    
    
    # hyperopt tuned:
    # considering randomness in simulations, optimal accuracy varies within 5%(62%-67%)
    
#    Output for 5 categories: 
#    Test acc score: 0.626262626263
#    Test Confusion Matrix: 
#     [[22  0  9  0  0]
#     [ 9 15  0  0  0]
#     [11  1  5  0  1]
#     [ 3  0  3  0  0]
#     [ 0  0  0  0 20]]
#    Test F1 score: 0.518054282047
     
    # load best model
    # use model-045.h5 for 5-category; model-035.h5 for 4-category
    model = load_model('model-045.h5')   
    # y_pred: prob. for each class    
    print ("Output for {0} categories: ".format(NUM_CATEGORY))
    y_pred = model.predict(x_test)
    print("Test acc score: {0}".format(np.mean(y_test.argmax(axis=1) == y_pred.argmax(axis=1))))
    mat_test = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print("Test Confusion Matrix: \n {0}".format(mat_test))
    print("Test F1 score: {0}".format(f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro')))
    
    
# plot accuracy rate and loss        
if plot:
    # summarize history for accuracy
#    print(history.summary())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
if check_prediction:
    # find false positives and false negatives
    # predict on test set then save incorrect predictions, note test set contains 99 records
    # y_pred: prob. for each class    
    test_predict = y_pred.argmax(axis=1) # find class with highest prob.
    # combine true label, predicted label and texts into one dataframe 
    compare_test = pd.DataFrame({'y_test':y_test.argmax(axis=1), 
                                'pred_test':test_predict.reshape((test_predict.shape[0],))})
    df_text_test = pd.Series(texts_test, name='text')
    df_label_test = pd.Series(y_test.argmax(axis=1), name='label')
    df_test = pd.concat([df_text_test, df_label_test], axis=1)
    df_test['pred'] = test_predict.reshape((test_predict.shape[0],))
    false_pred = df_test.loc[(df_test['label'] <>  df_test['pred'])]
    
    # save records for all preditions and false predictions separately
    df_all = df_test.copy()
    pred_prob_test = pd.DataFrame(data=y_pred,
              index = df_test.index.tolist(),
              columns=np.arange(0, NUM_CATEGORY))
    df_all = df_all.merge(pred_prob_test, left_index = True, right_index=True)
    false_pred = df_all.loc[(df_all['label'] <> df_all['pred'])]
    
    # target file name can be changed
    writer = pd.ExcelWriter('CNN_compare_pred5.xlsx')
    df_all.to_excel(writer,'all')
    false_pred.to_excel(writer, 'wrong')
    writer.save()
    
if interpret:    
    with DeepExplain(session=K.get_session()) as de:  
        # Need to reconstruct the graph in DeepExplain context, using the same weights.
        # explain(method_name, target_tensor, input_tensor, samples, ...args)
        # samples: np-array required
        
        # model-036 for 4-category classification, model-045 for 5-category classification
        model = load_model('model-045.h5')
        
        # reference to tensors
        input_tensor = model.get_layer("input_x").input
        embedding = model.get_layer("embedding").output
        pre_softmax = model.get_layer("dense2").output
        
        # choose sample, range from 0-99, we can find 10 samples in sig_visualization dumped in json files
        s_index = 88
        
        # for example:
        # this line is classified wrongly, labeled as 3, predicted as 1
        # context:
        #p id  p 0022  num  0025    x201c Electrical Injection and Detection of 
        #Spin Polarized Electrons in Silicon through an Fe sub 3  sub Si Si Schottky 
        #Tunnel Barrier  x201d , Y  Ando, K  Hamaya, K  Kasahara, Y  Kishi, K  Ueda, K  
        #Sawano, T  Sadoh, and M  Miyano,  i Applied Physics Letters  i , 
        #vol  94p  182105, (2009)
        x_interpret = x_test[s_index:s_index+1]        
        # perform embedding lookup
        # use Keras directly? 
        #embedding_out = sess.run(embedding, {input_tensor: x_interpret})
        #eModel = Model(inputs=input_tensor, outputs=embedding)
        #embedding_out = eModel(input_tensor)
        get_embedding_output = K.function([input_tensor],[embedding])
        embedding_out = get_embedding_output([x_interpret])[0]        
        
        # target the output of the last dense layer (pre-softmax)
        # To do so, create a new model sharing the same layers untill the last dense (index -2)
        fModel = Model(inputs=input_tensor, outputs = pre_softmax)
        target_tensor = fModel(input_tensor)
        
        # to target a specific neuron(class), we apply a binary map
        ys = [1, 0, 0, 0, 0]
        
        # input top layer after word embedding, bottom layer before softmax
        attributions = de.explain('elrp', pre_softmax*ys, embedding, embedding_out)

        # prepare texts
        text_dec = []
        for x in np.nditer(x_interpret):
            if x in token_index.values():
                text_dec.append(token_index.keys()[token_index.values().index(x)])
                
        # generate attribute significance
        p_gens = np.sum(attributions,axis=2).tolist()
        new_pgens = []
        for gen in p_gens[0]:
            new_pgens.append([gen])
            
        abstract_str = texts_test[s_index]
        true_label = y_test[s_index:s_index+1].argmax(axis=1).tolist()
        pred_prob = y_pred[s_index:s_index+1].tolist()
        pred_label = y_pred[s_index:s_index+1].argmax(axis=1).tolist()
        
        format_data = {'abstract_str': abstract_str, 'decoded_lst': text_dec, 
                       'p_gens': new_pgens, 'true_label': true_label, 'pred_label': pred_label,
                       'pred_prob': pred_prob}
        
    
if prepare_json:
    # save each sample into a different json file
    with open('/Users/lizhuoran/Desktop/UOFT/research/patents/sig_visualize/sig_vis_data9.json', 'w') as outfile:
        json.dump(format_data, outfile)
        
