#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:41:38 2018

@author: lizhuoran
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 19:07:28 2018

@author: lizhuoran
"""
import pandas as pd
import numpy as np
import re
import nltk
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

from itertools import chain, repeat, islice

load_data = False # options to choose # of categories, set constant "NUM_CATEGORY" below
load_embedding = True
SVM = False
tfIdf = False
SVM_embedding = False
output_false_pred = False

NUM_CATEGORY = 4
#MAX_NB_WORDS = 20000
VALIDATION_SPLIT = 0.333


if load_data: 
    # read top 300 rows from the spreadsheet into a pandas dataframe 
    # with columns: [first_author, appline, Pure Background, Tool..., Motivation..., concept..., not to tell..]
    df = pd.read_excel('20180419SampleText.xlsx', sheet_name='Sheet1').iloc[:300]
    # enumerate categories
    category_dict = {'Pure Background': 1, 'Tool/Technique/Formula/Input': 2,
                     'Motivation for Research/Difference from Existing Inventions': 3, 
                     'Similar concept being patented, idea that can be used with invention, or potential use of invention': 4,
                     'Impossible to Tell': 5}
    if NUM_CATEGORY == 4: 
        category_dict['Motivation for Research/Difference from Existing Inventions'] = 1
        
    for cate in category_dict: 
        df.loc[df[cate] == 1, 'category'] = category_dict[cate]
        
    df = df[['first_author', 'appline', 'category']]
    
    # clean text
    df['cleaned_appline'] = df.appline.replace(r"[^A-Za-z0-9(),!?\'\`]"," ", regex=True, inplace=False)
    df.cleaned_appline.replace(r"\\", "", regex=True, inplace=True)
    df.cleaned_appline.replace(r"\"", "", regex=True, inplace=True)

    # split into training and test sets
    # pandas do the sampling randomly
    df_test = df.sample(frac=VALIDATION_SPLIT)
    df_train = df[~df.index.isin(df_test.index)]
    
    texts_train = df_train.appline
    texts_test = df_test.appline
    
if load_embedding: 
    # Load the 50d Glove file 
    # Convert file format to word2vec format    
    glove_input_file = 'Glove/glove.6B.50d.txt'
    word2vec_output_file = 'word2vecGloveswitch.txt'
    glove2word2vec(glove_input_file, word2vec_output_file)
    
    # load the Stanford GloVe model converted to word2vec
    word2vecGloveswitch = 'word2vecGloveswitch.txt'
    # Model will take words and spit out word embedding
    model = KeyedVectors.load_word2vec_format(word2vecGloveswitch, binary=False)
    
    
    # Make text lower case and then tokenize, for test/train sets
    df_train_token = df_train.copy()
    df_test_token = df_test.copy()
    df_train_token['appline']=df_train_token.appline.str.lower()
    df_train_token['tokenized'] = df_train_token.apply(lambda row: nltk.word_tokenize(row['appline']), axis=1)
    tokens_train=df_train_token.tokenized.tolist()
    
    df_train_token['cleaned_appline']=df_train_token.cleaned_appline.str.lower()
    df_train_token['cleaned_tokenized'] = df_train_token.apply(lambda row: nltk.word_tokenize(row['cleaned_appline']), axis=1)
    cleaned_tokens_train = df_train_token.cleaned_tokenized.tolist()
    
    df_test_token['appline']=df_test_token.appline.str.lower()
    df_test_token['tokenized'] = df_test_token.apply(lambda row: nltk.word_tokenize(row['appline']), axis=1)
    tokens_test = df_test_token.tokenized.tolist()
    
    df_test_token['cleaned_appline']=df_test_token.cleaned_appline.str.lower()
    df_test_token['cleaned_tokenized'] = df_test_token.apply(lambda row: nltk.word_tokenize(row['cleaned_appline']), axis=1)
    cleaned_tokens_test = df_test_token.cleaned_tokenized.tolist()
    
            
    vocab = model.vocab.keys()
    vectors_train=[]
    vectors_test=[]
    for r in range(len(tokens_train)):
        new_temp=[0]*50
        count=0
        for w in range(len(tokens_train[r])):
            if tokens_train[r][w] in vocab:
                count=count+1
                new_temp=new_temp+model[tokens_train[r][w]]
        vectors_train.append(new_temp/count)
    for r in range(len(tokens_test)):
        new_temp=[0]*50
        count=0
        for w in range(len(tokens_test[r])):
            if tokens_test[r][w] in vocab:
                count=count+1
                new_temp=new_temp+model[tokens_test[r][w]]
        vectors_test.append(new_temp/count)
        
    # apply word embedding on preprocessed texts
    vectors_train2=[]
    vectors_test2=[]
    for r in range(len(cleaned_tokens_train)):
        new_temp=[0]*50
        count=0
        for w in range(len(cleaned_tokens_train[r])):
            if cleaned_tokens_train[r][w] in vocab:
                count=count+1
                new_temp=new_temp+model[cleaned_tokens_train[r][w]]
        vectors_train2.append(new_temp/count)
    for r in range(len(cleaned_tokens_test)):
        new_temp=[0]*50
        count=0
        for w in range(len(cleaned_tokens_test[r])):
            if cleaned_tokens_test[r][w] in vocab:
                count=count+1
                new_temp=new_temp+model[cleaned_tokens_test[r][w]]
        vectors_test2.append(new_temp/count)
    
# SVM without word embedding
if SVM:
    text_clf = Pipeline([('vect', CountVectorizer()), 
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, 
                                               random_state=42, max_iter=5, tol=None))])
    text_clf.fit(df_train.appline, df_train.category) 
    
    label_test=text_clf.predict(df_test.appline)
    label_train=text_clf.predict(df_train.appline)
    # confusion matrix
    mat_test = confusion_matrix(df_test.category, label_test)
    mat_train = confusion_matrix(df_train.category, label_train)

    print("SVM \n")    
    print("Test Confusion Matrix: \n {0}".format(mat_test))   
    print("Test acc score: {0}".format(np.mean(label_test == df_test.category)))
    print("Test F1 score: {0}".format(f1_score(df_test.category, label_test, average='macro')))
    print("Training Confusion Matrix: \n {0}".format(mat_train))
    print("Training acc score: {0}".format(np.mean(label_train == df_train.category)))
    print("Training F1 score: {0}".format(f1_score(df_train.category, label_train, average='macro')))
    
    # false prediction df
    df_SVM = df_test.copy()
    df_SVM['SVM'] = label_test
    false_pred_SVM = df_SVM.loc[(df_SVM['category'] <> df_SVM['SVM'])]
    
#    Outputs: 
#
#    Test Confusion Matrix: 
#     [[12  2  3  2  3]
#     [ 5 17  1  1  2]
#     [ 8  1  8  2  3]
#     [ 1  1  2  1  0]
#     [ 1  1  0  0 23]]
#    Test acc score: 0.61
#    Test F1 score: 0.529164089878
#    Training Confusion Matrix: 
#     [[60  0  0  0  1]
#     [ 1 40  0  0  0]
#     [ 0  0 33  0  0]
#     [ 1  0  0 11  0]
#     [ 1  0  0  0 52]]
#    Training acc score: 0.98
#    Training F1 score: 0.978610014215
    
    
if tfIdf: 
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, 
                                               random_state=42, max_iter=3, tol=None))])
    text_clf.fit(df_train.appline, df_train.category) 
    
    label_test=text_clf.predict(df_test.appline)
    label_train=text_clf.predict(df_train.appline)
    # confusion matrix
    mat_test = confusion_matrix(df_test.category, label_test)
    mat_train = confusion_matrix(df_train.category, label_train)

    print("tf-idf \n")   
    print("Test Confusion Matrix: \n {0}".format(mat_test))   
    print("Test acc score: {0}".format(np.mean(label_test == df_test.category)))
    print("Test F1 score: {0}".format(f1_score(df_test.category, label_test, average='macro')))
    print("Training Confusion Matrix: \n {0}".format(mat_train))
    print("Training acc score: {0}".format(np.mean(label_train == df_train.category)))
    print("Training F1 score: {0}".format(f1_score(df_train.category, label_train, average='macro')))
    
    # false prediction df
    df_tfIdf = df_test.copy()
    df_tfIdf['tf-idf'] = label_test
    false_pred_tfIdf = df_tfIdf.loc[(df_tfIdf['category'] <> df_tfIdf['tf-idf'])]
    
#    Outputs: 
#        
#    Test Confusion Matrix: 
#     [[16  1  4  1  0]
#     [ 8 15  2  0  1]
#     [13  1  7  0  1]
#     [ 2  0  2  1  0]
#     [ 2  0  0  0 23]]
#    Test acc score: 0.62
#    Test F1 score: 0.557940718127
#    Training Confusion Matrix: 
#     [[61  0  0  0  0]
#     [ 0 41  0  0  0]
#     [ 0  0 33  0  0]
#     [ 1  0  0 11  0]
#     [ 0  0  0  0 53]]
#    Training acc score: 0.995
#    Training F1 score: 0.989678331566
    
    
if SVM_embedding:
    # Fit embedded paragraphs to SVM
    text_clf=SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter = 200, random_state=42)
    text_clf.fit(vectors_train2,df_train.category)
    label_test = text_clf.predict(vectors_test2)
    label_train = text_clf.predict(vectors_train2)
    mat_test = confusion_matrix(df_test.category, label_test)
    mat_train = confusion_matrix(df_train.category, label_train)
    
    print("SVM w/ word embedding \n")
    print("Test Confusion Matrix: \n {0}".format(mat_test))  
    print("Test acc score: {0}".format(np.mean(label_test == df_test.category)))
    print("Test F1 score: {0}".format(f1_score(df_test.category, label_test, average='macro')))
    print("Training Confusion Matrix: \n {0}".format(mat_train))
    print("Training acc score: {0}".format(np.mean(label_train == df_train.category)))
    print("Training F1 score: {0}".format(f1_score(df_train.category, label_train, average='macro')))
    
    # false prediction df
    df_SVM_embedding = df_test.copy()
    df_SVM_embedding['SVM_embedding'] = label_test
    false_pred_SVM_embedding = df_SVM_embedding.loc[(df_SVM_embedding['category'] <> df_SVM_embedding['SVM_embedding'])]
    
#    SVM w/ word embedding (cleaned texts)
#    
#    Test Confusion Matrix: 
#     [[15  3  2  0  2]
#     [ 9 14  2  0  1]
#     [ 5  5 11  0  1]
#     [ 3  1  1  0  0]
#     [ 2  0  0  0 23]]
#    Test acc score: 0.63
#    Test F1 score: 0.514141122036
    
#    Training Confusion Matrix: 
#     [[48  3  8  0  2]
#     [ 5 33  1  0  2]
#     [11  1 21  0  0]
#     [ 6  2  3  0  1]
#     [ 0  0  0  0 53]]
#    Training acc score: 0.775
#    Training F1 score: 0.62982860376
    
if output_false_pred:
    false_pred_all = pd.concat([false_pred_SVM.appline, false_pred_tfIdf.appline, false_pred_SVM_embedding.appline]).drop_duplicates()   
    false_pred_all = false_pred_all.to_frame().merge(df[['first_author', 'category']], left_index=True, right_index=True, how='left')
    false_pred_all = false_pred_all.merge(false_pred_SVM[['SVM']], left_index=True, right_index=True, how='left')
    false_pred_all = false_pred_all.merge(false_pred_tfIdf[['tf-idf']], left_index=True, right_index=True, how='left')
    false_pred_all = false_pred_all.merge(false_pred_SVM_embedding[['SVM_embedding']], left_index=True, right_index=True, how='left')

    writer = pd.ExcelWriter('compare_pred.xlsx')
    false_pred_all.to_excel(writer, '5_categories')
    writer.save()

    

    

## grid search
#parameters = {'vect__ngram_range': [(1, 1), (1, 5)],
#              'tfidf__use_idf': (True, False),
#              'clf__alpha': (1e-2, 1e-3)}
## n_job = -1 will detect # of cores installed and uses them all
#gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
#gs_clf = gs_clf.fit(df_train.text, df_train.label)
##sample_train = df_train.sample(frac=0.1)
##sample_test = df_val.sample(frac=0.1)
##gs_clf = gs_clf.fit(sample_train.text, sample_train.label)
#gs_clf.best_score_ 
##0.724736
#CV_result = gs_clf.cv_results_
##optimized para ngram_range = (1, 1); use_idf=True; clf_alpha=1e-3






