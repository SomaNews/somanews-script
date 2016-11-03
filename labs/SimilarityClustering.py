import numpy as np
import pandas as pd
from konlpy.tag import Mecab
import math
import hanja
import re
import string
import operator
import random
import matplotlib.pyplot as plt
import itertools
import cnouns as cn
import check_utils as cu
import deep_utils as du
from sklearn.metrics import adjusted_rand_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_samples, silhouette_score
from time import time
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import datetime
from sklearn.decomposition import PCA
from gensim import models
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
import gensim.models.doc2vec
from collections import OrderedDict
from gensim.models.doc2vec import LabeledSentence

import multiprocessing
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
import cPickle as pickle
from spherecluster import SphericalKMeans
cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

from random import shuffle

from collections import namedtuple

Articles = namedtuple('Articles', 'words tags split')

class SimilarityClustering:
    # df_
    
    # times_
    
    # alldocs_
    # models_by_name_
    # dm_
    # centers_
    # scores_
    # countby_
    # topics_
    
    def __init__(self):
        self.times_ = {
            "preprocessing": {},
            "learning": {},
            "clustering": {},
            "topic": {}
        }
        
        
    def reset(self, train_df):
        self.df_ = train_df[:]
        
        
    def tokenize(self, tokenizer=cn.tokenize):
        self.times_["preprocessing"]["start"] = time()
        self.alldocs_ = []
        size = len(self.df_) / 4
        
        ## tokenize
        print("Tokenizing.....")
        self.df_['target_str'] = [tokenizer(row.title + " " + row.content) for idx, row in self.df_.iterrows()]
        print("Complete to tokenize.")
        
        ## make docs
        for idx, row in self.df_.iterrows():
            tokens = row['target_str'].split(' ')
            words = tokens[0:]
            tags = [idx]
            tmp = idx//size % 4
            split = ['train','test','extra','extra'][tmp]  # 25k train, 25k test, 25k extra
            self.alldocs_.append(Articles(words, tags, split))
            
        self.times_["preprocessing"]["end"] = time()
    
    
    def doc_train(self, w2v_path="../datastore/sejongcorpus_w2v2.p", alpha=0.025, min_alpha=0.001, passes=20):
        self.times_["learning"]["start"] = time()
        
        ## make doc2vec models
        simple_models = [
            # PV-DM Distributed Momory Model of PV
            # w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
            Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
            # PV-DBOW Distributed Bag of Words version of PV
            Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
            # PV-DM w/average
            Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
        ]
        
        ## load word2vec model
        print("Loading word2vec model.....")
        simple_models[0].load_word2vec_format(w2v_path)
        print("Complete to load.")
        
        ## reset doc2vec model
        simple_models[0].build_vocab(self.alldocs_)
        print simple_models[0]
        for model in simple_models[1:]:
            model.reset_from(simple_models[0])
            print(model)

        self.models_by_name_ = OrderedDict((str(model), model) for model in simple_models)
        
        ## training
        doc_list = self.alldocs_[:]
        alpha_delta = (alpha - min_alpha) / passes

        for epoch in range(passes):
            shuffle(doc_list)  # shuffling gets best results

            for name, train_model in self.models_by_name_.items():
                train_model.alpha, train_model.min_alpha = alpha, alpha
                train_model.train(doc_list)
                print("%i passes : %s" % (epoch + 1, name))

            print('completed pass %i at alpha %f' % (epoch + 1, alpha))
            alpha -= alpha_delta
    
        self.concat_vec()
        self.times_["learning"]["end"] = time()
        
    
    def select_model(self, model_name='Doc2Vec(dm/c,d100,n5,w5,mc2,t8)'):
        self.dm_ = self.models_by_name_[model_name]
        
        
    def clustering(self, threshold=0.8):
        self.times_["clustering"]["start"] = time()
        print("Similarity clustering.....")
        self.centers_ = du.similarity_clustering(self.df_, self.dm_.docvecs, threshold)
        du.calc_similarity(self.df_, self.dm_.docvecs, self.centers_)
        print("Complete to similarity clustering.")
        self.times_["clustering"]["end"] = time()
        
        
    def iner_score(self, cnt_threshold=10):
        self.scores_ = du.similarity_iner_score(self.centers_, self.df_, self.dm_.docvecs)
        size_1 = self.scores_[self.scores_.cnt==1]
        self.countby_ = self.scores_[self.scores_.cnt>cnt_threshold]
        print "total:", len(self.scores_), ", size_1:",len(size_1), ", countby:", len(self.countby_)
        ss = self.countby_.sum(axis=0)
        print "distance:", ss['distance'] * 100
        print "variance:", ss['variance']
        print "similarity:", (ss['similarity'] * 100)/len(self.countby_)
        print "cohesion:", ss['cohesion']
        
        
    def get_all_topics(self, get_topic_func=du.get_all_topics):
        self.times_["topic"]["start"] = time()
        print("Get topics.....")
        self.topics_ = get_topic_func(self.df_, self.countby_.cluster.tolist())
        print("Complete to get topics.")
        self.times_["topic"]["end"] = time()
        
        
    def calc_elapsed(self):
        for key, value in self.times_.iteritems():
            value["elapsed"]= value["end"] - value["start"]
            
            
    def save(self, path, prefix):
        self.models_by_name_['Doc2Vec(dm/c,d100,n5,w5,mc2,t8)'].save("%s/%sd2v-dmc.p" % (path, prefix))
        self.models_by_name_['Doc2Vec(dbow,d100,n5,mc2,t8)'].save("%s/%sd2v-dbow.p" % (path, prefix))
        self.models_by_name_['Doc2Vec(dm/m,d100,n5,w10,mc2,t8)'].save("%s/%sd2v-dmm.p" % (path, prefix))

        self.df_.to_pickle("%s/%sdf.p" % (path, prefix))
        
        pickle.dump(self.centers_, open("%s/%scenters.p" % (path, prefix), "wb"))
        pickle.dump(self.topics_, open("%s/%stopics.p" % (path, prefix), "wb"))
        pickle.dump(self.times_, open("%s/%stimes.p" % (path, prefix), "wb"))

        print("Complete to save model.")
        
    def concat_vec(self):
        self.models_by_name_['dbow+dmm'] = ConcatenatedDoc2Vec([self.models_by_name_['Doc2Vec(dbow,d100,n5,mc2,t8)'], self.models_by_name_['Doc2Vec(dm/m,d100,n5,w10,mc2,t8)']])
        self.models_by_name_['dbow+dmc'] = ConcatenatedDoc2Vec([self.models_by_name_['Doc2Vec(dbow,d100,n5,mc2,t8)'], self.models_by_name_['Doc2Vec(dm/c,d100,n5,w5,mc2,t8)']])

        
    def s_load(path, prefix):
        sc = SimilarityClustering()
        
        sc.models_by_name_ = OrderedDict()
        
        sc.models_by_name_['Doc2Vec(dm/c,d100,n5,w5,mc2,t8)'] = Doc2Vec.load("%s/%sd2v-dmc.p" % (path, prefix))
        sc.models_by_name_['Doc2Vec(dbow,d100,n5,mc2,t8)'] = Doc2Vec.load("%s/%sd2v-dbow.p" % (path, prefix))
        sc.models_by_name_['Doc2Vec(dm/m,d100,n5,w10,mc2,t8)'] = Doc2Vec.load("%s/%sd2v-dmm.p" % (path, prefix))

        sc.concat_vec()
        sc.select_model()
        
        sc.df_ = pd.read_pickle("%s/%sdf.p" % (path, prefix))
        
        sc.centers_ = pickle.load(open("%s/%scenters.p" % (path, prefix), "rb"))
        sc.topics_ = pickle.load(open("%s/%stopics.p" % (path, prefix), "rb"))
        sc.times_ = pickle.load(open("%s/%stimes.p" % (path, prefix), "rb"))

        sc.iner_score()
        
        return sc
    load=staticmethod(s_load)
    
    def print_clusters(size, top=10):
        for idx, row in self.countby_.sort_values('cohesion', ascending=False)[:top].iterrows():
            print du.test_print(row.cluster, self.df_, self.dm_.docvecs, self.centers_, self.topics_, self.countby_)
            print "\n------------------------------------------------------------\n"
            
            
    def train(self, train_df, path, prefix,
              tokenizer=cn.tokenize, 
              w2v_path="../datastore/sejongcorpus_w2v2.p", alpha=0.025, min_alpha=0.001, passes=20,
              model_name='Doc2Vec(dm/c,d100,n5,w5,mc2,t8)', 
              threshold=0.8, 
              cnt_threshold=10, 
              get_topic_func=du.get_all_topics
             ):
        self.reset(train_df[:])
        self.tokenize(cn.tokenize)
        self.doc_train(w2v_path, alpha, min_alpha, passes)
        self.select_model(model_name)
        self.clustering(threshold)
        self.iner_score(cnt_threshold)
        self.get_all_topics(get_topic_func)
        self.calc_elapsed()
        self.save(path, prefix)
            
            
#     def save_to_db(db):