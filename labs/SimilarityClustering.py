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
import articles_data
import datetime 

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
        self.topics_ = {}
        self.times_ = {
            "preprocessing": {},
            "learning": {},
            "clustering": {},
            "topic": {}
        }
        
        
    def reset(self, train_df):
        self.df_ = train_df[:]
        
        
    def tokenize(self, tokenizer=cn.tokenizer):
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
            
        self.df_.drop(['target_str'], axis=1, inplace=True)
        self.times_["preprocessing"]["end"] = time()
    
    
    def doc_train(self, w2v_path, alpha=0.025, min_alpha=0.001, passes=20):
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
        
        
    def clustering(self, threshold=0.8, repeat=5):
        self.times_["clustering"]["start"] = time()
        print("Similarity clustering.....")
        self.centers_, clusters = du.similarity_clustering(self.df_, self.dm_.docvecs, threshold, repeat)
#        self.centers_, clusters = du.similarity_clustering_time(self.df_, self.dm_.docvecs, threshold, repeat)
        self.df_['cluster'] = clusters
        du.calc_similarity(self.df_, self.dm_.docvecs, self.centers_)
        print("Complete to similarity clustering.")
        self.times_["clustering"]["end"] = time()
        
        
    def clustering_cate(self, threshold=0.8, repeat=5):
        self.times_["clustering"]["start"] = time()
        print("Similarity cate clustering.....")
        cates = articles_data.get_target_cate()
        centers = {}
        clusters = []
        for cate in cates:
            print("\nClustering category : %s"%cate)
            center, cluster = du.similarity_clustering(self.df_[self.df_.cate==cate], self.dm_.docvecs, threshold, repeat)
#            center, cluster = du.similarity_clustering_time(self.df_[self.df_.cate==cate], self.dm_.docvecs, threshold, repeat)
            centers.update(center)
            clusters.append(cluster)    
        self.centers_ = centers
        self.df_['cluster'] = pd.concat(clusters, axis=0)
        du.calc_similarity(self.df_, self.dm_.docvecs, self.centers_)
        print("Complete to similarity clustering.")
        self.times_["clustering"]["end"] = time()
        
        
    def iner_score(self, threshold=0.8, cnt_threshold=10):
        self.scores_, self.countby_ = du.iner_socre(self.centers_, self.df_, self.dm_.docvecs, threshold, cnt_threshold)
        
        
    def get_all_topics(self, get_topic_func=du.get_all_topics):
        self.times_["topic"]["start"] = time()
        print("Get topics.....")
        self.topics_ = get_topic_func(self.df_, self.countby_.cluster.tolist())
        print("Complete to get topics.")
        self.times_["topic"]["end"] = time()
        
        
    def calc_elapsed(self):
        for key, value in self.times_.iteritems():
            if 'end' in value:
                value["elapsed"]= value["end"] - value["start"]
            
            
    def save(self, path, prefix):
        self.models_by_name_['Doc2Vec(dm/c,d100,n5,w5,mc2,t8)'].save("%s/%s_d2v-dmc.p" % (path, prefix))
        self.models_by_name_['Doc2Vec(dbow,d100,n5,mc2,t8)'].save("%s/%s_d2v-dbow.p" % (path, prefix))
        self.models_by_name_['Doc2Vec(dm/m,d100,n5,w10,mc2,t8)'].save("%s/%s_d2v-dmm.p" % (path, prefix))

        self.df_.to_pickle("%s/%s_df.p" % (path, prefix))
        
        pickle.dump(self.centers_, open("%s/%s_centers.p" % (path, prefix), "wb"))
#        pickle.dump(self.topics_, open("%s/%s_topics.p" % (path, prefix), "wb")) # TODO
        pickle.dump(self.times_, open("%s/%s_times.p" % (path, prefix), "wb"))

        print("Complete to save model.")
        
    def concat_vec(self):
        self.models_by_name_['dbow+dmm'] = ConcatenatedDoc2Vec([self.models_by_name_['Doc2Vec(dbow,d100,n5,mc2,t8)'], self.models_by_name_['Doc2Vec(dm/m,d100,n5,w10,mc2,t8)']])
        self.models_by_name_['dbow+dmc'] = ConcatenatedDoc2Vec([self.models_by_name_['Doc2Vec(dbow,d100,n5,mc2,t8)'], self.models_by_name_['Doc2Vec(dm/c,d100,n5,w5,mc2,t8)']])

    def d2v_save(self, path, prefix):
        self.models_by_name_['Doc2Vec(dm/c,d100,n5,w5,mc2,t8)'].save("%s/%s_d2v-dmc.p" % (path, prefix))
        self.models_by_name_['Doc2Vec(dbow,d100,n5,mc2,t8)'].save("%s/%s_d2v-dbow.p" % (path, prefix))
        self.models_by_name_['Doc2Vec(dm/m,d100,n5,w10,mc2,t8)'].save("%s/%s_d2v-dmm.p" % (path, prefix))
        
    def s_d2v_load(path, prefix):
        sc = SimilarityClustering()
        
        sc.models_by_name_ = OrderedDict()
        
        sc.models_by_name_['Doc2Vec(dm/c,d100,n5,w5,mc2,t8)'] = Doc2Vec.load("%s/%s_d2v-dmc.p" % (path, prefix))
        sc.models_by_name_['Doc2Vec(dbow,d100,n5,mc2,t8)'] = Doc2Vec.load("%s/%s_d2v-dbow.p" % (path, prefix))
        sc.models_by_name_['Doc2Vec(dm/m,d100,n5,w10,mc2,t8)'] = Doc2Vec.load("%s/%s_d2v-dmm.p" % (path, prefix))
        
        return sc
    d2v_load=staticmethod(s_d2v_load)
    
    def s_load(path, prefix, threshold=0.8, cnt_threshold=10, only_d2v=False):
        sc = SimilarityClustering()
        
        sc.models_by_name_ = OrderedDict()
        
        sc.models_by_name_['Doc2Vec(dm/c,d100,n5,w5,mc2,t8)'] = Doc2Vec.load("%s/%s_d2v-dmc.p" % (path, prefix))
        sc.models_by_name_['Doc2Vec(dbow,d100,n5,mc2,t8)'] = Doc2Vec.load("%s/%s_d2v-dbow.p" % (path, prefix))
        sc.models_by_name_['Doc2Vec(dm/m,d100,n5,w10,mc2,t8)'] = Doc2Vec.load("%s/%s_d2v-dmm.p" % (path, prefix))

        sc.concat_vec()
        sc.select_model()
        
        sc.df_ = pd.read_pickle("%s/%s_df.p" % (path, prefix))
        
        if not(only_d2v):
            sc.centers_ = pickle.load(open("%s/%s_centers.p" % (path, prefix), "rb"))
#            sc.topics_ = pickle.load(open("%s/%s_topics.p" % (path, prefix), "rb")) # TODO
            sc.times_ = pickle.load(open("%s/%s_times.p" % (path, prefix), "rb"))

            sc.iner_score(threshold, cnt_threshold)
        
        return sc
    load=staticmethod(s_load)
    
    def print_clusters(self, sortby='cohesion', top=10, threshold=0.8, diff_threshold=0.01):
        cnt = 0
        for idx, row in self.countby_.sort_values(sortby, ascending=False)[:top].iterrows():
            cnt = cnt + 1
            print(cnt)
            self.print_cluster(row.cluster, threshold, diff_threshold)
#            print du.test_print(row.cluster, self.df_, self.dm_.docvecs, self.centers_, self.topics_, self.countby_, threshold, diff_threshold)
#            print("------------------------------------------------------------")
            
    
    def print_cluster(self, cluster, threshold=0.8, diff_threshold=0.01):
        print du.test_print(cluster, self.df_, self.dm_.docvecs, self.centers_, self.topics_, self.countby_, threshold, diff_threshold)
        print("------------------------------------------------------------")
            
            
    def print_topics(self, sortby='cohesion', top=10):
        cnt = 0
        for idx, row in self.countby_.sort_values(sortby, ascending=False)[:top].iterrows():
            cnt = cnt + 1
            print(cnt)
            du.topic_print(self.topics_[row.cluster])
            print("------------------------------------------------------------")
            
            
    def print_centers(self, sortby='cohesion', top=10):
        cnt = 0
        for idx, row in self.countby_.sort_values(sortby, ascending=False)[:top].iterrows():
            cnt = cnt + 1
            print cnt,  self.df_.loc[self.dm_.docvecs.most_similar([self.centers_[row.cluster]])[0][0]].title
            
    def getMainArticle(self, cluster):
        return self.df_.loc[self.dm_.docvecs.most_similar([self.centers_[cluster]])[0][0]]
            
    def train(self, typ, w2v_path, train_df, path, prefix,
              tokenizer=cn.tokenizer, 
              alpha=0.025, min_alpha=0.001, passes=20,
              model_name='Doc2Vec(dm/c,d100,n5,w5,mc2,t8)', 
              threshold=0.8, 
              cnt_threshold=10, 
              repeat=5,
              get_topic_func=du.get_all_topics
             ):
        self.reset(train_df[:])
        self.tokenize(tokenizer)
        self.doc_train(w2v_path, alpha, min_alpha, passes)
        self.d2v_save(path, prefix)
        self.select_model(model_name)
        self.cluster_train(typ, path, prefix, threshold, cnt_threshold, repeat, get_topic_func)
    
    
    def cluster_train(self, typ, path, prefix,
              threshold=0.8, 
              cnt_threshold=10, 
              repeat=5,
              get_topic_func=du.get_all_topics
             ):
        if(typ=='cate'):
            self.clustering_cate(threshold, repeat)
        else:
            self.clustering(threshold, repeat)
        self.iner_score(threshold, cnt_threshold)
#        self.get_all_topics(get_topic_func)#TODO
        self.calc_elapsed()
#        self.save(path, prefix)#TODO
        
            
    def save_to_db(self, prefix, collection):
        clusters = []

        time = datetime.datetime.now()
        clusters_infors = self.countby_.sort_values('cohesion', ascending=False)

        prefix = prefix * 1000

        for idx, info in clusters_infors.iterrows():
            new_cluster = prefix + idx
            leading = self.getMainArticle(info.cluster).to_dict()
            
            articles = []
            for idx, row in self.df_[self.df_.cluster==info.cluster].iterrows():
                row_dict = row.to_dict()
                row_dict['cluster'] = new_cluster
                articles.append(row_dict)
            
            if(leading['imageURL'] == ''):
                for article in articles:
                    if(article['imageURL'] != ''):
                        leading['imageURL'] = article['imageURL']
                        break
            
            cluster = {
                "cluster": new_cluster,
                "cohesion": info.cohesion,
                "count": info.cnt,
                "leading": leading,
                "clusteredAt": time,
                "articles": articles
            }
            clusters.append(cluster)
            
        collection.insert_many(clusters)