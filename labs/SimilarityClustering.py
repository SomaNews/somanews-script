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
import gensim
from gensim import models
from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec, LabeledSentence
from collections import OrderedDict
import articles_data
import datetime 
from tqdm import tqdm
from random import sample

import pymongo
import multiprocessing
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
import cPickle as pickle
from spherecluster import SphericalKMeans
cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

from random import shuffle

from pymongo.errors import BulkWriteError
from ntc_rank import calc_issue_rank, d_factor
from sklearn.decomposition import PCA

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
        self.set_infer()
        self.is_pretrained = False
        
    def get_docvecs(self):
        if self.is_pretrained:
            return self.inferred_docvecs
        else:
            return self.dm_.docvecs
        
    def reset(self, train_df):
        self.df_ = train_df.reset_index(drop=True)
        self.df_ = self.df_.sort_index()
        
    def set_infer(self, alpha=0.1, min_alpha=0.001, steps=5):
        self.alpha_ = alpha
        self.min_alpha_ = min_alpha
        self.steps_ = steps
        
    def tokenize(self, tokenizer=cn.tokenizer):
        self.times_["preprocessing"]["start"] = time()
        self.alldocs_ = []
        size = len(self.df_) / 4
        
        ## tokenize
        self.df_['target_str'] = [tokenizer(row.title + " " + row.content) for idx, row in tqdm(self.df_.iterrows(), desc="Tokenizing", total=len(self.df_))]
        
        ## make docs
        for idx, row in self.df_.iterrows():
            words = row['target_str'].split(' ')
            tags = [idx]
            self.alldocs_.append(TaggedDocument(words, tags))
            
        print("Size of documents: %d"%(len(self.alldocs_)))
        self.df_.drop(['target_str'], axis=1, inplace=True)
        self.times_["preprocessing"]["end"] = time()
    
    def calc_error_rate(self, model, sample_ratio=0.1):
        test_data = sample(self.alldocs_, int(sample_ratio * len(self.alldocs_)))

        corrects = 0
        for doc in test_data:
            inferred_docvec = model.infer_vector(doc.words, alpha=self.alpha_, min_alpha=self.min_alpha_, steps=self.steps_)
            most_similar_docs = model.docvecs.most_similar([inferred_docvec], topn=1)
            for msd in most_similar_docs:
                if msd[0] == doc.tags[0]:
                    corrects = corrects + 1
                    break

        return (len(test_data) - corrects) / float(len(test_data))
    
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
        self.models_by_name_ = OrderedDict()
        simple_models[0].build_vocab(self.alldocs_)
        #print simple_models[0]
        for model in simple_models[1:]:
            model.reset_from(simple_models[0])
            #print(model)

        self.models_by_name_['dmc'] = simple_models[0]
        self.models_by_name_['dbow'] = simple_models[1]
        self.models_by_name_['dmm'] = simple_models[2]
        
        print("Size of docvecs: %d"%(len(simple_models[0].docvecs)))
        
        assert len(simple_models[0].docvecs) == len(self.alldocs_), "docvecs and documents is not matched."  
            
        ## training
        doc_list = self.alldocs_[:]
        alpha_delta = (alpha - min_alpha) / passes

        self.error_rates_ = []    
        for epoch in tqdm(range(passes), desc="Doc2vec train"):
            shuffle(doc_list)  # shuffling gets best results
            
            eorror_rates = {"dmm":-1,"dmc":-1,"dbow":-1}
            for name, train_model in self.models_by_name_.items():
                train_model.alpha, train_model.min_alpha = alpha, alpha
                train_model.train(doc_list)
                error_rate = self.calc_error_rate(train_model)
                eorror_rates[name] = error_rate
                print("%i passes : %s, error_rate:%f" % (epoch + 1, name, error_rate))
                
            self.error_rates_.append(eorror_rates)
            #print('completed pass %i at alpha %f' % (epoch + 1, alpha))
            alpha -= alpha_delta
    
        self.concat_vec()
        self.times_["learning"]["end"] = time()
        
    
    def select_model(self, model_name='dmm'):
        self.dm_ = self.models_by_name_[model_name]
        
        
    def clustering(self, threshold=0.8, repeat=5, dt_threshold=0.05):
        self.times_["clustering"]["start"] = time()
        print("Similarity clustering.....")
        self.centers_, clusters = du.similarity_clustering(self.df_, self.get_docvecs(), threshold, repeat, dt_threshold)
#        self.centers_, clusters = du.similarity_clustering_time(self.df_, self.get_docvecs(), threshold, repeat)
        sorted_clusters = clusters.sort_index()
        self.df_['cluster'] = sorted_clusters.parent
        du.calc_similarity(self.df_, self.get_docvecs(), self.centers_)
        print("Complete to similarity clustering.")
        self.times_["clustering"]["end"] = time()
        
        
    def clustering_cate(self, threshold=0.8, repeat=5, dt_threshold=0.05):
        self.times_["clustering"]["start"] = time()
        print("Similarity cate clustering.....")
        cates = articles_data.get_target_cate()
        centers = {}
        clusters = []
        for cate in cates:
            print("\nClustering category : %s"%cate)
            center, cluster = du.similarity_clustering(self.df_[self.df_.cate==cate], self.get_docvecs(), threshold, repeat, dt_threshold)
#            center, cluster = du.similarity_clustering_time(self.df_[self.df_.cate==cate], self.get_docvecs(), threshold, repeat)
            centers.update(center)
            clusters.append(cluster)    
        self.centers_ = centers
        cluster_df = pd.concat(clusters, axis=0)
        sorted_clusters = cluster_df.sort_index()
        self.df_['cluster'] = sorted_clusters.parent
        du.calc_similarity(self.df_, self.get_docvecs(), self.centers_)
        print("Complete to similarity clustering.")
        self.times_["clustering"]["end"] = time()
        
        
    def iner_score(self, threshold=0.8, cnt_threshold=10):
        self.scores_, self.countby_ = du.iner_socre(self.centers_, self.df_, self.get_docvecs(), threshold, cnt_threshold)
        
        
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
        self.models_by_name_['dmc'].save("%s/%s_d2v-dmc.p" % (path, prefix))
        self.models_by_name_['dbow'].save("%s/%s_d2v-dbow.p" % (path, prefix))
        self.models_by_name_['dmm'].save("%s/%s_d2v-dmm.p" % (path, prefix))

        self.df_.to_pickle("%s/%s_df.p" % (path, prefix))
        
        pickle.dump(self.centers_, open("%s/%s_centers.p" % (path, prefix), "wb"))
        if not self.is_pretrained:
            pickle.dump(self.error_rates_, open("%s/%s_error_rates.p" % (path, prefix), "wb"))
#        pickle.dump(self.topics_, open("%s/%s_topics.p" % (path, prefix), "wb")) # TODO
        pickle.dump(self.times_, open("%s/%s_times.p" % (path, prefix), "wb"))

        print("Complete to save model.")
        
    def concat_vec(self):
        self.models_by_name_['dbow+dmm'] = ConcatenatedDoc2Vec([self.models_by_name_['dbow'], self.models_by_name_['dmm']])
        self.models_by_name_['dbow+dmc'] = ConcatenatedDoc2Vec([self.models_by_name_['dbow'], self.models_by_name_['dmc']])

    def d2v_save(self, path, prefix):
        self.models_by_name_['dmc'].save("%s/%s_d2v-dmc.p" % (path, prefix))
        self.models_by_name_['dbow'].save("%s/%s_d2v-dbow.p" % (path, prefix))
        self.models_by_name_['dmm'].save("%s/%s_d2v-dmm.p" % (path, prefix))
        
        self.df_.to_pickle("%s/%s_df.p" % (path, prefix))
    
    def infer_docvecs(self, alpha, min_alpha, steps):
        self.tokenize()
        self.inferred_docvecs = [self.dm_.infer_vector(doc.words, alpha=alpha, min_alpha=min_alpha, steps=steps) for doc in tqdm(self.alldocs_, desc="Infer docvecs")]
    
    def s_load_pretrained(path, prefix, model_path, train_df, tokenizer, threshold=0.8, cnt_threshold=10, model_name='dbow+dmm', alpha=0.1, min_alpha=0.001, steps=5):
        sc = SimilarityClustering()
        sc.is_pretrained = True
        
        sc.models_by_name_ = OrderedDict()
        
        sc.models_by_name_['dmc'] = Doc2Vec.load("%s/%s_d2v-dmc.p" % (path, model_path))
        sc.models_by_name_['dbow'] = Doc2Vec.load("%s/%s_d2v-dbow.p" % (path, model_path))
        sc.models_by_name_['dmm'] = Doc2Vec.load("%s/%s_d2v-dmm.p" % (path, model_path))

        sc.concat_vec()
        sc.select_model(model_name)
        
        sc.reset(train_df)
        sc.infer_docvecs(alpha, min_alpha, steps)
        
        return sc
    load_pretrained=staticmethod(s_load_pretrained)
    
    def s_load(path, prefix, threshold=0.8, cnt_threshold=10, model_name='dbow+dmm', only_d2v=False):
        sc = SimilarityClustering()
        
        sc.models_by_name_ = OrderedDict()
        
        sc.models_by_name_['dmc'] = Doc2Vec.load("%s/%s_d2v-dmc.p" % (path, prefix))
        sc.models_by_name_['dbow'] = Doc2Vec.load("%s/%s_d2v-dbow.p" % (path, prefix))
        sc.models_by_name_['dmm'] = Doc2Vec.load("%s/%s_d2v-dmm.p" % (path, prefix))

        sc.concat_vec()
        sc.select_model(model_name)
        
        sc.df_ = pd.read_pickle("%s/%s_df.p" % (path, prefix))
        
        if not(only_d2v):
            sc.centers_ = pickle.load(open("%s/%s_centers.p" % (path, prefix), "rb"))
#            sc.topics_ = pickle.load(open("%s/%s_topics.p" % (path, prefix), "rb")) # TODO
#            sc.error_rates_ = pickle.load(open("%s/%s_error_rates.p" % (path, prefix), "rb"))
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
#            print du.test_print(row.cluster, self.df_, self.get_docvecs(), self.centers_, self.topics_, self.countby_, threshold, diff_threshold)
#            print("------------------------------------------------------------")
            
    
    def print_cluster(self, cluster, threshold=0.8, diff_threshold=0.01):
        print du.test_print(cluster, self.df_, self.get_docvecs(), self.centers_, self.topics_, self.countby_, threshold, diff_threshold)
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
            print cnt,  self.df_.loc[self.get_docvecs().most_similar([self.centers_[row.cluster]])[0][0]].title
            
    def getMainArticle(self, cluster):
        return self.df_.loc[self.get_docvecs().most_similar([self.centers_[cluster]])[0][0]]
            
    def train(self, typ, w2v_path, train_df, path, prefix,
              tokenizer=cn.tokenizer, 
              alpha=0.025, min_alpha=0.001, passes=20,
              model_name='dmm', 
              threshold=0.8, 
              cnt_threshold=10, 
              repeat=5,
              get_topic_func=du.get_all_topics,
              dt_threshold=0.05
             ):
        self.reset(train_df[:])
        self.tokenize(tokenizer)
        self.doc_train(w2v_path, alpha, min_alpha, passes)
        self.d2v_save(path, prefix)
        self.select_model(model_name)
        self.cluster_train(typ, path, prefix, threshold, cnt_threshold, repeat, get_topic_func, dt_threshold)
    
    
    def cluster_train(self, typ, path, prefix,
              threshold=0.65, 
              cnt_threshold=10, 
              repeat=3,
              get_topic_func=du.get_all_topics,
              dt_threshold=0.05
             ):
        if(typ=='cate'):
            self.clustering_cate(threshold, repeat, dt_threshold)
        else:
            self.clustering(threshold, repeat, dt_threshold)
        self.iner_score(threshold, cnt_threshold)
#        self.get_all_topics(get_topic_func)#TODO
        self.calc_elapsed()
#        self.save(path, prefix)#TODO
    
    def print_cluster_rank(self, calced_clusters):
        cdf = pd.DataFrame(calced_clusters, columns=['cluster', 'portion', 'deltaTime', 'cohesion', 'rank'])
        sort_cdf = cdf.sort_values('rank', ascending=False)
        cnt = 0
        for idx, row in sort_cdf.iterrows():
            cnt = cnt + 1
            for c in calced_clusters:
                if(c['cluster'] == row.cluster):
                    print cnt, row['cluster'], row['rank'], c['leading']['title']
           
    def print_error_rate(self):
        dmc = []
        dmm = []
        dbow = []
        for er in self.error_rates_:
            dmc.append(er['dmc'])
            dmm.append(er['dmm'])
            dbow.append(er['dbow'])

        plt.ylabel('Error Rate')
        plt.plot(range(len(self.error_rates_)), dmc, 'b', label='DM/M')
        plt.plot(range(len(self.error_rates_)), dmm, 'g', label='DM/C')
        plt.plot(range(len(self.error_rates_)), dbow, 'r', label='DBOW')
        plt.legend()
        plt.show()

        print("Error Rates")
        print("\tDM/C - Min: %f, Last: %f"%(min(dmc), dmc[-1]))
        print("\tDM/M - Min: %f, Last: %f"%(min(dmm), dmm[-1]))
        print("\tDBOW - Min: %f, Last: %f"%(min(dbow), dbow[-1]))
        #print("\tDM/C - %f"%(dmc[-1]))
        #print("\tDM/M - %f"%(dmm[-1]))
        #print("\tDBOW - %f"%(dbow[-1]))
        
    def get_cluster_similarity(self, threshold, repeat, path, prefix):
        self.cluster_train("cate", path=path, prefix=prefix,
                      threshold=threshold, 
                      cnt_threshold=10, 
                      repeat=repeat)

        vec = []
        for idx, row in self.countby_.iterrows():
            simliarities = self.df_[self.df_.cluster==row['cluster']].similarity
            sims = sum(simliarities) / len(simliarities)
            vec.append(sims)

        return vec

    def print_cluster_similarity(self):
        vec = []
        for idx, row in self.countby_.iterrows():
            simliarities = self.df_[self.df_.cluster==row['cluster']].similarity
            sims = sum(simliarities) / len(simliarities)
            vec.append(sims)

        plt.xlabel("Cluster")
        plt.ylabel("Similarity")
        plt.plot(range(len(vec)), vec, 'b')
        plt.show()
        
    def save_to_db(self, prefix, cluster_collection, article_collection, target_time, factor=d_factor, test=False):
        clusters = []

        time = target_time
        clusters_infors = self.countby_.sort_values('similarity', ascending=False)
        
        prefix = prefix * 1000
        
        if self.is_pretrained:
            vec_size = len(self.get_docvecs())
        else:
            vec_size = len(self.models_by_name_['dmm'].docvecs)
        vectors = []
        for i in range(vec_size):
            vectors.append(self.get_docvecs()[i])
        pca = PCA(n_components=100).fit_transform(vectors)
        
        cluster_vectors = []
        cluster_vector_idx = {}
        for idx, cluster in enumerate(self.df_.cluster.unique()):
            cluster_vectors.append(self.centers_[cluster])
            cluster_vector_idx[cluster] = idx
        cluster_pca = PCA(n_components=100).fit_transform(cluster_vectors)

        article_list = []
        for idx, info in clusters_infors.iterrows():
            new_cluster = prefix + idx
            cluster_vector = cluster_pca[cluster_vector_idx[info['cluster']]].tolist()

            articles = []
            for idx, row in self.df_[self.df_.cluster==info.cluster].iterrows():
                self.df_.set_value(idx, 'new_cluster', new_cluster)
                row_dict = row.to_dict()
                row_dict['cluster'] = new_cluster
                row_dict['clusteredAt'] = time
                row_dict['vector'] = pca[idx].tolist()
                row_dict['article_id'] = row_dict['_id']
                del row_dict['_id']
                articles.append(row_dict)
                article_list.append(row_dict)
            
            cates = {}
            for cate in articles_data.get_target_cate():
                cate_items = [article for article in articles if article[u'cate'] == cate]
                count = len(cate_items)
                cates[cate] = count
            
            leading = articles[0]
            for article in articles:
                if article[u'imageURL'] != '':
                    if((article['publishedAt'] - leading['publishedAt']).total_seconds() > 0):
                        leading = article
                    elif leading[u'imageURL'] == '':
                        leading = article
                        
#            if(leading['imageURL'] == ''):
#                for article in articles:
#                    if(article['imageURL'] != ''):
#                        leading['imageURL'] = article['imageURL']
#                        break
            
            cluster = {
                "cluster": new_cluster,
                "cohesion": info.similarity,
                "count": info.cnt,
                "cate": cates,
                "leading": leading,
                "clusteredAt": time,
                "vector": cluster_vector,
                "articles": articles
            }
            clusters.append(cluster)
        # end for  
        
        cluster_set = set(clusters_infors.cluster)
        for idx, row in self.df_.iterrows():
            if not row.cluster in cluster_set:
                row_dict = row.to_dict()
                row_dict['cluster'] = -1
                row_dict['new_cluster'] = -1
                row_dict['clusteredAt'] = time
                row_dict['vector'] = pca[idx].tolist()
                row_dict['article_id'] = row_dict['_id']
                del row_dict['_id']
                article_list.append(row_dict)
            
        for cluster in clusters:
            if(len(cluster['leading']['imageURL']) < 1):
                print("imageURL is empty! Cluster: %d, count: %d"%(cluster['cluster'], cluster['count']))
                
        calced_cluster, sort_cdf = calc_issue_rank(clusters, factor)
        
        print("Number of clusters is %d"%len(calced_cluster))
        print("Number of articles is %d"%len(article_list))
        
        if(not test):
            try:
                cluster_collection.insert_many(calced_cluster)
            except BulkWriteError as bwe:
                print(bwe.details)
                raise
                
            for doc in article_list:
                try:
                    article_collection.insert(doc)
                except pymongo.errors.DuplicateKeyError:
                    print 'Duplicate article %s' % doc['_id']
        
        return calced_cluster, sort_cdf