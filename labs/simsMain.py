# coding: utf-8

from SimilarityClustering import SimilarityClustering
import articles_data

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import cnouns as cn
from pymongo import MongoClient
import datetime

client = MongoClient('mongodb://localhost:27017/somanews')
client.somanews.authenticate('ssomanews', 'ssomanews1029')
db = client.get_database('somanews')

crawled_collection = db.get_collection('crawledArticles')
clusters_collection = db.get_collection('bclusters')
articles_collection = db.get_collection('barticles')

datastore_dir = "../datastore/"
catelist_path = datastore_dir + "category2.p"
w2v_src_dir = datastore_dir + "w2v_src4"
w2v_path = datastore_dir + "sejongcorpus_w2v4_2.p"
nnp_dict_path = datastore_dir + "nnps2.p"
corpus_path = datastore_dir + "corpus2.p"

target_time = datetime.datetime.now()
# target_time = datetime.datetime(2016, 11, 15)
prefix = int("%.2d%.2d"%(target_time.month, target_time.day))
prefix_str = "%d_00" % prefix

nnp_dict_df = pd.read_pickle(nnp_dict_path)
nnp_dict_df = nnp_dict_df[nnp_dict_df>10]
nnp_dict = nnp_dict_df.index.tolist()

custom_dict = [u'새누리', u'새누리당', u'더민주', u'더민주당', u'최순실', u'박대통령', u'국회의장', u'야권의요구', u'정기국회', u'참여정부']
dicts = set(nnp_dict + custom_dict)

def tokenizer(inp_str):
    return cn.custom_pos_tags(inp_str, dicts)

# # Model
train_df = articles_data.find_recent_articles(crawled_collection, catelist_path, target_time)
sc = SimilarityClustering()
sc.train("cate", w2v_path, train_df, path=datastore_dir, prefix=prefix_str, tokenizer=tokenizer,
            threshold=0.65,
            cnt_threshold=10,
            repeat=3,
            model_name='dbow+dmm')

# # Save
sc.iner_score(threshold=0.7, cnt_threshold=8)
sc.save(path=datastore_dir, prefix=prefix_str)
calced_clusters = sc.save_to_db(prefix, clusters_collection, articles_collection, target_time)