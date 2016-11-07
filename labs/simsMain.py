# coding: utf-8

from SimilarityClustering import SimilarityClustering
import articles_data

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import cnouns as cn
from pymongo import MongoClient
from datetime import datetime

# # DB
client = MongoClient('mongodb://localhost:27017/somanews')
client.somanews.authenticate('ssomanews', 'ssomanews1029')
db = client.get_database('somanews')

crawled_collection = db.get_collection('crawledArticles')
clusters_collection = db.get_collection('clusters')

# # Params
catelist_path = '../datastore/category2.p'
w2v_src_dir = "../datastore/w2v_src"
w2v_path = "../datastore/sejongcorpus_w2v2.p"
corpus_path = "../datastore/corpus2.p"
now = datetime.now()
prefix = int("%.2d%.2d"%(now.month, now.day))
prefix_str = "%d_00" % prefix

# # Clustering
train_df = articles_data.find_recent_articles(crawled_collection, catelist_path)
sc = SimilarityClustering()
sc.train("cate", w2v_path, train_df, path="../datastore", prefix=prefix_str)

# # Save
sc.save_to_db(prefix, clusters_collection)

print("complete!!")