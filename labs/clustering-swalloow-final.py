
# coding: utf-8

# # News Clustering using KMeans Algorithm
# By Datetime : 2016-08-29 ~ 2016-09-05

import numpy as np
import pandas as pd
import articles_data_py3 as articles_data
from pymongo import MongoClient


# ## Load data from MongoDB
client = MongoClient('mongodb://localhost:27017/somanews')
client.somanews.authenticate('ssomanews', 'ssomanews1029')
db = client.get_database('somanews')

crawled_collection = db.get_collection('crawledArticles')
clusters_collection = db.get_collection('articles')


# ## Select Categories
catelist_path = '../datastore/category.p'
headline_path = '../datastore/headline.p'

train = articles_data.find_recent_articles(crawled_collection, catelist_path)


# ## Preprocessing
import datetime
from konlpy.tag import Mecab
import cnouns
import hanja
import re

mecab = Mecab()

def tokenize(data):
    return [' '.join(e for e in mecab.nouns(data))]

train['title_flat'] = train['title'].apply(lambda text: cnouns.text_cleaning_without_special_ch(text))
train['title_flat'] = train['title'].apply(lambda text: articles_data.remove_headlines(text, headline_path))
title = [tokenize(each[1]['title_flat']) for each in train.iterrows()]


# ## Training
# 1. Feature extraction - TfidVectorizer
# 2. Decomposition - PCA
# 3. Cluster - KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1,2))
title_flat = [item for sublist in title for item in sublist]
x_list = vectorizer.fit_transform(title_flat)

x_list_100d = PCA(n_components=100).fit_transform(x_list.toarray())


# ### Scoring
from sklearn.metrics import silhouette_samples, silhouette_score

# ### Best Silhoutte Score
best_score = 0.0
best_k = 0

for k in range(15, 35):
    km = KMeans(n_clusters=k, n_jobs=-1).fit(x_list_100d)
    score = silhouette_score(x_list_100d, km.labels_)
    if best_score < score:
        best_score = score
        best_k = k


# ### K-Means Algorithm
km = KMeans(n_clusters=best_k, n_jobs=-1).fit(x_list_100d)
labels = km.labels_
centroids = km.cluster_centers_

train = train.drop(['title_flat'], axis=1)
train['cluster'] = labels


# ## Choose Best Cluster
# - Minimum inertia
sample_silhouette_values = silhouette_samples(x_list_100d, labels)
sample_silhouette_score = []
best_cluster = []

for i in range(best_k):
    ith_cluster_silhouette_values =         sample_silhouette_values[labels == i]
    sample_silhouette_score.append(abs(ith_cluster_silhouette_values.mean()))

sample_silhouette_score.sort(reverse=True)
sample_silhouette_score = sample_silhouette_score[:12]

for i in range(best_k):
    ith_cluster_silhouette_values =         sample_silhouette_values[labels == i]

    if abs(ith_cluster_silhouette_values.mean()) in sample_silhouette_score:
        best_cluster.append(i)

train = train[train['cluster'].isin(best_cluster)]


# ## Save Dataframe to MongoDB
client = MongoClient('mongodb://localhost:27017/somanews')
client.somanews.authenticate('ssomanews', 'ssomanews1029')
db = client.get_database('somanews')
articles = db.get_collection('articles')

articles.insert_many(train.to_dict(orient='records'))
client.close()
