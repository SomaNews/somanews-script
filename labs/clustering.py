# coding: utf-8

# # Baseline script for SomaNews Clustering

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import datetime
from konlpy.tag import Mecab
import hanja
import re

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

train = pd.read_pickle("../datastore/weekly.p")
train = train.drop(['newsNum'], axis=1)

mecab = Mecab()

def text_cleaning(text):
    text = hanja.translate(text, 'substitution')
    text = re.sub('[^가-힝0-9a-zA-Z\\s]', '', text)
    text = text.replace(u"카드뉴스", '').replace(u"조선일보", '')
    return text

def tokenize(data):
    return [' '.join(e for e in mecab.nouns(data))]

train['title'] = train['title'].apply(lambda text: text_cleaning(text))
title = [tokenize(each[1]['title']) for each in train.iterrows()]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

vectorizer = TfidfVectorizer(lowercase=False)
title_flat = [item for sublist in title for item in sublist]
x_list = vectorizer.fit_transform(title_flat)
x_list_10d = PCA(n_components=10).fit_transform(x_list.toarray())

best_score = 0.0
best_k = 0

for k in range(5, 15):
    km = KMeans(n_clusters=k, n_jobs=-1).fit(x_list_10d)
    score = silhouette_score(x_list, km.labels_)
    if best_score < score:
        best_score = score
        best_k = k
print("In Clusters = ", best_k, ", Best score is : %0.3f" % best_score)

km = KMeans(n_clusters=best_k, n_jobs=-1).fit(x_list)
labels = km.labels_
centroids = km.cluster_centers_
print("Cluster ineritia = ", km.inertia_)

train = train.reset_index(drop=True)
train['cluster'] = labels

x_list_vector = x_list_10d.tolist()
train['vector'] = x_list_vector

from pymongo import MongoClient

client = MongoClient('mongodb://ssomanews:ssomanews1029@ds021346.mlab.com:21346/somanews', 27017)
db = client.get_database('somanews')
clusters = db.get_collection('clusters')
clusters.insert_many(train.to_dict(orient='records'))