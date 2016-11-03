import numpy as np
import math
from collections import OrderedDict
import itertools
import check_utils as cu
import cnouns as cn
from time import time
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from gensim.models.ldamodel import LdaModel
from gensim import corpora, models

def test_print(cluster, df, docvecs, centers, topics, score):
    print(score[score.cluster==cluster])
    print("")
    print("Most similar -", df.loc[docvecs.most_similar([centers[cluster]])[0][0]].title)
    print("")
    topic_print(topics[cluster])
    print("")
    clusters = df[df.cluster==cluster]
    clusters = clusters[['title','similarity']]
    for idx, row in clusters.sort_values('similarity', ascending=False).iterrows():
        print(row['similarity'], idx, row['title'])

## topics
def get_all_topics(df, clusters, num_topics=3, num_words=3):
    topics = {}
    idx = 0
    size = len(clusters)
    print("Number of cluster : %d" % size)
    for n in clusters:
        print("progress - %d / %d" % (idx, size))
        topics[n] = get_topics(df, n, num_topics, num_words)
        idx = idx + 1
        
    return topics

def get_topics(df, cluster, num_topics=3, num_words=3):
    clusters = df[df.cluster==cluster]
    texts = [cn.tokenize_nouns(row.title + " " + row.content).split(' ') for idx, row in clusters.iterrows()]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=20)
    topics = ldamodel.show_topics(num_topics=num_topics, num_words=num_words, log=True, formatted=False)
    return [t[1] for t in topics]

def topic_print(topics):
    topic_words = [[t[0] for t in topic] for topic in topics]
    topic_strs = []
    for topic_word in topic_words:
        topic_str = ""
        for tw in topic_word:
            topic_str = topic_str + " " + tw
        topic_strs.append(topic_str)

    for t in topic_strs:
        print(t)

## similarity
def calc_similarity(df, docvecs, centers):
    df['similarity'] = [cs_similarity(docvecs[idx], centers[row['cluster']]) for idx, row in df.iterrows()]
    
def similarity_iner_score(centers, df, docvecs):
    scores = []
    clusters = df.cluster
    for n in clusters.unique():
        center = centers[n]
        cluster = df[df.cluster==n]
        size = len(cluster)
        
        cluster_similarity = 0
        cluster_distance = 0
        for idx, row in cluster.iterrows():
            cluster_similarity = cluster_similarity + cs_similarity(center, docvecs[idx])
            cluster_distance = cluster_distance + cs_distance(center, docvecs[idx])
        
        variance = cluster_distance / size
        similarity = cluster_similarity / size
        scores.append((n, size, cluster_distance, variance, similarity, similarity / variance))

    return pd.DataFrame(scores, columns = ['cluster', 'cnt', 'distance', 'variance', 'similarity', 'cohesion'])

def similarity_clustering(df, docvecs, threshold=0.5):
    t0 = time()
    # Initialize
    df['rank'] = np.zeros(len(df))
    df['cluster'] = [idx for idx, row in df.iterrows()]
    centers = {}
    sd = {}
    for idx, row in df.iterrows():
        centers[idx] = docvecs[idx]
        
    # Calculate similarity
    print("Calculate similarity. size:%d"%len(df))
    for t in itertools.combinations(range(len(df)), 2):
        similarity = docvecs.similarity(d1=t[0], d2=t[1])
        if(similarity >= threshold):
            sd[t] = similarity
            
    # Sort
    print("Sorting")
    ordered_sd = OrderedDict(sorted(sd.items()), key=lambda x: -x[1])
    
    # Clustering
    size = len(ordered_sd.items()[:-1])
    cnt = 0
    per = 0
    print("Clustering - size:%d" % (size))
    
    for key, similarity in ordered_sd.items()[:-1]:
        cnt = cnt + 1
        tmp = int(1000 * cnt / size)
        if(per + 1 == tmp): 
            per = per + 1
            print("progress - %d / 1000" % per)
        u_cluster = find(df, key[0])
        v_cluster = find(df, key[1])
        vec_sim = cs_similarity(centers[u_cluster], centers[v_cluster])
        if(u_cluster != v_cluster and vec_sim >= threshold):
            cluster = union(df, u_cluster, v_cluster)
            centers[cluster] = merge_similarity(centers[u_cluster], centers[v_cluster])
            
    for idx, row in df.iterrows():
        find(df, idx)
            
    print("Done in %0.3fs." % (time() - t0))
    return centers
    
# cosine
def cs_similarity(v1, v2, n = 100):
    cs = cosine_similarity(v1.reshape(1,n), v2.reshape(1,n))[0][0]
    if(cs>1): return 1
    elif(cs<-1): return -1
    else: return cs
    
def cs_distance(v1, v2, n = 100):
    similarity = cs_similarity(v1, v2, n)
    return np.arccos(similarity) / math.pi

def merge_similarity(v1, v2):
    return (v1 + v2) / 2

## merge
def find(df, idx):
    parent = df.loc[idx]['cluster']
    if(idx == parent): return idx

    newParent = find(df, parent)
    df.set_value(idx, 'cluster', newParent)
    return newParent

def union(df, u_cluster, v_cluster):
    if (u_cluster == v_cluster): return
    
    u_rank = df.loc[u_cluster]['rank']
    v_rank = df.loc[v_cluster]['rank']
    if(u_rank > v_rank):
        df.set_value(v_cluster, 'cluster', u_cluster)
        return u_cluster
    else:
        df.set_value(u_cluster, 'cluster', v_cluster)
        if(u_rank == v_rank):
            df.set_value(v_cluster, 'rank', v_rank + 1)
        return v_cluster

## Deprecated
def similarity_clustering_tuple(df, docvecs, threshold=0.5):
    t0 = time()
    # Initialize
    df['rank'] = np.zeros(len(df))
    df['cluster'] = [idx for idx, row in df.iterrows()]
    sd = {}
    
    # Calculate similarity
    print("Calculate similarity. size:%d"%len(df))
    for t in itertools.combinations(range(len(df)), 2):
        similarity = docvecs.similarity(d1=t[0], d2=t[1])
        if(similarity >= threshold):
            sd[t] = similarity
            
    # Sort
    print("Sorting")
    ordered_sd = OrderedDict(sorted(sd.items()), key=lambda x: -x[1])
    
    size = len(ordered_sd.items()[:-1])
    cnt = 0
    persize = int(size / 100)
    per = 0
    print("Clustering - size:%d, persize:%d" % (size, persize))
    # Clustering
    for key, similarity in ordered_sd.items()[:-1]:
        cnt = cnt + 1
        if(cnt % persize == 0): 
            per = per + 1
            print("passed - %d" % per)
        u_cluster = find(df, key[0])
        v_cluster = find(df, key[1])
        if(u_cluster != v_cluster):
            union(df, u_cluster, v_cluster)
            
    print("Done in %0.3fs." % (time() - t0))