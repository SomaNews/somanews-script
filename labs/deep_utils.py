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
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import mktime
import math
from datetime import datetime

def test_print(cluster, df, docvecs, centers, topics, score, threshold, diff_threshold):
    print(score[score.cluster==cluster])
    print("")
    print "Most similar -", df.loc[docvecs.most_similar([centers[cluster]])[0][0]].title
    print("") 
#    topic_print(topics[cluster]) #TODO
    print("")
    clusters = df[df.cluster==cluster]
    clusters = clusters[['title','similarity']]
    sims = []
    for idx, row in clusters.sort_values('similarity', ascending=False).iterrows():
        sims.append(row['similarity'])
        print row['similarity'], idx, row['title']
        
    margin = 0.001
    plt.plot(range(len(sims)), sims)
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.ylim(min(sims) - margin, max(sims) + margin)
    plt.ylabel('Similarity')
    plt.show()
    
    sims_diff = np.absolute(np.diff(np.array(sims)))
    plt.plot(range(len(sims_diff)), sims_diff)
    plt.axhline(y=diff_threshold, color='r', linestyle='-')
    plt.ylabel('Differential')
    plt.show()

## topics
def get_all_topics(df, clusters, num_topics=3, num_words=3):
    topics = {}
    idx = 0
    size = len(clusters)
    print("Number of cluster : %d" % size)
    for n in tqdm(clusters):
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
    
def similarity_iner_score(centers, df, docvecs, threshold):
    scores = []
    clusters = df.cluster
    total_size = len(df)
    now = datetime.now()
    
    for n in clusters.unique():
        center = centers[n]
        cluster = df[df.cluster==n]
        size = len(cluster)
        
        cluster_similarity = 0
        cluster_distance = 0
        cluster_time = 0
        for idx, row in cluster.iterrows():
            cluster_similarity = cluster_similarity + cs_similarity(center, docvecs[idx])
            cluster_distance = cluster_distance + cs_distance(center, docvecs[idx])
            cluster_time = cluster_time + (now - row.publishedAt).total_seconds()
        
        time_mean = cluster_time / size
        time_v = 0
        for idx, row in cluster.iterrows():
            time_v = math.pow(time_mean - (now - row.publishedAt).total_seconds(), 2)
        
        time_v = time_v / size
        variance = cluster_distance / size
        similarity = cluster_similarity / size
        in_threshold = len(cluster[cluster.similarity>=threshold])
        in_threshold_ratio = in_threshold / float(size)
        portion = size / float(total_size)
        cohesion = 100 * portion * in_threshold_ratio * similarity / variance
        scores.append((n, size, portion, in_threshold, in_threshold_ratio, time_mean, time_v, cluster_distance, variance, similarity, cohesion))

    return pd.DataFrame(scores, columns = ['cluster', 'cnt', 'portion', 'in_threshold', 'in_ratio', 'time_mean', 'time_v', 'distance', 'variance', 'similarity', 'cohesion'])

def similarity_clustering(df_, docvecs, threshold=0.8, repeat=5):
    t0 = time()
    # Initialize
    pr = []
    centers = {}
    for idx, row in df_.iterrows():
        centers[idx] = docvecs[idx]
        pr.append((idx, 0))
    clu_rank = pd.DataFrame(pr, df_.index, columns = ['parent', 'rank'])

    for i in range(0, repeat):
        print("Iter %d/%d"%(i+1,repeat))
                      
        # Calculate similarity
        sd = {}
        p_unique = clu_rank.parent.unique()
        print("Calculate similarity. size:%d"%len(p_unique))
        for t in tqdm(list(itertools.combinations(p_unique, 2))):
            similarity = cs_similarity(centers[t[0]], centers[t[1]])
            if(similarity >= threshold):
                sd[t] = similarity

        # Sort
        print("Sorting")
        ordered_sd = OrderedDict(sorted(sd.items()), key=lambda x: -x[1])

        # Clustering
        size = len(ordered_sd.items()[:-1])
        print("Clustering - size:%d" % (size))

        for key, similarity in tqdm(ordered_sd.items()[:-1]):
            u_cluster = find(clu_rank, key[0])
            v_cluster = find(clu_rank, key[1])
            vec_sim = cs_similarity(centers[u_cluster], centers[v_cluster])
            if(u_cluster != v_cluster and vec_sim >= threshold):
                cluster = union(clu_rank, u_cluster, v_cluster)
                centers[cluster] = merge_similarity(centers[u_cluster], centers[v_cluster])

        for idx, row in clu_rank.iterrows():
            find(clu_rank, idx)
        
        if(i != repeat - 1):
            for idx, row in clu_rank.iterrows():
                center = centers[row.parent]
                similarity = cs_similarity(docvecs[idx], center)
                if(similarity<threshold):
                    clu_rank.set_value(idx, 'parent', idx)
                    
            for p in clu_rank.parent.unique():
                cluster = clu_rank[clu_rank.parent==p]
                center = docvecs[p]
                for i, c in cluster.iterrows():
                    center = merge_similarity(center, docvecs[i])
                centers[p] = center
                
#        iner_socre(centers, df_, docvecs, threshold, 10)
            
    print("Done in %0.3fs." % (time() - t0))
    return centers, clu_rank.parent
    
def similarity_clustering_time(df_, docvecs, threshold=0.8, repeat=5):
    t0 = time()
    now = datetime.now()
    # Initialize
    pr = []
    centers = {}
    for idx, row in df_.iterrows():
        centers[idx] = docvecs[idx]
        pr.append((idx, 0))
    clu_rank = pd.DataFrame(pr, df_.index, columns = ['parent', 'rank'])

    breaked = True
    i = 0
    while breaked:
        i = i + 1
        cnt = 0
        breaked = False
        p_unique = clu_rank.parent.unique()
        for key in itertools.combinations(p_unique, 2):
            cnt = cnt + 1
            u_cluster = find(clu_rank, key[0])
            v_cluster = find(clu_rank, key[1])
            vec_sim = cs_similarity(centers[u_cluster], centers[v_cluster])
            if(u_cluster != v_cluster and vec_sim >= threshold):
                cluster = union(clu_rank, u_cluster, v_cluster)
                centers[cluster] = merge_similarity(centers[u_cluster], centers[v_cluster])
                breaked = True
                break
        print("Iterated: %d, passed cnt: %d"%(i, cnt))
        if(i - 1 > repeat):
            break

    for idx, row in clu_rank.iterrows():
        find(clu_rank, idx)

    for idx, row in clu_rank.iterrows():
        center = centers[row.parent]
        similarity = cs_similarity(docvecs[idx], center)
        if(similarity<threshold):
            clu_rank.set_value(idx, 'parent', idx)

    for p in clu_rank.parent.unique():
        cluster = clu_rank[clu_rank.parent==p]
        center = docvecs[p]
        for i, c in cluster.iterrows():
            center = merge_similarity(center, docvecs[i])
        centers[p] = center
                
    print("Done in %0.3fs." % (time() - t0))
    return centers, clu_rank.parent
    
    
def iner_socre(centers, df, docvecs, threshold, cnt_threshold):
    scores = similarity_iner_score(centers, df, docvecs, threshold)
    size_1 = scores[scores.cnt==1]
    countby = scores[scores.cnt>cnt_threshold]
    print "total:", len(scores), ", size_1:",len(size_1), ", countby:", len(countby)
    ss = countby.sum(axis=0)
    print "distance:", ss['distance'] * 100
    print "variance:", ss['variance']
    print "similarity:", (ss['similarity'] * 100)/len(countby)
    print "cohesion:", ss['cohesion']
    print "in_threshold:", ss['in_threshold']
    print "time_mean:", ss['time_mean']
    print "time_v:", ss['time_v']
    print "portion:", ss['portion']
    return scores, countby
    
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
    parent = df.loc[idx]['parent']
    if(idx == parent): return idx

    newParent = find(df, parent)
    df.set_value(idx, 'parent', newParent)
    return newParent

def union(df, u_parent, v_parent):
    if (u_parent == v_parent): return
    
    u_rank = df.loc[u_parent]['rank']
    v_rank = df.loc[v_parent]['rank']
    if(u_rank > v_rank):
        df.set_value(v_parent, 'parent', u_parent)
        return u_parent
    else:
        df.set_value(u_parent, 'parent', v_parent)
        if(u_rank == v_rank):
            df.set_value(v_parent, 'rank', v_rank + 1)
        return v_parent

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