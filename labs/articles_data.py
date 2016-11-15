# -*- coding: utf-8 -*- 

import numpy as np
import pandas as pd
import datetime
from gensim.models import Word2Vec
import cnouns as cn
import cPickle as pickle
from multiprocessing import Pool
from time import time
import os
import re
from tqdm import tqdm

def is_dirty_article(title, content, min_len = 100):
    if(len(content) < min_len):
        return True
    
    dh = get_dirty_headlines()
    result = re.match(r"[^[]*\[([^]]*)\]", title)
    if result:
        if result.groups()[0] in dh:
            return True
        
    return False

def get_dirty_headlines():
    return [u"경향포토", u"오늘의 날씨"]

def get_target_cate():
    return [u"정치", u"사회", u"경제", u"과학"]

def find_recent_articles(collection, catelist_path, target_time):
    articles = collection

    categories = pd.read_pickle(catelist_path)

    article_list = []
    d = target_time - datetime.timedelta(days=7)
    for article in articles.find({"publishedAt": {"$gt": d, "$lt": target_time}}).sort("publishedAt"):
        if(not is_dirty_article(article['title'], article['content'])):
            article_list.append(article)

    articles_df = pd.DataFrame.from_dict(article_list)

    new_categories = []

    for idx, row in articles_df.iterrows():
        category = categories[categories.category==row.category]
        if(len(category) > 0):
            new_categories.append(category['name'].iloc[0])
        else:
            new_categories.append('none')

    articles_df['cate'] = new_categories
    target_list = get_target_cate()
    
    return articles_df[articles_df['cate'].isin(target_list)].reset_index(drop=True)

class Sentences(object):
    def __init__(self, dirname, size):
        self.dirname = dirname
        self.size = size
 
    def __iter__(self):
        for fname in os.listdir(self.dirname)[:self.size]:
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
                                
def makeDataset(collection, target_dir, corpus_path, batch_size=5000, workers=4, tokenizer=cn.tokenizer):
    articles = collection.find()
    
    articles_list = []
    for article in articles:
        articles_list.append(article)
    articles_df = pd.DataFrame.from_dict(articles_list)
    print("Number of articles - %d" % len(articles_df))
    
    corpus_df = pd.read_pickle(corpus_path)
    print("Number of corpus - %d" % len(corpus_df))
    
    corpus_words = [row[1] for row in corpus_df.iteritems()]
    articles_words = [aricle['title'] + ' ' + aricle['content'] for idx, aricle in articles_df.iterrows()]
    words = corpus_words + articles_words
    corpus_words = []
    articles_words = []
    print("Number of words - %d" % len(words))
    
    batchs = [words[i:i + batch_size] for i in xrange(0, len(words), batch_size)]
    print("Number of batchs - %d" % len(batchs))
    
    # p = Pool(workers)
    for idx, batch in tqdm(enumerate(batchs)):
        t0 = time()
        # tokens = p.map(tokenizer, batch)
        tokens = [tokenizer(b) for b in batch]
        f = open("%s/%d"%(target_dir, idx), "w")
        f.write("\n".join(tokens).encode('utf8'))
        f.close()
        #print("Batch#%d - tokenizer took %f sec"%(idx, time() - t0))
        
    return len(batchs)
        
def trainWord2Vec(src, dest, size=50):
    sentences = Sentences(src, size)
    w2v = Word2Vec(sentences)
    w2v.save_word2vec_format(dest)