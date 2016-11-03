# -*- coding: utf-8 -*- 

import numpy as np
import pandas as pd
from pymongo import MongoClient
import datetime

def find_recent_articles(collection_name='articles'):
    client = MongoClient('mongodb://localhost:27017/somanews')
    db = client.get_database('somanews')
    articles = db.get_collection(collection_name)

    categories = pd.read_pickle('../datastore/category2.p')

    article_list = []
    d = datetime.datetime.now() - datetime.timedelta(days=7)
    for article in articles.find({"publishedAt": {"$gt": d}}).sort("publishedAt"):
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
    target_list = [u"정치", u"사회", u"과학", u"경제"]
    
    return articles_df[articles_df['cate'].isin(target_list)]