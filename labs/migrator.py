from pymongo import MongoClient

def migrate():
    client = MongoClient('mongodb://ssomanews:ssomanews1029@localhost/somanews', 27017)
    db = client.get_database('somanews')

    for clusterType in 'ab':
        articlesDB = db.get_collection('%sarticles' % clusterType)
        clusterDB = db.get_collection('%sclusters' % clusterType)

        clusters = list(clusterDB.find())
        articles = []
        for cluster in clusters:
            clusterID = cluster['cluster']
            clusteredAt = cluster['clusteredAt']
            for article in cluster['articles']:
                if '_id' in article:
                    article['article_id'] = article['_id']
                    del article['_id']
                article['cluster'] = clusterID
                article['clusteredAt'] = clusteredAt
                articles.append(article)
        articlesDB.delete_many({})
        articlesDB.insert_many(articles)
        
    client.close()
