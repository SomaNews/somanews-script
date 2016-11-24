from time import mktime
import math
import datetime
import pandas as pd


d_factor = {
    "portionRank": 1,
    "deltaTimeRank": 1.5,
    "cohesionRank": 0,
    "portion": 10,
    "deltaTime": 10,
    "cohesion": 1
}

def calc_ntc(portion, deltalTime, cohesion):
    return n * 10 + c * 1 + 10 / t

def calc_rank(row, factor):
    ntc = factor['portion'] * row['portion'] + factor['deltaTime'] / row['deltaTime'] + factor['cohesion'] * row['cohesion']
    rank = factor['portionRank'] * row['portionRank'] + factor['deltaTimeRank'] * row['deltaTimeRank'] + factor['cohesionRank'] * row['cohesionRank']
    return ntc, rank

def calc_issue_rank(clusters, factor=d_factor):
    total_count = 0
    for cluster in clusters:
        total_count = total_count + cluster['count']

    now = datetime.datetime.now()
    for cluster in clusters:
        articles = cluster['articles']
        cluster_size = len(articles)

        base_time = 0
        cluster_time = 0
        for article in articles:
            dt = (now - article['publishedAt']).total_seconds()
            cluster_time = cluster_time + dt
            if(dt > base_time):
                base_time = dt

        t = (cluster_time / cluster_size) / base_time
        n = cluster['count'] / float(total_count)
        c = cluster['cohesion']

        cluster['portion'] = n
        cluster['deltaTime'] = t

    cdf = pd.DataFrame(clusters, columns=['cluster', 'portion', 'deltaTime', 'cohesion', 'count'])
    sort_cdf = cdf.sort_values('portion', ascending=False)
    sort_cdf['portionRank'] = range(len(sort_cdf))
    sort_cdf = sort_cdf.sort_values('cohesion', ascending=False)
    sort_cdf['cohesionRank'] = range(len(sort_cdf))
    sort_cdf = sort_cdf.sort_values('deltaTime', ascending=True)
    sort_cdf['deltaTimeRank'] = range(len(sort_cdf))

    ranks = {}
    for idx, row in sort_cdf.iterrows():
        ntc, rank = calc_rank(row, factor)
        ranks[row.cluster] = float(rank)

    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1])
    rank = len(sorted_ranks)
    rank_of_rank = {}
    for tup in sorted_ranks:
        k = tup[0]
        v = tup[1]
        rank_of_rank[k] = rank
        rank = rank - 1
        
    for cluster in clusters:
        cluster['rank'] = ranks[cluster['cluster']]
        
    sort_cdf = sort_cdf.sort_values('cluster', ascending=True)
    sort_cdf['rank'] = [t[1] for t in sorted(ranks.items(), key=lambda x: x[0])]
    sort_cdf['rank_desc'] = [t[1] for t in sorted(rank_of_rank.items(), key=lambda x: x[0])]
    return clusters, sort_cdf
