from time import mktime
import math
import datetime

def ntc_rank(n, t, c, n0 = 1, t0 = 10, c0 = 10):
    return n * n0 + c * c0 + t0 / t

def calc_issue_rank(clusters):
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
        cluster['ntc'] = ntc_rank(n, t, c)
    return clusters

def calc_issue_rank_A(clusters):
    ntc = []
    for cluster in clusters:
        n = cluster['count']
        t = cluster['clusteredAt']
        c = cluster['cohesion']
        ntc = ntc_rank(n, t, c)
        
    cluster['ntc'] = ntc
    return clusters