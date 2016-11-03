import matplotlib.pyplot as plt
import itertools
import operator
import pandas as pd

def test_print(cluster, km, inp, df):
    center_idx = find_center_article(km, cluster, inp)
    print(center_idx, df.loc[center_idx].title)

    target_cluster = df[df.cluster==cluster]
    print("size ", len(target_cluster))
    print(target_cluster.title)

def match_cluster_topic(train_df, is_cluster, topics, num_clusters):
    if(is_cluster):
        print("Cluster -> Topic")
        criteria = 'cluster'
        target = 'topic_idx'
    else:
        print("Topic -> Cluster")
        criteria = 'topic_idx'
        target = 'cluster'
        
    total_doc = 0    
    total_accuracy = 0    
    for i in range(0, num_clusters):
        criteria_set = train_df[train_df[criteria]==i]
        target_count = {}
        for j in range(0, num_clusters):
            target_set = criteria_set[criteria_set[target]==j]
            target_count[j] = len(target_set)
        max_target_idx = max(target_count.iteritems(), key=operator.itemgetter(1))[0]
        accuracy = 100*target_count[max_target_idx]/float(len(criteria_set))
        total_accuracy = total_accuracy + accuracy
        if(is_cluster):
            topic_str = topics[max_target_idx]
        else:
            topic_str = topics[i]
        print("#%d -> #%d Accuracy is %.4d/%.4d = %.10f \t %s" % (i, max_target_idx, target_count[max_target_idx], len(criteria_set), accuracy, topic_str))
        total_doc = total_doc + target_count[max_target_idx]
        
    print("%.4f" % (total_accuracy/num_clusters))
    print("%.4f" % (100 * total_doc/len(train_df)))
    
    
def print_topics():
    for idx in topics:
        topic = topics[idx]
        print("%.4d - %s" % (len(train_df[train_df.topic==topic]), topic)) 
        
def get_cartesian(df, num_clusters):
    cartesian = itertools.product(range(num_clusters), range(num_clusters))

    temp = {
        'cluster': [],
        'topic_idx': [],
        'counts': []
    }
    for c, t in cartesian:
        tmp_clusters = df[df.cluster==c]
        tmp_topics = tmp_clusters[tmp_clusters.topic_idx==t]
        temp['cluster'].append(c)
        temp['topic_idx'].append(t)
        temp['counts'].append(len(tmp_topics))

    results = pd.DataFrame(temp)
    results = results[results.counts!=0]
    
    return results

def show_cartesian(df, num_clusters):
    results = get_cartesian(df, num_clusters)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(results.topic_idx, results.cluster, 'ro')
    for index, row in results.iterrows():
        x = row['topic_idx']
        y = row['cluster']
        ax.annotate('  %d' % row['counts'], xy=(x,y), textcoords='data')
    plt.axis([-1, 7, -1, 7])
    plt.xlabel('Topic')
    plt.ylabel('Cluster')

    plt.grid()
    plt.show()
    
def show_cluster(df, clusters):
    for n in clusters:
        print("#%.2d - %d" % (n, len(df[df.cluster==n])))

def test_similar(test_idx, docvecs, df, threadsold=0.5, is_last = False):
    sims = docvecs.most_similar(test_idx, topn=len(docvecs))
    most_sims = [s for s in sims if s[1]>=threadsold]
    last_sims = most_sims[len(most_sims)-5:]
    print("most_similar -", len(most_sims))
    print("target - ", df.loc[test_idx].title)
    print("")
    if(is_last):
        for s in last_sims:
            print(df.loc[s[0]].title)
    else:
        for s in most_sims:
            print(df.loc[s[0]].title)
            
def sort_count(df, clusters):
    sized = [(n, len(df[df.cluster==n])) for n in clusters]
    sorted_cluster = sized[:]
    sorted_cluster.sort(key=lambda x: -x[1])
    return sorted_cluster

def find_center_article(km, cluster, inp):
    center = km.cluster_centers_[cluster]

    dists = [vdist(v, center) for v in inp]
    i = dists.index(min(dists))
    return i

def vdist(v, center):
    dv = v - center
    return sum(dv * dv)