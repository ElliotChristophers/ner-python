from k_means_clustering import *
from semantic_similarity import standalone_similarity
from ner_models import flair_model
import json

file = 0
products_n_times = 1
top_cluster_proportion_threshold = 0
n_init = 1000
min_similarity_threshold = 0.9
draw = True


files = ['aapl_q1_2023', 'abb_q3_2022', 'abbv_q3_2022', 'abt_q4_2022']
with open(fr'{files[file]}.json') as f:
    data = json.load(f)
transcript = data["transcript"]

#iterate through ner model multiple times, aiming to catch all positives
products = []
for i in range(products_n_times):
    print(i+1)
    p = flair_model(transcript)
    for q in p:
        products.append(q)
products = list(set(products))
print(products)

#similarity score. 
#standalone is chosen rather than context_driven:
#with context_driven all products are given the same similarity, as the context, rather than the product names, is overweighted
data = standalone_similarity(products)
print(data)

scores_basic = [score for (_, score) in data]
scores = np.array(scores_basic).reshape(-1, 1)
#choose an appropriate number of clusters for k-means clustering
k = choosing_k(scores_basic, scores, top_cluster_proportion_threshold, n_init, draw)
#given the number of clusters, we find a similarity threshold in order for two products (nodes) to be name variations (edge)
threshold = edge_threshold(scores_basic, scores, k, n_init, min_similarity_threshold)
print(k, threshold)
#prints name variations, plots the nodes, with edges for name variations
print(graph_clusters(data, threshold, draw))