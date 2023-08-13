import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
import networkx as nx


def choosing_k(scores_basic, scores, top_cluster_proportion_threshold, n_init, draw):
    #k is chosen using a silhouette method
    #since we only care about the top cluster -- that with name variations -- we only consider the silhouette score of the top cluster
    #max_k: "One commonly used heuristic is to set the maximum number of clusters to be equal to the square root of half the number of data points. 
    #Its primary goal is to prevent the model from creating too many clusters when the number of data points are limited."
    #I decide to multiply by two, as it is essential to have many clusters, given our focus on the topcluster accuracy.
    max_k = int(2 * (0.5 * len(scores_basic))**0.5)
    K = range(2, max_k)
    silhouette_top_cluster_avgs = []
    for k in K:
        #n_init: "controls the number of times the KMeans algorithm will be run with different centroid seeds." Increased accuracy.
        kmeans = KMeans(n_clusters=k, n_init=n_init).fit(scores)
        labels = kmeans.labels_
        top_cluster_label = labels[scores_basic.index(max(scores_basic))]
        silhouette_samples_all = silhouette_samples(scores, labels)
        silhouette_samples_top_cluster = silhouette_samples_all[labels == top_cluster_label]
        silhouette_avg_top_cluster = silhouette_samples_top_cluster.mean()
        #only allow k where the fraction of pairs in the top cluster is greater than some predefined proportion
        silhouette_top_cluster_avgs.append(np.where(len(silhouette_samples_top_cluster) / len(silhouette_samples_all) > top_cluster_proportion_threshold, silhouette_avg_top_cluster, 0))
    
    k = 2 + silhouette_top_cluster_avgs.index(max(silhouette_top_cluster_avgs))
    
    if draw:
        plt.plot(K, silhouette_top_cluster_avgs, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Silhouette score for top cluster')
        plt.show()
    return k


def edge_threshold(scores_basic, scores, k, n_init, min_similarity_threshold):
    #the threshold is the lowest similarity in the top cluster
    kmeans = KMeans(n_clusters=k, n_init=n_init).fit(scores)
    labels = kmeans.labels_

    top_cluster_label = labels[scores_basic.index(max(scores_basic))]
    s = [x for x, l in zip(scores_basic, labels) if l == top_cluster_label]
    #it is possible that there are no name variations. therefore, have a minimum threshold for similarity
    return max([min(s), min_similarity_threshold])


def graph_clusters(data, threshold, draw):
    #we can use graph theory to find name variations
    #if the nodes a and b have an edge, as do b and c, but a and c do not, then we can say that a and c are name variations
    G = nx.Graph()
    products = list(set([prod for pair, _ in data for prod in pair]))
    for product in products:
        G.add_node(product)

    #Add edges for product pairs with similarity above the threshold
    for (prod1, prod2), similarity in data:
        if similarity >= threshold:
            G.add_edge(prod1, prod2)

    clusters = list(nx.connected_components(G))
    clusters = [tuple(x) for x in clusters if len(x) > 1]

    if draw:
        nx.draw(G, with_labels=True)
        plt.show()

    return clusters