# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:37:32 2018

@author: João Pina
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from matplotlib import style
style.use("ggplot")

pca = PCA(n_components=2)

best_centroids_for_silhouette = []
best_centroids_for_davies_bouldin = []
best_silhouette_scores = []
best_davies_bouldin_scores=[]

for i in range(0,10):
    fileName = "base_aps-failure/base_aps-failure_unsupervised-mining_sample_" + str(i) + ".csv"
    dataset = pd.read_csv(fileName, sep=',', engine='python')
    
    dataset = StandardScaler().fit_transform(dataset)
    principalComponents = pca.fit_transform(dataset)
    dataset = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
    
    best_centroids_silhouette = 0
    best_score_silhouette = 0
    
    best_centroids_davies_bouldin = 0
    best_score_davies_bouldin = 50
    
    centroid_count = []
    silhouette_scores = []
    davies_bouldin_scores = []
    
    for i in range(2,11):
        centroid_count += [i]
        print("Number of centroids: ", i)
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(dataset)
        
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        kmeansLabels = pd.DataFrame(labels)
        
        #print(centroids)
        #print(labels)
        
        if(silhouette_score(dataset, labels) > best_score_silhouette):
            best_score_silhouette = silhouette_score(dataset, labels)
            best_centroids_silhouette = i
        if(davies_bouldin_score(dataset, labels) < best_score_davies_bouldin):
            best_score_davies_bouldin = davies_bouldin_score(dataset, labels)
            best_centroids_davies_bouldin = i
        silhouette_scores += [silhouette_score(dataset, labels)]
        davies_bouldin_scores += [davies_bouldin_score(dataset, labels)]
        print("Silhouette Score = ", silhouette_score(dataset, labels))
        print("Davies-Bouldin Score = ", davies_bouldin_score(dataset, labels))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(dataset['pc1'],dataset['pc2'], c=kmeansLabels[0],s=50)
        ax.set_title('K-Means Clustering')
        ax.set_xlabel('pc1')
        ax.set_ylabel('pc2')
        plt.colorbar(scatter)
        plt.show()
    
    print("----------------------------------------------------------------")
    print("Best number of centroids for Silhouette: ", best_centroids_silhouette)
    best_centroids_for_silhouette += [best_centroids_silhouette]
    print("Best score for silhouette: ", best_score_silhouette)
    best_silhouette_scores += [best_score_silhouette]
    print()
    print("Best number of centroids for Davies-Bouldin: ", best_centroids_davies_bouldin)
    best_centroids_for_davies_bouldin += [best_centroids_davies_bouldin]
    print("Best score for Davies-Bouldin: ", best_score_davies_bouldin)
    best_davies_bouldin_scores += [best_score_davies_bouldin]
     
    
    
    
    plt.plot(centroid_count, silhouette_scores)
    plt.xlabel('Centroids')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Coefficient for Centroids from 2 to 10')
    plt.show()
    
    
    plt.plot(centroid_count, davies_bouldin_scores)
    plt.xlabel('Centroids')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Score for Centroids from 2 to 10')
    plt.show()
    
    plt.plot(centroid_count, silhouette_scores, '.-', color="r", label="Silhouette Coefficient")
    plt.plot(centroid_count, davies_bouldin_scores, '.-', color="b", label="Davies-Bouldin Score")
    plt.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))	
    plt.xlabel('Centroids')
    plt.ylabel('Scores')
    plt.title('Evaluation for Centroids from 2 to 10')
    plt.show()

print("BEST OF THE BEST FOR SILHOUETTE")
print("Centroids: ", np.array(best_centroids_for_silhouette).mean())
print("Silhouette Score: ", np.array(best_silhouette_scores).mean())
print()
print("BEST OF THE BEST FOR DAVIES-BOULDIN")
print("Centroids: ", np.array(best_centroids_for_davies_bouldin).mean())
print("Davies-Bouldin Score: ", np.array(best_davies_bouldin_scores).mean())

