# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:02:08 2024

@author: Acer
"""

import pandas as pd
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans, DBSCAN 
import matplotlib.pyplot as plt
from numpy import ndarray
import matplotlib.pyplot as plt
import seaborn as sns


class Cluster(ABC):
    @abstractmethod
    def compute(self,data: pd.DataFrame) -> list:
        pass
    
    
    
class KMeansCluster(Cluster):
    def __init__(self):
        #initialize kmeans parameters
        self.kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "random_state": 1,
        }
    def compute_sse(self,df: pd.DataFrame) -> list:
        #create list to hold SSE (sum of squared errors) values for each k
        sse = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, **self.kmeans_kwargs)
            kmeans.fit(df)
            sse.append(kmeans.inertia_)
        return sse
    
    def visualize_sse(self, sse:list)->None:
        #visualize results
        plt.plot(range(1, 11), sse)
        plt.xticks(range(1, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.show()
        
    def visualize_clusters(self,df:pd.DataFrame())-> None:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='labels', palette='viridis')
        plt.title('PCA and K-means Clustering Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster')
        plt.show()
     
    def compute(self,data: ndarray, optimal_k) -> list:
        #instantiate the k-means class, using optimal number of clusters
        kmeans = KMeans(init="random", n_clusters=optimal_k, n_init=10, random_state=1)
        
        #fit k-means algorithm to data
        kmeans.fit(data)
        
        return kmeans.labels_
      

class DBScanCluster(Cluster):
    def __init__(self):
        pass
        
    def visualize_clusters(self,df:pd.DataFrame())-> None:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='labels', palette='viridis')
        plt.title('PCA and  Clustering Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster')
        plt.show()
     
    def compute(self,data: ndarray) -> list:
       db_default = DBSCAN(eps = 0.5).fit(data) 
       return db_default.labels_      
    