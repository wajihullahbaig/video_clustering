# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:29:03 2024

@author: Acer
"""
from clustering import KMeansCluster,DBScanCluster
from array_utils import find_kneed, visualize_kneed
import pandas as pd

def do_kmeans(df, pca_data, visualize_graphs):
    # Compute clusters
    kmc = KMeansCluster()

    sse = kmc.compute_sse(pca_data)  
    if visualize_graphs:
        kmc.visualize_sse(sse)

    optimal_k = find_kneed(sse)*3
    if visualize_graphs:
        visualize_kneed(optimal_k, sse)

    optimal_k = optimal_k
    labels = kmc.compute(pca_data, optimal_k)
    #save cluster assignments for each observation
    df['labels'] = pd.Series(labels, index=df.index)
    if visualize_graphs:
        # Make sure you have 2D data for visualizations
        kmc.visualize_clusters(df)
            
def do_dbscan(df,pca_data,visualize_graphs):
    # Compute the cluster
    dbc = DBScanCluster()
    labels = dbc.compute(pca_data)
    #save cluster assignments for each observation
    df['labels'] = pd.Series(labels, index=df.index)
    if visualize_graphs:
        # Make sure you have 2D data for visualizations
        dbc.visualize_clusters(df)