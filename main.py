# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2 
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from features import HOGFeature,LBPFeature
from clustering import KMeansCluster
from video_reader import VideoReader
from image_utils import resize_with_aspect_ratio
from array_utils import find_kneed, visualize_kneed


vr = VideoReader("big_buck_bunny_360p_20mb.webm")
print(vr.video_meta())

feature = LBPFeature()

#feature = HOGFeature()
feature_vector = []
df = pd.DataFrame(columns=["frame_no"])

visualize_graphs = True
show_frames = True

for frame in vr.get_frame():
    if frame is not None:
        # Take one channel image, gives you a pseudo grey image
        frame = frame[:,:,0]
        frame = resize_with_aspect_ratio(frame,width=160)
        feature_vector.append(feature.compute(frame))
        if show_frames:
            cv2.imshow("Resized Video frame",frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
if visualize_graphs:
    cv2.destroyAllWindows()        
# Prepare data for clustering
# We use 2 components so we can also visualize the clusters
pca = PCA(n_components=2)
pca_data = pca.fit_transform(feature_vector)
df["frame_no"] = list(range(len(pca_data)))
df["PCA1"] = pca_data[:,0]
df["PCA2"] = pca_data[:,1]

# Compute clusters
kmc = KMeansCluster()

sse = kmc.compute_sse(pca_data)  
if visualize_graphs:
    kmc.visualize_sse(sse)

optimal_k = find_kneed(sse)
if visualize_graphs:
    visualize_kneed(optimal_k, sse)

optimal_k = optimal_k*2
labels = kmc.compute(pca_data, optimal_k)
#save cluster assignments for each observation
df['labels'] = pd.Series(labels, index=df.index)
if visualize_graphs:
    # Make sure you have 2D data for visualizations
    kmc.visualize_clusters(df)

