# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2 
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import imageio

from features import HOGFeature,LBPFeature
from clustering import KMeansCluster
from video_reader import VideoReader
from image_utils import resize_with_aspect_ratio
from array_utils import find_kneed, visualize_kneed


vr = VideoReader("big_buck_bunny_360p_20mb.webm")
meta = vr.video_meta()
print(meta)

feature = LBPFeature()

#feature = HOGFeature()
feature_vector = []
df = pd.DataFrame(columns=["frame_no"])

visualize_graphs = True
show_frames = True
n_seconds_to_process = meta["fps"]*50
n_seconds= 0
for frame in vr.get_frame():
    if frame is not None and n_seconds < n_seconds_to_process:
        # Take one channel image, gives you a pseudo grey image
        frame = frame[:,:,0]
        frame = resize_with_aspect_ratio(frame,width=160)
        feature_vector.append(feature.compute(frame))
        n_seconds +=1
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

optimal_k = find_kneed(sse)*2
if visualize_graphs:
    visualize_kneed(optimal_k, sse)

optimal_k = optimal_k
labels = kmc.compute(pca_data, optimal_k)
#save cluster assignments for each observation
df['labels'] = pd.Series(labels, index=df.index)
if visualize_graphs:
    # Make sure you have 2D data for visualizations
    kmc.visualize_clusters(df)


# Close video read it again. Opencv rewind does not seem to work
del vr
vr = VideoReader("big_buck_bunny_360p_20mb.webm")
import os
frame_count = 0 
df["path"] = None
for frame in vr.get_frame():
    if frame is not None and frame_count in df["frame_no"].values:
        v = df.loc[frame_count]
        if not os.path.exists("clusters/"+str(int(v["labels"]))):
            os.makedirs("clusters/"+str(int(v["labels"])))
        file_path = "clusters/"+str(int(v["labels"]))+"/"+str(int(v["frame_no"]))+'.jpg'
        cv2.imwrite(file_path,frame)
        df.loc[frame_count,["path"]] = file_path
        frame_count += 1 

df_gif_frames = df.drop_duplicates('labels', keep='last')
import imageio
images = []
for filename in df_gif_frames["path"].values:
    images.append(imageio.imread(filename))
imageio.mimsave('movie.gif', images,fps=1)