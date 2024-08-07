# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2 
import pandas as pd
from sklearn.decomposition import PCA
import imageio
import matplotlib.pyplot as plt
import numpy as np
import shutil

from features import HOGFeature,LBPFeature
from video_reader import VideoReader
from image_utils import resize_with_aspect_ratio
from do_clustering import do_kmeans,do_dbscan,do_afinity_propogation



    
vr = VideoReader("big_buck_bunny_360p_20mb.webm")
meta = vr.video_meta()
print(meta)

feature = LBPFeature()
feature = HOGFeature()
feature_vector = []
df = pd.DataFrame(columns=["frame_no"])

visualize_graphs = True
show_frames = True
n_frames_to_process = meta["fps"]*30
nth_frame= 0
for frame in vr.get_frame():
    if frame is not None and nth_frame < n_frames_to_process:
        # Take one channel image, gives you a pseudo grey image
        print(f"Processing frame no: {nth_frame}")
        frame = frame[:,:,0]
        # Resizing the frame, faster processing
        # Make sure this is same for win_size in HoG class
        frame = resize_with_aspect_ratio(frame,width=128)
                        
        feature_vector.append(feature.compute(frame))
        nth_frame +=1
        if show_frames:
            cv2.imshow("Resized Video frame",frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
if visualize_graphs:
    cv2.destroyAllWindows()    

# Show a plot of features
# Plot feature vectors
plt.bar(np.arange(len(feature_vector[0])),feature_vector[0],width=5,align='center')
plt.show()
    
# Prepare data for clustering
# We use 2 components so we can also visualize the clusters
pca = PCA(n_components=2)
pca_data = pca.fit_transform(feature_vector)
df["frame_no"] = list(range(len(pca_data)))
df["PCA1"] = pca_data[:,0]
df["PCA2"] = pca_data[:,1]

# Do the clustering
#do_kmeans(df, pca_data, visualize_graphs)
#do_dbscan(df, pca_data, visualize_graphs)
do_afinity_propogation(df, pca_data, visualize_graphs)


# Close video read it again. Opencv rewind does not seem to work
del vr
vr = VideoReader("big_buck_bunny_360p_20mb.webm")
import os
frame_count = 0 
df["path"] = None
root_folder = "clusters/"
if os.path.exists(root_folder):
    shutil.rmtree("clusters/")

for frame in vr.get_frame():
    if frame is not None and frame_count in df["frame_no"].values:
        v = df.loc[frame_count]
        if not os.path.exists(root_folder+str(int(v["labels"]))):
            os.makedirs(root_folder+str(int(v["labels"])))
        
            
        file_path = root_folder+str(int(v["labels"]))+"/"+str(int(v["frame_no"]))+'.jpg'
        cv2.imwrite(file_path,frame)
        df.loc[frame_count,["path"]] = file_path
        frame_count += 1 

# Lets use some frames to create animated GIF
df_gif_frames = df.drop_duplicates('labels', keep='last')

images = []
for filename in df_gif_frames["path"].values:
    images.append(imageio.imread(filename))
imageio.mimsave('movie.gif', images,fps=1)


