import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import itertools
import random

def page_visualizer_body():
    st.write("### Leaf Visualizer")
    st.info(
        f"**Business Requirement 1**\n"
        f"* Visually differentiate healthy and diseased leaves."
    )
    
    version = 'v1'

    if st.checkbox("Image Montage"): 
        st.write("* Click the 'Create Montage' button to view a random set of images.")
        # Use a small subset for Heroku to stay under slug limit
        my_data_dir = 'data/montage'
        # Fallback for local development if montage dir isn't created yet
        if not os.path.exists(my_data_dir):
            my_data_dir = 'data/cherry-leaves'
        
        if os.path.exists(my_data_dir):
            labels = [f for f in os.listdir(my_data_dir) if os.path.isdir(os.path.join(my_data_dir, f))]
            label_to_display = st.selectbox(label="Select Label", options=labels, index=0)
            
            if st.button("Create Montage"): 
                image_montage(dir_path=my_data_dir, label_to_display=label_to_display, nrows=3, ncols=3, figsize=(10,10))
        else:
            st.error(f"Data directory not found: {my_data_dir}. Please ensure the dataset is uploaded to the server.")
            

def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15,10)):
    sns.set_style("white")
    labels = ['healthy', 'powdery_mildew']
    if label_to_display not in labels:
        st.error("The selected label doesn't exist.")
        return
    

    images_list = os.listdir(os.path.join(dir_path, label_to_display))
    if len(images_list) < nrows * ncols:
        st.error(
            f"Not enough images to create montage. \n"
            f"You requested {nrows * ncols} images, but only have {len(images_list)}."
        )
        return
    
    img_idx = random.sample(images_list, nrows * ncols)
    
    # Plotting
    list_rows = range(0, nrows)
    list_cols = range(0, ncols)
    plot_idx = list(itertools.product(list_rows, list_cols))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for x in range(0, nrows*ncols):
        img = imread(os.path.join(dir_path, label_to_display, img_idx[x]))
        img_shape = img.shape
        axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
        axes[plot_idx[x][0], plot_idx[x][1]].set_title(f"Width {img_shape[1]}px x Height {img_shape[0]}px")
        axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
        axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
    plt.tight_layout()
    st.pyplot(fig=fig)
