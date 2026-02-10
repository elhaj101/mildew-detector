import streamlit as st
import matplotlib.pyplot as plt

def page_summary_body():
    st.write("### Project Summary")
    
    st.info(
        f"**General Information**\n"
        f"* Powdery mildew is a fungal disease that affects a wide range of plants. "
        f"The disease manifests as white powdery spots on leaves and stems.\n"
        f"* The project is designed to help Cherry Leaf farm managers to detect this disease instantly."
    )

    st.write(
        f"**Project Dataset**\n"
        f"* The dataset is sourced from Kaggle and contains over 4000 images of healthy and infected cherry leaves."
    )

    st.write(
        f"**Business Requirements**\n"
        f"* 1 - Visually differentiate between healthy and powdery mildew leaves.\n"
        f"* 2 - Accurately predict whether a given leaf is healthy or infected."
    )

    st.success(
        f"**Project Outcomes**\n"
        f"* A dashboard for visual inspection.\n"
        f"* A deep learning model for instant diagnosis."
    )
