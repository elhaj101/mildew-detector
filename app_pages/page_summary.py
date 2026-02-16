import streamlit as st
import matplotlib.pyplot as plt

def page_summary_body():
    st.write("### Project Summary")
    
    st.info(
        f"**General Information**\n\n"
        f"Powdery mildew is a widespread fungal disease that poses a significant threat to agricultural "
        f"productivity, affecting a diverse range of plant species including cherry trees, grapes, roses, "
        f"and various vegetables. The disease is characterized by distinctive white or grayish powdery spots "
        f"that appear on leaves, stems, and sometimes fruit, caused by various species of fungi in the order "
        f"Erysiphales. These fungal pathogens thrive in warm, dry climates with high humidity, making cherry "
        f"plantations particularly vulnerable during certain seasons.\n\n"
        f"For Farmy & Foods, a leading agricultural company with extensive cherry plantations across multiple "
        f"farms, powdery mildew represents both a quality control challenge and a significant operational burden. "
        f"Currently, the detection process relies entirely on manual inspection, where trained employees spend "
        f"approximately 30 minutes per tree examining leaf samples to identify signs of infection. Once detected, "
        f"treatment involves applying specialized fungicidal compounds, which takes an additional minute per tree. "
        f"While this manual approach can be effective, it is inherently time-consuming, labor-intensive, and "
        f"difficult to scale across thousands of trees spread over vast geographical areas.\n\n"
        f"This project leverages the power of machine learning and computer vision to revolutionize the detection "
        f"process. By training a Convolutional Neural Network (CNN) on thousands of labeled cherry leaf images, "
        f"we have developed an automated system capable of instantly identifying powdery mildew infection from a "
        f"simple photograph. This solution dramatically reduces inspection time from 30 minutes to mere seconds, "
        f"enables field workers to make immediate treatment decisions using mobile devices, and provides a scalable "
        f"framework that can be extended to detect pests and diseases in other crops. The result is improved crop "
        f"quality, reduced labor costs, faster response times to disease outbreaks, and ultimately, a more efficient "
        f"and sustainable agricultural operation."
    )

    st.write(
        f"**Project Dataset**\n"
        f"* The dataset is sourced from Kaggle and contains over 4000 images of healthy and infected cherry leaves."
    )

    st.write(
        f"**Business Requirements**\n"
        f"* 1 - Visually differentiate between healthy and powdery mildew leaves using an image montage.\n"
        f"* 2 - Accurately predict whether a given leaf is healthy or infected using a deep learning model."
    )

    st.success(
        f"**Project Outcomes**\n"
        f"* A dashboard for visual inspection.\n"
        f"* A deep learning model for instant diagnosis."
    )
