import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

def page_mildew_detector_body():
    st.write("### Mildew Detector")
    st.info(
        f"**Business Requirement 2**\n"
        f"* The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew."
    )

    st.write(
        f"* You can download a set of infected and healthy leaves for live prediction from "
        f"[here](https://www.kaggle.com/codeinstitute/cherry-leaves)."
    )

    st.write("---")

    st.write(
        f"The detector can reliably give you the correct answer only when the leaf is "
        f"infected with Mildew disease that causes white spots or marks. "
        f"This detector is not for damaged, burnt, or infested leaves."
    )

    images_buffer = st.file_uploader('Upload leaf images. You may select more than one.', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
   
    if images_buffer is not None:
        if len(images_buffer) > 0:
            make_prediction(images_buffer)

def make_prediction(images_buffer):
    model_path = 'out/modeling/mildew_detector_model.keras'
    
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return

    df_report = pd.DataFrame([])
    for image in images_buffer:
        
        img_pil = (Image.open(image)).convert('RGB')
        st.info(f"Leaf Image: **{image.name}**")
        img_array = np.array(img_pil.resize((100, 100))) # Resize to match training
        img_array = np.expand_dims(img_array, axis=0) / 255.0 # Normalize
        
        pred_prob = model.predict(img_array)[0][0]
        
        pred_class = 'Powdery Mildew' if pred_prob > 0.5 else 'Healthy'
        prob_per = pred_prob if pred_prob > 0.5 else 1 - pred_prob
        
        st.image(img_pil, caption=f"Prediction: {pred_class} ({prob_per*100:.2f}%)")
        
        df_report = pd.concat([df_report, pd.DataFrame({"Name": [image.name], 'Result': [pred_class], 'Probability': [prob_per]})], ignore_index=True)
        
    if not df_report.empty:
        st.success("Analysis Report")
        st.table(df_report)
        st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)

def download_dataframe_as_csv(df):
    csv = df.to_csv(index=False).encode('utf-8')
    from base64 import b64encode
    b64 = b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="report.csv" target="_blank">Download Report</a>'
    return href
