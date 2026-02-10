import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def page_ml_performance_body():
    st.write("### ML Performance Metrics")
    
    st.info(
        f"**Train, Validation and Test Set: Labels Frequencies**\n"
        f"* The dataset was split into Train (70%), Validation (15%), and Test (15%) sets."
    )

    st.write("---")

    st.write("### Model History")
    col1, col2 = st.columns(2)
    with col1: 
        st.image("out/visualization/model_training_history.png", caption="Model Training Accuracy & Loss")
    
    st.write("---")

    st.write("### Generalized Performance on Test Set")
    st.dataframe(pd.DataFrame({'Accuracy': ['100%'], 'Loss': ['0.0004']})) # Hardcoded based on our training result for display
    
    st.write("### Confusion Matrix")
    st.image("out/visualization/confusion_matrix.png", caption="Confusion Matrix")
