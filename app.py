import streamlit as st
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- CONFIG ---
MODEL_PATH = os.path.join('out', 'modeling', 'best_model.keras')
IMG_SIZE = (224, 224)
CLASS_NAMES = ['healthy', 'powdery_mildew']

# --- SIDEBAR NAVIGATION ---
pages = [
    'Home / Project Overview',
    'Data Exploration',
    'Model Performance',
    'Predict on New Images',
    'How It Works & FAQ'
]
page = st.sidebar.selectbox('Navigate', pages)

# --- PAGE 1: HOME ---
if page == 'Home / Project Overview':
    st.title('🍒 Cherry Leaf Mildew Detector')
    st.markdown('''
    ## Project Overview

    The Cherry Tree Mildew Leaf Detector ML project leverages machine learning to identify powdery mildew in cherry tree leaves, a common fungal disease caused by *Podosphaera clandestina*.

    ### Applications of the Project

    - **Precision Agriculture:** Farmers can use the detector to monitor cherry orchards in real-time, enabling targeted interventions such as localized fungicide application, reducing chemical use and environmental impact.
    - **Orchard Management:** The system aids in early detection, allowing growers to manage disease spread, optimize yield, and maintain fruit quality.
    - **Research and Development:** Agricultural researchers can utilize the model to study disease patterns, evaluate resistant cherry varieties, and develop sustainable farming practices.
    - **Smart Farming Integration:** The detector can be integrated with IoT devices and drones for automated monitoring, providing a scalable solution for large orchards.

    ### Countries Where It Can Be Used

    This ML-based detector is applicable in countries with significant cherry production, where powdery mildew poses a threat to crop health. Key regions include:

    - **United States:** A leading cherry producer, particularly in states like Washington, Oregon, and California, where powdery mildew is a recurring issue due to humid conditions.
    - **Turkey:** The world’s largest cherry producer, with extensive orchards that could benefit from automated disease detection to maintain export quality.
    - **Chile:** A major cherry exporter, where early detection can prevent losses in high-value international markets.
    - **European Countries (e.g., Spain, Italy, Germany):** These nations grow cherries in diverse climates, and the technology can help manage mildew in regions with wet or shaded conditions conducive to fungal growth.
    - **China:** With growing cherry cultivation, particularly in Shandong and Liaoning provinces, the detector can support large-scale production and quality control.

    The project is particularly valuable in countries with intensive cherry farming and where labor-intensive manual inspections are costly or impractical.

    ### Why It Is Useful

    The Cherry Tree Mildew Leaf Detector ML project addresses critical challenges in cherry production:

    - **Early Detection:** Powdery mildew can spread rapidly, reducing fruit quality and yield. Early identification enables timely treatment, minimizing crop losses.
    - **Cost Efficiency:** Automated detection reduces the need for manual inspections, saving labor costs and enabling farmers to focus resources on treatment rather than diagnosis.
    - **Environmental Benefits:** By pinpointing affected areas, the system supports precise fungicide application, reducing chemical overuse and environmental harm.
    - **Scalability:** The technology can be deployed across small family farms to large commercial orchards, making it versatile for various scales of operation.
    - **Economic Impact:** By protecting crop yields and quality, the detector helps secure farmers’ income and supports food security in cherry-dependent regions.

    ### Past Methods for Powdery Mildew Detection

    Historically, detecting powdery mildew in cherry trees relied on labor-intensive and subjective methods:

    - **Manual Inspection:** Farmers or agronomists visually inspected leaves for white powdery spots, a time-consuming process prone to human error, especially in large orchards.
    - **Traditional Image Processing:** Early automated methods used basic image processing techniques, such as color and texture analysis, to identify disease symptoms. These approaches often struggled with complex backgrounds and low-resolution images, achieving limited accuracy (e.g., below 90% in some cases).
    - **Laboratory Testing:** Leaf samples were sent to labs for microscopic analysis or pathogen culturing, which was accurate but slow and impractical for real-time field use.
    - **Chemical Indicators:** Some methods involved applying chemical markers to detect fungal presence, but these were costly and not scalable for widespread use.

    These methods were often inefficient, requiring significant expertise and resources, and were less effective for early-stage detection.

    ### Future Technology for Powdery Mildew Detection

    Advancements in machine learning and related technologies promise to enhance the Cherry Tree Mildew Leaf Detector project:

    - **Improved Deep Learning Models:** Future iterations could leverage advanced convolutional neural networks (CNNs) like YOLOv8 or EfficientNet, achieving higher accuracy (potentially >98%) and faster processing for real-time detection.
    - **Integration with UAVs and IoT:** Combining the detector with unmanned aerial vehicles (UAVs) and IoT sensors could enable large-scale, real-time monitoring of orchards, as demonstrated in vineyard mildew detection with 89% accuracy at the vine level.
    - **Hyperspectral Imaging:** Incorporating hyperspectral or multispectral imaging could detect mildew before visible symptoms appear, improving early intervention. Studies on similar crops have reported over 90% accuracy using these techniques.
    - **Explainable AI (XAI):** Integrating XAI, such as Grad-CAM, could provide farmers with visual explanations of disease detection, increasing trust and usability.
    - **Edge Computing:** Deploying models on edge devices (e.g., smartphones or embedded systems) would allow on-site processing, reducing dependency on cloud servers and enabling use in remote areas with limited connectivity.
    - **Global Dataset Expansion:** Expanding training datasets to include diverse cherry varieties and environmental conditions from multiple countries could improve model robustness, addressing challenges like data imbalance and symptom variability.

    These advancements would make the detector more accurate, accessible, and adaptable, revolutionizing cherry orchard management worldwide.
    ''')
    st.info('Use the sidebar to navigate between pages.')

# --- PAGE 2: DATA EXPLORATION ---
elif page == 'Data Exploration':
    st.header('Data Exploration')
    st.subheader('Interactive Image Montage')
    # Show random images from each class
    data_dir = os.path.join('data', 'cherry-leaves')
    class_choice = st.selectbox('Choose class', CLASS_NAMES)
    img_dir = os.path.join(data_dir, class_choice)
    if os.path.exists(img_dir):
        img_files = [f for f in os.listdir(img_dir) if f.endswith('.JPG')]
        n_show = st.slider('Number of images', 3, 12, 9)
        if img_files:
            sel_imgs = np.random.choice(img_files, min(n_show, len(img_files)), replace=False)
            cols = st.columns(3)
            for i, img_name in enumerate(sel_imgs):
                img = Image.open(os.path.join(img_dir, img_name))
                with cols[i % 3]:
                    st.image(img, caption=class_choice, use_column_width=True)
        else:
            st.warning('No images found in this class directory.')
        # Class distribution
        try:
            healthy_count = len(os.listdir(os.path.join(data_dir, 'healthy')))
        except Exception:
            healthy_count = 0
        try:
            mildew_count = len(os.listdir(os.path.join(data_dir, 'powdery_mildew')))
        except Exception:
            mildew_count = 0
        st.subheader('Class Distribution')
        st.bar_chart({'Healthy': healthy_count, 'Powdery Mildew': mildew_count})
        st.write(f"Healthy: {healthy_count} images")
        st.write(f"Powdery Mildew: {mildew_count} images")
    else:
        st.warning('Image directory not found. Example images and class distribution are unavailable on this deployment.')

# --- PAGE 3: MODEL PERFORMANCE ---
elif page == 'Model Performance':
    st.header('Model Performance')
    st.markdown('''
    ## Model Evaluation & Results

    The Cherry Leaf Mildew Detector model was evaluated on a held-out test set to assess its ability to distinguish between healthy and powdery mildew-infected leaves. Below are the key performance metrics and visualizations:

    ### 1. Accuracy & Loss Curves
    These curves show how the model's accuracy and loss evolved during training and validation. A high final accuracy and low loss indicate good learning and generalization.
    ''')
    perf_dir = os.path.join('out', 'visualization')
    acc_loss_path = os.path.join(perf_dir, 'accuracy_loss.png')
    if os.path.exists(acc_loss_path):
        st.image(acc_loss_path, caption='Accuracy and Loss Curves', use_column_width=True)
    else:
        st.info(f"Add 'accuracy_loss.png' to {perf_dir} to display here.")

    st.markdown('''
    ### 2. Confusion Matrix
    The confusion matrix summarizes the number of correct and incorrect predictions for each class. It helps visualize the model's ability to distinguish between healthy and diseased leaves.
    ''')
    cm_path = os.path.join(perf_dir, 'confusion_matrix.png')
    if os.path.exists(cm_path):
        st.image(cm_path, caption='Confusion Matrix', use_column_width=True)
    else:
        st.info(f"Add 'confusion_matrix.png' to {perf_dir} to display here.")

    st.markdown('''
    ### 3. ROC Curve
    The ROC (Receiver Operating Characteristic) curve illustrates the trade-off between sensitivity (recall) and specificity. The area under the curve (AUC) is a measure of the model's ability to distinguish between classes.
    ''')
    roc_path = os.path.join(perf_dir, 'roc_curve.png')
    if os.path.exists(roc_path):
        st.image(roc_path, caption='ROC Curve', use_column_width=True)
    else:
        st.info(f"Add 'roc_curve.png' to {perf_dir} to display here.")

    st.markdown('''
    ### 4. Example Predictions
    Here are sample predictions made by the model on test images. Each image is labeled as either healthy or powdery mildew, along with the model's confidence.
    ''')
    ex_pred_path = os.path.join(perf_dir, 'example_predictions.png')
    if os.path.exists(ex_pred_path):
        st.image(ex_pred_path, caption='Example Predictions', use_column_width=True)
    else:
        st.info(f"Add 'example_predictions.png' to {perf_dir} to display here.")

    st.markdown('''
    ---
    
    **Key Metrics:**
    - **Test Accuracy:** _e.g., 96.5%_  
    - **Precision:** _e.g., 97%_  
    - **Recall:** _e.g., 96%_  
    - **F1 Score:** _e.g., 96.5%_  
    
    > _Note: Replace these values with your actual results if available._

    The model demonstrates strong performance in distinguishing between healthy and powdery mildew-infected cherry leaves, making it a valuable tool for early disease detection in agriculture.
    ''')


# --- PAGE 4: PREDICT ON NEW IMAGES ---
elif page == 'Predict on New Images':
    st.header('Predict on New Images')
    st.markdown('''Upload one or more cherry leaf images (JPG) to get a prediction.''')
    uploaded_files = st.file_uploader('Choose image(s)', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    model_load_error = None
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            @st.cache_resource
            def load_best_model():
                return load_model(MODEL_PATH)
            model = load_best_model()
        except Exception as e:
            model_load_error = str(e)
    else:
        model_load_error = f"Model file not found at {MODEL_PATH}."

    predict_disabled = not uploaded_files or (model is None)
    predict_clicked = st.button('Predict', disabled=predict_disabled)

    if model_load_error:
        st.error(f"Model could not be loaded: {model_load_error}\nPrediction is unavailable on this deployment.")
    elif uploaded_files and model is not None:
        if predict_clicked:
            with st.spinner('Predicting...'):
                results = []
                for uploaded_file in uploaded_files:
                    img = Image.open(uploaded_file).convert('RGB').resize(IMG_SIZE)
                    arr = img_to_array(img) / 255.0
                    arr = np.expand_dims(arr, axis=0)
                    pred_prob = model.predict(arr)[0][0]
                    pred_class = CLASS_NAMES[int(pred_prob > 0.5)]
                    if pred_class == 'healthy':
                        message = f"✅ This leaf is predicted to be HEALTHY. (Confidence: {pred_prob:.2%})"
                    else:
                        message = f"⚠️ This leaf is predicted to have POWDERY MILDEW. (Confidence: {pred_prob:.2%})"
                    st.image(img, caption=message, use_column_width=True)
                    results.append((uploaded_file.name, pred_class, pred_prob))
                # Show a summary message
                healthy_count = sum(1 for _, c, _ in results if c == 'healthy')
                mildew_count = sum(1 for _, c, _ in results if c == 'powdery_mildew')
                st.success(f"Prediction complete! {healthy_count} healthy, {mildew_count} with powdery mildew out of {len(results)} image(s) processed.")
    elif predict_clicked:
        st.warning('Please upload at least one image before clicking Predict.')

# --- PAGE 5: HOW IT WORKS & FAQ ---
else:
    st.header('How It Works & FAQ')
    st.markdown('''
    ## How It Works

    The Cherry Leaf Mildew Detector uses a deep learning pipeline to classify cherry leaf images as either healthy or infected with powdery mildew. Here’s an overview of the process:

    1. **Image Upload:** Users upload one or more images of cherry leaves (JPG, JPEG, or PNG).
    2. **Preprocessing:** Each image is resized to 224x224 pixels and normalized so that pixel values are between 0 and 1.
    3. **Model Prediction:** The preprocessed image is passed to a trained convolutional neural network (CNN) that outputs a probability score for each class (healthy or powdery mildew).
    4. **Result Display:** The app displays the predicted class and confidence for each image, along with a visual indicator (✅ for healthy, ⚠️ for mildew).
    5. **Batch Processing:** Multiple images can be uploaded and processed at once, with a summary of results shown at the end.

    ### Model Pipeline
    - **Input:** Cherry leaf image (RGB)
    - **Preprocessing:** Resize, normalize
    - **Model:** Deep CNN (e.g., EfficientNet, custom Keras model)
    - **Output:** Probability for each class
    - **Threshold:** If probability > 0.5, classified as powdery mildew; otherwise, healthy

    ## Step-by-Step Guide
    1. Navigate to the **Predict on New Images** page using the sidebar.
    2. Upload clear, well-lit images of cherry leaves.
    3. Click **Predict** to run the model.
    4. View the prediction, confidence score, and image preview.
    5. Check the summary for the number of healthy and mildew-infected leaves detected.

    ## FAQ
    **Q: What kind of images work best?**  
    A: Use clear, focused, and well-lit images of single cherry leaves. Avoid blurry or dark photos for best results.

    **Q: Can I upload multiple images at once?**  
    A: Yes, you can upload several images and get predictions for each.

    **Q: How accurate is the model?**  
    A: See the **Model Performance** page for detailed metrics such as accuracy, precision, recall, and example predictions.

    **Q: What should I do if mildew is detected?**  
    A: Follow agricultural best practices for treatment, such as targeted fungicide application, and consult a plant pathologist if needed.

    **Q: Does the app work on other plant species?**  
    A: No, the model is trained specifically for cherry leaves and may not generalize to other plants.

    **Q: Is my data stored?**  
    A: Uploaded images are processed in memory and not stored or shared.

    **Q: Can I use this tool in the field?**  
    A: Yes, the app is designed for both research and practical use in orchards, provided you have internet access and a device with a camera.

    **Q: Who can I contact for support or feedback?**  
    A: See the project README or contact the developer for questions, bug reports, or suggestions.

    ---
    For more details on the model and dataset, explore the other pages using the sidebar.
    ''')
