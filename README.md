# Debugging Checklist

Use this checklist to systematically debug issues in the Cherry Leaf Mildew Detector app. This covers common problems in Streamlit apps, deep learning models, and image data pipelines.

## 1. App Launch & Environment
- [ ] Is Python 3.7+ installed and the correct environment activated?
- [ ] Are all required packages installed? (`pip install -r requirements.txt`)
- [ ] Does `streamlit run app.py` start the app without errors?
- [ ] Are there any missing or misnamed files (e.g., `app.py`, model files, data folders)?

## 2. Data & File Structure
- [ ] Are the data directories (`data/cherry-leaves/healthy`, `data/cherry-leaves/powdery_mildew`) present and populated with images?
- [ ] Are image files in supported formats (JPG, JPEG, PNG) and not corrupted?
- [ ] Are there enough images in each class for the app to display and process?

## 3. Model Loading & Inference
- [ ] Is the trained model file (`out/modeling/best_model.keras`) present and accessible?
- [ ] Does the model load without errors in the app?
- [ ] Are the input image size and preprocessing steps consistent with model training (e.g., 224x224, normalization)?
- [ ] Are predictions returned for uploaded images?

## 4. Streamlit UI & Navigation
- [ ] Does the sidebar navigation work for all pages?
- [ ] Are all pages (Overview, Data Exploration, Model Performance, Prediction, FAQ) rendering as expected?
- [ ] Are images and charts displayed correctly (no broken images or missing plots)?

## 5. Prediction & Results
- [ ] Can you upload and predict on new images without errors?
- [ ] Are the prediction results and confidence scores reasonable?
- [ ] Is the summary of predictions accurate for multiple uploads?

## 6. Visualization & Metrics
- [ ] Are performance plots (accuracy/loss, confusion matrix, ROC, example predictions) present in `out/visualization/`?
- [ ] Are these plots up-to-date and relevant to the current model?
- [ ] Are metrics (accuracy, precision, recall, F1) consistent with expectations?

## 7. Error Handling & Logs
- [ ] Are user-friendly warnings shown for missing files, empty folders, or upload errors?
- [ ] Are exceptions logged or displayed in the Streamlit app for debugging?
- [ ] Are there any stack traces or error messages in the terminal running Streamlit?

## 8. Advanced Checks
- [ ] If using GPU, is TensorFlow/Keras configured to use it? (Check with `nvidia-smi` and TensorFlow logs)
- [ ] Are package versions compatible (TensorFlow, Keras, Pillow, Streamlit, etc.)?
- [ ] Is the app responsive and performant for large images or batch uploads?

## 9. Deployment
- [ ] If deploying (e.g., Heroku, Streamlit Cloud), are all environment variables and files included?
- [ ] Is the `Procfile` (if used) correctly configured?
- [ ] Does the app run as expected in the deployed environment?

---
If you encounter issues, work through this checklist and consult the FAQ in the app or project documentation. For persistent problems, open an issue with error logs and details.

# Mildew Detector

This project provides a machine learning solution for detecting powdery mildew in cherry leaves using image analysis. It includes a dashboard for real-time predictions and visualizations to support agricultural efficiency.




**Labor time**: Manual inspection might take 2-4 hours per hectare per season for leaf-specific checks, based on crop density and inspection frequency.  
**Wage rates**: Using U.S. farm labor rates ($15-$30/hour), a 10-hectare farm could incur $150-$1,200 in labor costs per season for leaf checks.  
**Technology savings**: Studies like the EOS one suggest that tools like drones or satellite imagery can reduce this time by 50-80%, lowering costs significantly.  

**Citations**:  
1. Cheema, M. J. M., et al., "Precision Agriculture Technologies: Present Adoption and Future Strategies," Precision Agriculture, 2023, Elsevier.  
2. USDA Economic Research Service, "Farm Labor," 2023.  
3. EOS Data Analytics, "Satellite Monitoring Revolutionizes 21st Century Agricultural Practices," 2024-09-26; "GIS For Agriculture: Solutions, Applications, Benefits," 2025-07-02.

## Getting Started

1. Clone the repository and open it in your IDE.
2. In the terminal, run `pip3 install -r requirements.txt` to install dependencies.
3. Open the `jupyter_notebooks` directory and use the provided notebooks for data exploration and model development.
4. To run the dashboard, follow the deployment instructions below.

## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to _Account Settings_ in the menu under your avatar.
2. Scroll down to the _API Key_ and click _Reveal_
3. Copy the key
4. In the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you, so do not share it. If you accidentally make it public, then you can create a new one with _Regenerate API Key_.

## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
- The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## User Stories

- **As a User (General),** I want to navigate easily through the dashboard so that I can access different functionalities without confusion.
- **As a Farm Manager,** I want to see a visual differentiation between healthy and mildew-infected leaves (average images, difference plots) so that I can understand the disease patterns better.
- **As a Farm Manager,** I want to view an image montage of healthy and infected leaves so that I can train my eye to identify the disease manually if needed.
- **As a Field Worker,** I want to upload leaf images to the dashboard and get an instant prediction so that I can take immediate action on infected trees.
- **As a Field Worker,** I want to download a report of the predictions (if possible) so that I can track the disease spread (Future Feature).
- **As a Data Analyst,** I want to see the model performance metrics (accuracy, loss, confusion matrix) so that I can validate the reliability of the predictions.
- **As a Data Analyst,** I want to see the project hypothesis and how it was validated so that I can ensure the project is scientifically sound.

## Business Requirements

The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

- 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
- 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

## Business Case


1. **What is the business objective requirement for the machine learning solution?**  
   - Build an image montage so the business manager can visually differentiate between sick and healthy leaves.  
   - Build an interactive real-time dashboard that can upload images and tell whether a leaf is healthy or sick on the spot. This will be used by the gatherers.

2. **Can traditional data analysis be used?**  
   Build an interactive real-time dashboard that can upload images and determine whether a leaf is healthy or sick on the spot. This will be used by the gatherers.

3. **Does the customer need a dashboard or an API endpoint?**  
   The customer needs a dashboard only.

4. **What does success look like?**  
   - Both business objectives will be accomplished: one by a CNN model and one by a non-machine learning solution using only data analytics.  
   - Over 70% accuracy results from the machine learning model.  
   - A functioning and clear image montage.

5. **Can you break down the project into epics and user stories?**
   Here are the User Stories
   - **As a user,** I want to see a comprehensive overview of the system’s capabilities and goals when I open the first tab, including where this technology can be applied (such as in cherry-growing regions like the United States, Turkey, and Italy), so I can understand how it addresses powdery mildew detection and supports agricultural efficiency.  
   - **As a farm manager,** I want the image montage feature on the dashboard to include a description section at the top explaining the differences between healthy and diseased cherry leaves, so I can make informed crop management decisions.  
   - **As a field worker,** I want to be able to quickly upload images of cherry leaves and receive real-time results indicating whether the leaves are healthy or affected by powdery mildew.  
   - **As a quality control officer,** I want an interactive section on the dashboard that allows me to scroll through data visualizations showing changes in time and cost savings compared to manual inspections, so I can evaluate the efficiency of the detection process.  
   - **As a data analyst,** I want to access detailed information about the model architecture and its decision-making process, along with performance metrics, so I can understand how the model reaches its conclusions and assess its effectiveness.
  
     
-And here are the Epics
     
-Information gathering and data collection. ( DataCollection )
-Data visualization, cleaning, and preparation. ( Visualization )
-Model training, optimization and validation. ( ModelingandEvaluation )
-Dashboard planning, designing, and development. ( streamlit_pages )
-Dashboard deployment and release. ( Heroku )

7. **Are there ethical or privacy concerns?**  
   No, there are no ethical or privacy concerns because the dataset does not include any sensitive data.

8. **What level of prediction performance is needed?**  
   Over 70% accuracy.

9. **What are the project's inputs and intended outputs?**  
   - **Inputs:** Images from the Kaggle dataset called "cherry leaves," along with other information about agricultural processes, best practices, and possibly maps.  
   - **Outputs:** An image montage and a functioning machine learning model.

10. **Does the data suggest a particular model?**  
   Yes, a CNN model.

11. **How will the customer benefit?**  
   The customer will benefit by gaining an efficient and reliable tool for detecting powdery mildew in cherry leaves, leading to improved crop management and yield. The dashboard will provide real-time insights, allowing field workers to act quickly, while the image montage will enhance understanding of leaf health. Overall, this solution will reduce reliance on manual inspections, save time, and increase agricultural efficiency, ultimately supporting better decision-making and profitability.

## Project Hypothesis and Validation

**Hypothesis**: Cherry leaves infected with powdery mildew have distinct whitish, powdery fungal patches on their surface that visually differentiate them from healthy leaves.
- **Validation**:
    - **Average Image Study**: We will calculate the average image for both "Healthy" and "Powdery Mildew" classes. We expect the "Powdery Mildew" average image to show lighter, whitish coloring compared to the healthy one.
    - **Difference between Averages**: We will compute the difference between the two average images. This should highlight the regions where the disease is most prominent (likely distinct white patches).
    - **Image Variability Study**: We will analyze the variability (standard deviation) of images. We expect infected leaves to show higher variability in texture and color due to the irregular fungal growth.
    - **Model Learning**: If a CNN can be trained to distinguish between the two classes with high accuracy (>97%), it validates that the visual features (white patches) are distinct enough for a machine learning model to learn.

**Hypothesis**: A deep learning model can effectively replace manual inspection with high accuracy.
- **Validation**:
    - **Model Performance**: We will evaluate the model on an unseen test set. An accuracy of >97% with balanced precision and recall will validate this hypothesis.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

**Business Requirement 1:** Visually differentiate healthy and diseased leaves.
- **Data Visualization Task:** We will create an image montage to visually demonstrate the differences between healthy and powdery mildew-infected leaves.
- **Data Visualization Task:** We will calculate and plot the average image and the variability image for each class (Healthy vs Powdery Mildew) to identify distinct patterns.
- **Data Visualization Task:** We will plot the difference between the average healthy leaf and the average infected leaf to highlight the specific regions where the disease manifests (white powdery patches).

**Business Requirement 2:** Predict if a cherry leaf is healthy or has powdery mildew.
- **ML Task:** We will train a Convolutional Neural Network (CNN) for binary classification. The model will take a leaf image as input and output the probability of it being healthy or infected.
- **ML Task:** We will evaluate the model's performance using accuracy, precision, recall, and F1-score to ensure it meets the >97% accuracy target on the test set.
- **ML Task:** We will integrate this model into the Streamlit dashboard to allow for real-time predictions on uploaded images.

**Business Requirement 3:** Provide interactive dashboard features for different user roles.
- **Dashboard Task:** We will design a user-friendly interface with a sidebar for navigation.
- **Dashboard Task:** We will include a file uploader widget for the "Mildew Detector" page.
- **Dashboard Task:** We will display prediction results with clear visual indicators (e.g., Green/Red text or icons).
- **Dashboard Task:** We will include a "Project Hypothesis" page to explain the scientific validation of the project.

## ML Business Case

**1. Business Objective**:
The primary objective is to automate the detection of powdery mildew in cherry leaves to reduce labor costs and improve crop quality. The current manual process is slow (30 mins/tree) and prone to human error.

**2. Model Details**:
- **Task**: Binary Classification (Healthy vs. Powdery Mildew).
- **Model Type**: Convolutional Neural Network (CNN).
- **Input Data**: RGB images of cherry leaves (resized to 224x224px).
- **Output**: Probability score (0-1) indicating the likelihood of powdery mildew infection.

**3. Success Metrics**:
- **Accuracy**: >97% on the test set.
- **Recall**: High recall is critical to ensure infected leaves are not missed (False Negatives are costly).
- **Speed**: Real-time prediction (<1 second per image).

**4. Business Outcome**:
- **Scalability**: The system can process thousands of images instantly, allowing for frequent and widespread monitoring.
- **Cost Reduction**: Replaces the need for manual inspection, saving estimated labor costs.
- **Quality Assurance**: Ensures only healthy crops reach the market, maintaining Farmy & Foods' reputation.

## Dashboard Design

**Planned Dashboard Pages and Features:**

1. **Home/Overview Page:**
   - Project introduction and business context
   - Overview of powdery mildew and its impact
   - Application regions and user stories

2. **Image Montage Page:**
   - Side-by-side display of healthy and diseased leaf images
   - Descriptive text explaining visual differences

3. **Prediction Page:**
   - Image upload widget for users
   - Real-time prediction output (healthy/diseased)
   - Model confidence score

4. **Data Visualization Page:**
   - Plots showing time/cost savings vs. manual inspection
   - Model performance metrics (accuracy, confusion matrix, etc.)

5. **Model Details Page:**
   - Explanation of model architecture and decision process
   - Downloadable reports or additional resources

## Unfixed Bugs

- You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment

### Heroku

- The App live link is: `https://mildew-detector.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.


## Main Data Analysis and Machine Learning Libraries

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations and array operations.
- **Matplotlib** & **Seaborn**: For data visualization and plotting.
- **Streamlit**: For building the interactive dashboard.
- **Scikit-learn**: For machine learning tasks and evaluation metrics.
- **TensorFlow** & **Keras**: For building and training the Convolutional Neural Network (CNN).
- **Pillow (PIL)**: For image processing.

## Credits

### Content
- The text for the Home page and project introduction details were adapted from the business requirements provided by Code Institute.
- Instructions on how to implement the multi-page Streamlit dashboard were adapted from the Code Institute walkthrough project "Malaria Detector".

### Media
- The dataset was sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).
