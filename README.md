
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
- As a user, I want to see a comprehensive overview of the system’s capabilities and goals when I open the first tab, including where this technology can be applied (such as in cherry-growing regions like the United States, Turkey, and Italy), so I can understand how it addresses powdery mildew detection and supports agricultural efficiency.

- As a farm manager, I want the image montage feature on the dashboard to include a description section at the top explaining the differences between healthy and diseased cherry leaves, so I can make informed crop management decisions.

- As a field worker, I want to be able to quickly upload images of cherry leaves and receive real-time results indicating whether the leaves are healthy or affected by powdery mildew.

- As a quality control officer, I want an interactive section on the dashboard that allows me to scroll through data visualizations showing changes in time and cost savings compared to manual inspections, so I can evaluate the efficiency of the detection process.

- As a data analyst, I want to access detailed information about the model architecture and its decision-making process, along with performance metrics, so I can understand how the model reaches its conclusions and assess its effectiveness.

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

## Hypothesis and how to validate?

**Hypothesis:**
1. A convolutional neural network (CNN) can accurately distinguish between healthy cherry leaves and those affected by powdery mildew with at least 70% accuracy.
2. An interactive dashboard will improve the efficiency of field workers and quality control officers in identifying diseased leaves compared to manual inspection.

**Validation:**
- Train and test the CNN model on the provided Kaggle dataset, using accuracy, precision, recall, and F1-score as metrics. Validate with a holdout test set.
- Deploy the dashboard and collect user feedback or simulate user tasks to compare time and accuracy against manual inspection benchmarks.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

**Business Requirement 1:** Visually differentiate healthy and diseased leaves.
- **Data Visualization:** Create an image montage showing examples of both healthy and diseased leaves, with descriptive text.

**Business Requirement 2:** Predict if a cherry leaf is healthy or has powdery mildew.
- **ML Task:** Train a CNN model to classify images as healthy or diseased. Integrate this model into the dashboard for real-time predictions.

**Business Requirement 3:** Provide interactive dashboard features for different user roles.
- **Data Visualization:** Add widgets for image upload, real-time prediction, and interactive plots showing time/cost savings and model performance.

## ML Business Case

Manual inspection of cherry leaves for powdery mildew is time-consuming and not scalable. By automating detection with a CNN-based model and providing a user-friendly dashboard, the business can:
- Reduce inspection time from 30 minutes per tree to near-instantaneous results.
- Improve accuracy and consistency in disease detection.
- Enable rapid intervention, reducing crop loss and improving yield.
- Replicate the solution for other crops and diseases in the future.

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

- Here, you should list the libraries used in the project and provide an example(s) of how you used these libraries.

## Credits

- In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism.
- You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

### Media

- The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site.

## Acknowledgements (optional)

- Thank the people who provided support throughout this project.
