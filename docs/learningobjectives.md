Learning Outcomes
Learning Outcome	Description
LO1	Develop an understanding of the fundamentals of Artificial Intelligence, Machine Learning (ML) and Data Science.
LO2	Frame Machine Learning problems that map to business requirements to produce value.
LO3	Define the main business drivers that maximise the effectiveness in a Data Science project.
LO4	Obtain actionable insights using Data Analysis and manipulation.
LO5	Create intelligent systems using Machine Learning.
LO6	Represent data stories via Data Visualisation.
LO7	Collect, arrange and process data from one or more data sources.
Pass Performance
All Learning Outcomes mapped to the Milestone project, contain a set of criteria that are assessed throughout the project. Please find below all Learning Outcomes and their respective assessment criteria descriptions.

LO1: Develop an understanding of the fundamentals of Artificial Intelligence, Machine Learning (ML) and Data Science.
Criteria	Description
1.1	Demonstrate compliance with Business Understanding in terms of the CRISP-DM by describing the contents of the dataset and the business requirements.
LO2: Frame Machine Learning problems that map to business requirements to produce value.
Criteria	Description
2.1	Map the business requirements in a User Story based format to each of the Data Visualisation and ML Tasks along with the specific actions required for the enablement of each task.
2.2	Ensure at least 1 ML task is mentioned in the “Rationale to map the business requirements to the Data Visualisations and ML tasks” section in the README file.

LO3: Define the main business drivers that maximise the effectiveness in a Data Science project
Criteria	Description
3.1	Implement data analysis and data visualisation for the dataset using techniques covered in the course. See supporting information at the end of this section.
3.2	Articulate a Business Case for each Machine Learning task which must include the aim behind the predictive analytics task, the learning method, the ideal outcome for the process, success/failure metrics, model output and its relevance for the user, and any heuristics and training data used.
3.3	Use Git & GitHub for version control.

LO4: Obtain actionable insights using Data Analysis and manipulation.
Criteria	Description
4.1	Outline the conclusions of the data analytics task undertaken that helps answer a given business requirement in the appropriate section on the dashboard page.
4.2	Provide a clear statement on the dashboard to inform the user that the ML model/pipeline has been successful (or otherwise) in answering the predictive task it was intended to address.

LO5: Create intelligent systems using Machine Learning
Criteria	Description
5.1	Model the data to answer a business requirement using a Jupyter Notebook. See supporting information at the end of this section.
5.2	Evaluate, using a Jupyter Notebook, whether the ML Pipeline/Model meets the project requirements as defined in the ML Business Case. See supporting information at the end of this section.
5.3	Maintain Procfile, requirements.txt, runtime.txt, and setup.sh to enable the deployment at Heroku.
5.4	Implement a dashboard using Streamlit
5.5	Include a text section at the top of all Jupyter Notebooks, that describe the Objectives/Inputs/Outputs.
5.6	Implement “app_pages” and “src” folders to manage the dashboard pages for the application and other auxiliary tasks. See supporting information at the end of this section

LO6: Represent data stories via Data Visualisation.
Criteria	Description
6.1	Provide a textual outline of each dashboard page in the Dashboard Design section of the README file. See supporting information at the end of this section.
6.2	Provide a textual interpretation for every plot (or set of plots) in the dashboard. See supporting information at the end of this section.
6.3	Incorporate the main navigation menu in the dashboard that will answer the project requirements, using a structured layout.

LO7: Collect, arrange and process data from one or more data sources
Criteria	Description
7.1	Implement a data collection mechanism, from an endpoint, using Jupyter Notebook.
Supporting Information
Here you will find supporting information related to specific Pass Criteria:

Criteria	Supporting Information
3.1	
If it is an image dataset, it should contain notebooks with tasks to: set image shape, study for average and variability image; the difference between average images, image montage, plot number of images in train, validation, and test set.

If it is a tabular dataset - it should use for example pandas profiling; correlation and/or PPS study; it should visualise data by plotting relevant variables/correlations.

5.1	Your notebook should execute tasks, when applicable, for defining the ML pipeline steps; conducting hyperparameter optimisation; assessing feature importance; augmenting images and loading from folder to memory; defining neural network architecture; using techniques to prevent overfitting (such as early stopping and dropout layer), and fit the model/pipeline. Your modelling notebook has to create and save an ML pipeline/model that was fitted with the collected data.
5.2	If you have a tabular dataset, and if the ML task is regression, you should evaluate at least the R2 score as well as the plot Actual vs Prediction for both train and test sets. Or if the ML task is classification it should evaluate the confusion matrix and Scikit Learn classification report for both train and test sets. If it is an image dataset, it should indicate the learning curve (loss/accuracy) for both the train and validation sets. It should also evaluate and indicate the performance of the test set. Regardless of the case, it should be clearly stated in the notebook section where you evaluate the performance, whether or not your model/pipeline performance meets the performance requirement indicated in the business case
5.6	Other auxiliary tasks include, but are not limited to: handling and rendering predictions, and model evaluation on the dashboard.
6.1	List in bullet points the dashboard pages, and for each page: describe the content (such as text, plot, widgets etc) and, when applicable, indicate the business requirement that a given page is answering.
6.2	The plot should fit in with the scope of the dashboard page. It is not expected that a given plot is disconnected from the main purpose of that dashboard page. For example, consider the “Heritage Housing Issues” dataset has a “project summary” page and displays a disconnected histogram for SalePrice distribution.



Merit Performance
To evidence performance at the Merit level, a learner will, in general, demonstrate characteristics of performance at the Merit level as outlined below. The learner must achieve all Pass and Merit criteria for Merit to be awarded.

The learner has a clear rationale for the development of this project and has produced a fully functioning, well-documented Web Dashboard for a real-life audience.

The finished project has a clear, well-defined purpose addressing a particular target audience (or multiple related audiences). Its purpose would be immediately evident to a new user without having to look at supporting documentation. The Web Dashboard’s design follows the principles of UX design and accessibility guidelines, and the site is fully responsive.

The code is well-organised and easy to follow. The development process is evident through commit messages. The project’s documentation provides a clear rationale for the development of this project and covers all stages of the development life cycle.

The learner demonstrates characteristics of higher-level performance as described below.

Criteria	Description
1.2	Indicate the project hypothesis(es) and their validation process(es) in the README file.
2.3	Justify through statistical means that the hypotheses postulated as a part of the Business Requirements have been successfully met
4.3	Explain the conclusions from the project hypothesis(es) and the steps taken to validate it(them). See supporting information at the end of this section.
5.7	Document and Demonstrate the iterations of parameter tuning strategies or model selection strategies implemented to reach the final model used.
6.4	Implement at least 4 plots in your dashboard, either interactive or not, using libraries such as Matplotlib, Seaborn, or Plotly to answer the business requirements.
6.5	Include interactive visualisations, to allow the user to dynamically interact with graphical components.
7.2	Implement data preparation tasks, using Jupyter Notebook(s), to demonstrate compliance to the Data Preparation step from the CRISP-DM. See supporting information at the end of this section.
Supporting Information
Here you will find supporting information related to specific Merit Criteria

Criteria	Description
4.3	The rationale to validate the hypothesis(es) may be written either/both in the README or on a dashboard page. If applicable, list potential course of actions after your project hypothesis conclusions
7.2	The notebook(s) content will depend on whether your data is tabular or image-based. If the data is tabular you should have separate notebooks for data cleaning and feature engineering, where each should describe which tasks and techniques were applied. In the data cleaning notebook, you should clearly state how you investigated the missing levels and how you handled them. In the feature engineering notebook, your data should be cleaned and you should clearly state how you investigated which potential transformations could be applied to the data and how you validated if a given transformation may look reasonable to consider. If it is image-based, it should have tasks for removing non-image files, split train/validation/test folders.
Important: All Merit criteria must be achieved for a merit to be awarded.