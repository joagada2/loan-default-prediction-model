# END TO END MACHINE LEARNING PROJECT
This is an end-to-end machine learning project. Here, I will describe the project steps, provide information on how to clone and run the project and itemize the technologies/skills used in this project.

## PROJECT OVERVIEW/STEPS
### Selection of Problem
Loan, when taken has the probability of repayment/default in line with loan terms. It is the desire of every lending organization to have their borrowers pay back their loans when due. However, this is not usually the case. In the event of loan default, lending organizations bear the loss as the loans are charged off and the shortfall is treated as expenditure. The provisions for loan defaults reduce the total loan portfolio of lending organizations and as such affect interest earnings on assets, thereby constituting a huge cost to lending houses. In view of this, every organization wishes to give loans only to those who will pay them back, only if they can accurately identify borrowers who will pay back their loans. In this project, I built a Machine Learning solution capable of predicting loan default at the point of application, with high accuracy (accuracy:81%, precission:94%, recall:80%). The model can be used for data-driven decisions as regards the approval or rejection of loan applications.
### Model Training
The model training process involved extensive EDA, feature engineering, missing value handling, and data preprocessing. XGBoost was used for training. Here, I also carried out hyperparameter tunning, established the optimum hyperparameter, and trained the final model. Details are in the notebook.
### Experiment Tracking
The experiment was registered and model parameters as well as metrics logged using MLflow and dagshub.The combination of MLflow and dagshub helped to overcome the major challenge of using MLflow which is the difficulties in collaboration since MLflow experiment tracking and model registry are traditionally done locally. In the case of this project, experiment tracking is done remotely. The project with experiments is also hosted on dagshub at https://dagshub.com/joe88data/loan-default-prediction-model
### Workflow Orchestration
Workflow orchestration is necessary to enable continuous model re-training, batch inferencing and continous monitoring. The workflow was orchestrated using prefect. There are 3 different flows. The first contain all tasks from data preprocessing, model training to model evaluation and experiment tracking. The second flow contains all tasks for batch inferencing while the third flow contains all tasks for continuous monitoring of model in production. While the first 2 flows are triggered manually when need arises, the third ia configured to run every 24 hours for continuous model observability in production.
### Model Serving
The model was wrapped in a FastAPI
### Remote Hosting of API
The FastAPI was exposed on Heroku for remote consumption
### Continous Integration and Continous Deployment (CI/CD)
I configured continuous integration and continuous deployment workflow for the project using GitHub Action. On pushing a new version of the project to Github, the process of building and deployment of the new version to Heroku triggered and completed automatically.
### Web App
I built a web application that uses the model for prediction, using Streamlit
### Offline Inferencing
The model is configured for offline inferencing. Unlabeled dataframe can be taken in, processed, passed to the model for prediction, and result returned as dataframe with predicted label
### Model Monitoring/Full Stack Observability
Model inputs, outputs and metrics are being monitored continously for data drift, concept drift and performance decay. WhyLabs was used for this purpose.
### Project Versioning
Git was used for project version control while DVC was used for model and data version control/storage in the cloud
## USING THE PROJECT
The following steps can be followed to clone and run the project
 -   Clone project by running: git clone https://github.com/joagada2/loan-default-prediction-model.git from your terminal.
 -   create a virtual environment and install requirements.txt
 -   Download the dataset from: https://www.kaggle.com/wordsforthewise/lending-club and save it in the raw data sub folder in the data folder. name the dataset lending_club_loan_two.csv
 -   Change directory to the project by running the following command from your command line: cd loan-default-prediction-model (type cd loan and use shift + tab to autocomplete)
 -  To use the API, run the following command from your command line: python app/app.py
 -  To run the re-training code, run the following command from your command line: python training/src/main.py
 -  To run the streamlit application, run the following command from your command line: streamlit run wev/app.py
 -  To use the model for offline batch inferencing, select a subset of the training dataset, delete the label column, save the dataframe in the input_data subfolder in the inferencing folder, and run the following code from your command line: python inferencing/batch_inferencing.py
 ## TOOLS/SKILLS USED IN THIS PROJECT
  - Python (Pandas, matplotlib, seaborn, scikitlearn etc)
  - XGBoost - for model training
  - MLFlow - Experiment tracking
  - dagshub - Experiment tracking, data storage in the cloud and code hosting
  - Hydra - Managing configuration
  - FastAPI - Model Serving
  - Heroku - API exposed on Heroku
  - GitHub/GitHub Action - Code hosting and CI/CD pipeline
  - Streamlit - For creating web app that uses model for prediction
  - WhyLabs - Continous monitoring of model in production
  - Prefect - For workflow orchestration
  - Git - For project version control
  - DVC - For data version control
  - etc
### contact me on whatsapp number +2348155337422 for further clarifications on this project. Note: I am open to new opportunities in machine learning




