# END TO END MACHINE LEARNING PROJECT
This is and end-to-end machine learning project. Here, I will describe the project steps and provide information on how to clone and run the project.

## PROJECT OVERVIEW/STEPS
### Selection of Problem
Loan, when taken has the probability of repayment/default in line with loan terms. It is the desire of every lending organization to have their borrowers pay back their loan as at when due. However, this is not usually the case. In event of loan default, lending organizations bear the loss as the loans are charged off and the shortfall treated as expenditure. The provisions for loan defaults reduce total loan portfolio of lending organizations and as such affect interest earnings on assets, thereby constituting a huge cost to lending houses. In view of this, every organization wish to give loan only to those who will pay back, only if they can accurately identify borrowers who will pay back their loan. In this project, I shall be building a Machine Learning model capable of predicting loan default at the point of application, with high accuracy. The model can be adopted for data-driven decisions as regards the approval or rejection of loan applications.
### Model Training
The model training process involved extensive EDA, feature engineering, missing value handling and data preprocessing. XGBoost was used for training. Here, I also carried out hyperparameter tunning, established the optimum hyperparameter and trained the final model. Details are in the notebook.
### Experiment Tracking
The experiment was registered and model parameters as well as metrics logged using MLflow and dagshub.The combination of MLflow and dagshub helped to overcome the major challenge of using MLflow which is the difficulties in collaboration since MLflow experiment tracking and model registry are traditionally done locally. In the case of this project, experiment tracking is done remotely. The project with experiments is hosted on dagshub at https://dagshub.com/joe88data/loan-default-prediction-model
### Workflow Orchestration
Workflow orchestration is necessary to enable continuous model re-training. The workflow was orchestrated using Hydra. Hence, with a single command, data preprocessing, model training, and model evaluation as well as model registry are triggered.
### Model Serving
The model was wrapped in a FastAPI
### Remote Hosting of API
The FastAPI was exposed on Heroku for remote consumption
### Continous Integration and Continous Deployment (CI/CD)
I configured continuous integration and continuous deployment workflow for the project using GitHub Action. On pushing a new version of the project to Github, the process of building and deployment of the new version to Heroku is done automatically.
### Web App
I built a web application that uses the model for prediction, using Streamlit
### Offline Inferencing
The model is configured for offline inferencing. Unlabeled dataframe can be taken in, processed, passed to the model for prediction, and result returned as dataframe with predicted label
### Project Versioning
Git was used for project version control while DVC was used for model and data version control/storage in the cloud
## USING THE PROJECT
The following steps can be followed to clone and run the project
 -   Clone project by running: git clone https://github.com/joagada2/loan-default-prediction-model.git
 -   Download the dataset from: https://www.kaggle.com/wordsforthewise/lending-club and save it in the raw data sub folder in the data folder. name the dataset lending_club_loan_two.csv
 -   Change the directory to the project by running the following command from your command line: cd loan-default-prediction-model
 -  To use the API, run the following command from your command line: python app/app.py
 -  To run the re-training code, run the following command from your command line: python training/src/main.py
 -  To run the streamlit application, run the following command from your command line: streamlit run wev/app.py
 -  To use the model for offline batch inferencing, select a subset of the training dataset, delete the label column, save the dataframe in the input_data subfolder in the inferencing folder, and run the following code from your command line: python inferencing/batch_inferencing.py
### contact me on whatsapp number +2348155337422 for further clarifications on this project. Note: I am open to new opportunities in machine learning




