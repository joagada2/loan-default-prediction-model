# END TO END MACHINE LEARNING PROJECT
This is and end-to-end machine learning project. Here, I will describe the project steps and provide information on how to clone and run the project.

## PROJECT OVERVIEW/STEPS
### Selection of Problem
Loan, when taken has the probability of repayment in line with loan terms or default. It is the desire of every lending organization to have their borrowers pay back their loan as at when due. However, this is not usually the case. In event of loan default, lending organizations bear the loss as the loans are charged off and the shortfall treated as expenditure. The provisions for loan defaults reduce total loan portfolio of lending organizations and as such affects interest earnings on assets, thereby constituting a huge cost to lending houses. In view of this, every organization wish to give loan only to those who will pay back only if they can accurately identify borrowers who will pay back their loan. In this project, I shall be building a Machine Learning model capable of predicting loan default at the point of application with high accuracy. The model can be adopted for data driven decision as regards approval or rejection of approval for loansdecisionsdata-drivenaffectwishesloansis the .
### Model Training
The model training process involved extensive EDA, feature engineering, missing value handling and data prethe ,handling,processing. XGBoost was used for training. Here, I also carried out hyperparameter tunning, established the optimum hyperparameter and trained the final model. Details are in the notebook.
### Experiment Tracking
The experiment was registered and model parameters as well as metrics logged using MLflow and dagshub.The combination of MLflow and dagshub helped to overcome the major challenge of using MLflow which is the difficulties in collaboration since MLflow experiment tracking and model registry are traditionally done locally. In the case of this project, experiment tracking is done remotely. The project with experiments is hosted at https://dagshub.com/joe88data/loan-default-prediction-model
### Workflow Orchestration
Workflow orchestration is neccessary tp enable continuous model re-training. The workflow was orchestration using Hydra. Hence, with a single command, data preprocessing, model training, and model evaluation as well as model registry are done automatically
### Model Serving
The model was wrapped in a FastAPI
### Remote Hosting of API
The FastAPI was exposed on Heroku for remote consumption
### Continous Integration and Continous Deployment (CI/CD)
I configured continuous integration and continuous deployment workflow for the project usingGitHubb Action. On pushing a new version of the project to Github, the process of building and deployment of the new version to Heroku is done automatically.
### Web App
I built a web application that uses the model for prediction using Streamlit
### Offline Inferencing
The model is configured for offline inferencing. Unlabel dataframe can be taken in and returned as dataframe with predicted label
### Project Versioning
Git was used for version control while DVC was used for model and data version control
## USING THE PROJECT
The following steps can be followed to clone and run the project
clone project by running: git clone https://github.com/joagada2/loan-default-prediction-mode.git
download dataset from 




