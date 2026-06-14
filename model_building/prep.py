# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Defining constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Pammi123/engine-health-prediction/engine_data.csv"
engine_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")


# Defining the target variable for the classification task
target = 'Engine Condition'

# List of independent numerical features in the dataset
ind_features = [
    'Engine rpm',                 # Engine Speed in RPM
    'Lub oil pressure',           # Pressure of the lubricating oil in kilopascals (kPa)
    'Fuel pressure',              # Pressure at which fuel is supplied to the engine in kilopascals (kPa)
    'Coolant pressure',           # Pressure of the engine coolant in kilopascals (kPa)
    'lub oil temp',               # Temperature of the lubricating oil in degrees Celsius (°C)
    'Coolant temp'                # Temperature of the engine coolant in degrees Celsius (°C)
]

# Defining predictor matrix (X) using selected numeric and categorical features
X = engine_dataset[ind_features]

# Defining target variable
y = engine_dataset[target]


# Splitting the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting fixed random seed
)

# Saving individual training and testing sets
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# Uploading the train and test datasets back to the Hugging Face data space
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Pammi123/engine-health-prediction",
        repo_type="dataset",
    )
