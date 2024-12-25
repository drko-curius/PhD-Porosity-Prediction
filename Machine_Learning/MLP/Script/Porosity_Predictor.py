###############################
#Coding the porosity predictor#
###############################
###########
#Libraries#
###########
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import loguniform
from joblib import dump

#################################################################################################################################################################

#Loads dataset
data = pd.read_csv("Path to Dataset for Neural Network")

#Splits data into features (X) and target variable (y)
X = data[["Print Temperature", "Layer height", "Infill Overlap", "Printing Speed", "Infill Type", "Z-Height"]]
y = data["Porosity"]

#Ensures categorical features are properly encoded
categorical_features = ["Infill Type"]
numerical_features = ["Print Temperature", "Layer height", "Infill Overlap", "Printing Speed", "Z-Height"]

#Create a column transformer with standard scaling for numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),  # StandardScaler will handle wide range like Z-Height
        ("cat", OneHotEncoder(sparse_output=False, drop='first'), categorical_features)
    ]
)

#################################################################################################################################################################

#Simplified MLP structure with early stopping to prevent overfitting
regressor = MLPRegressor(
    hidden_layer_sizes=(50,),  # Starts simpler
    alpha=0.01,  # Slight regularization
    max_iter=1000,
    solver='adam',
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1  #Uses a fraction of the training data as validation for early stopping
)

#Creates a pipeline with preprocessing and regression
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", regressor)
])

#Broadens search for hyperparameters
param_distributions = {
    'regressor__hidden_layer_sizes': [(10,), (50,), (100,), (50, 50), (100, 50)],
    'regressor__alpha': loguniform(1e-5, 1e-1),
    'regressor__learning_rate_init': loguniform(1e-5, 1e-2),
    'regressor__activation': ['relu', 'tanh']
}

#Uses KFold for regression
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Initializes and fits Randomized Search CV
random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=20, cv=cv, scoring='r2', verbose=2, random_state=42)
random_search.fit(X, y)

#Retrieves best parameters and best cross-validation score
print("Best Parameters:", random_search.best_params_)
print("Best Cross-Validation R-squared:", random_search.best_score_)

#Performs cross-validation with the optimized pipeline
cv_scores = cross_val_score(random_search.best_estimator_, X, y, cv=cv, scoring='r2')

print("Cross-Validation R-squared Scores:", cv_scores)
print("Mean R-squared Score:", np.mean(cv_scores))

#################################################################################################################################################################

#Save the optimized model
dump(random_search.best_estimator_, 'SC_porosity_predictor_optimized.joblib')

#################################################################################################################################################################

#Displays the final chosen hyperparameters
print("Final Neural Network Features:")
print("Hidden Layer Sizes:", random_search.best_params_['regressor__hidden_layer_sizes'])
print("Activation Function:", random_search.best_params_['regressor__activation'])
print("Learning Rate:", random_search.best_params_['regressor__learning_rate_init'])
print("Regularization (alpha):", random_search.best_params_['regressor__alpha'])

##############################
#Using the porosity predictor#
##############################
###########
#Libraries#
###########
import joblib
import pandas as pd

#Loads the trained model
model = joblib.load('Path to where the .joblib is stored')

#################################################################################################################################################################

#Defines a template input dictionary with common values, said values here left as an example
input_template = {
    'Print Temperature': 190,
    'Layer height': 100,
    'Infill Overlap': 35,
    'Printing Speed': 30,
    'Infill Type': 'Grid',
}

#Defines a list of Z-Heights for which I want predictions and can be increased as needed
z_heights = [135, 162, 189, 216, 243]

#Creates a list to store porosity predictions
porosity_predictions = []

#################################################################################################################################################################

#Makes a porosity prediction for each Z-Height
for z_height in z_heights:
    input_data = input_template.copy()
    input_data['Z-Height'] = z_height  # Update only the Z-Height
    input_df = pd.DataFrame([input_data])
    porosity = model.predict(input_df)[0]
    porosity_predictions.append(porosity)

#Displays the predicted porosities for each Z-Height
results_df = pd.DataFrame({'Z-Height (um)': z_heights, 'Predicted Porosity': porosity_predictions})
print(results_df)

#################################################################################################################################################################

#Saves to csv
results_df.to_csv('porosity_prediction.csv', index=False)
