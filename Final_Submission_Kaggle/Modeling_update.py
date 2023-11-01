#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error,mean_absolute_error,adjusted_rand_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
import sys
import warnings
warnings.filterwarnings("ignore")

"""
Purpose: The purpose of the following code is to predict the labels for the given test set (public and private) via the use of an ensemble model. The ensemble at the very end averages the predictions of a random forest model 
and an XGBoost model. The script also includes some data cleaning and feature engineering steps to ensure that the test data and training data are in the same format. With cleaned data sets, the model training process 
is conducted. The hyperparmeters of the models are tuned via the use of grid search and cross-validation. The final predictions are saved to a csv file.

Functions: ensemble_rf_xbg: This function's parameters are the training data, test data, categorical columns, and the output directory. The function returns a dataframe that contains the predictions for the test data.

Inputs: 
    - training_data: This is the training data that will be used for training the models. The training data should be the file path to the training data csv file produced via the Train_Set_Engineering.py script.
    - test_data: This is the test data that will be used for making predictions. The test data should be the file path to the test data csv file produced via the Test_Set_Engineering.py script.
    - categorical_columns: This is a list of the categorical columns that will be used for one-hot encoding. The default value is None. In general, this parameter should NOT be changed. 
    - output_directory: This is the file path to the csv file that will contain the predictions for the test data. You MUST set the location of the output directory if wanting to save the predictions to a csv file.

Outputs: 
    - prediction_df: This is a dataframe that contains the predictions for the test data. The dataframe has two columns, which is the ID and Predicted columns. The ID column has the account.id values for the test data. 

""" 

# First, we can create a function that will perform the modeling process on our training data and will return the test predictions for the test data
def ensemble_rf_xbg (training_data, test_data, categorical_columns= None, output_directory = None):
    
    # Define the default categorical columns that will be used for one-hot encoding. 
    if categorical_columns is None:
        categorical_columns = ['amount.donated.2013', 'amount.donated.lifetime', 'no.donations.lifetime', 'first.donated_ordinal', 'subscription_tier', 'multiple.subs', 'price.level',	
        'no.seats', 'multiple.tickets', 'package', 'section', 'TotalWages',	'set', 'season_x', 'season_y', 'State', 'billing.city', 'concert.name_x', 'who_x', 'location',
        'concert.name_y', 'season', 'who_y', 'what']
    
    # Now we need to handle any missing values within these categorical columns. We will replace the NaN values with 'Missing'
    for columns in categorical_columns:
        training_data[columns].fillna('Missing', inplace=True)
        test_data[columns].fillna('Missing', inplace=True)
        training_data[columns] = training_data[columns].astype('category')
    
    # Based on good practice, we should one-hot encode all of the categorical variables
    training_data = pd.get_dummies(training_data, columns=categorical_columns)

    # Now we can define the X and y variables for the training process 
    X = training_data.drop(columns=['label'], axis = 1)
    y = training_data['label']

    # Now we can split the data into the training and validation sets for cross-validation process 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Similar to before, we can now one-hot encode the test data 
    test_data = pd.get_dummies(test_data, columns=categorical_columns)

    # After one-hot encoding there could be some features/columns that appear in the training data but not in the test data. To prevent this mismatch of columns, we can subtracts the columns in test data 
    # from the columns in the training data. 
    missing_columns = set(X_train.columns) - set(test_data.columns)
    # Now we can add the missing columns to the test data and fill them with 0 via the use of a for loop
    for column in missing_columns:
        test_data[column] = 0
    test_data = test_data[X_train.columns]

    # Now we can actually conduct the modeling process which includes also includes the cross-validation process 
    ############ Random Forest Model ############
    # First, we can define our random forest model 
    random_forest = RandomForestClassifier(random_state=100)
    # We can now define our parameter grid for the random forest model 
    random_forest_parameter_grid = {
        # First, we can define the number of trees we want to use
        'n_estimators': [50, 100, 150],
        # Next, we can define the maximum depth of each tree
        'max_depth': [None, 10, 20, 30],
        # Next, we can define the minimum samples required for a split 
        'min_samples_split': [2, 5, 10],
        # Next, we define how many samples are needed for a leaf node
        'min_samples_leaf': [1, 2, 4],
        # Finally, we can createa a boolean value that determines whether or not bootstrap samples are used when building trees
        'bootstrap': [True, False]
    }
    # Now we can perform the grid search process using the GridSearchCV() function 
    random_forest_grid_search = GridSearchCV(random_forest, random_forest_parameter_grid, scoring=make_scorer(accuracy_score), cv=5, verbose=1)
    # We can now fit the grid search model on the training data 
    random_forest_grid_search.fit(X_train, y_train)
    # We can get the optimal parameters for the random forest model and then use those parameters to create the final model
    random_forest_optimal = random_forest_grid_search.best_params_
    # Here we can update the model with the optimal parameters
    random_forest_optimal_model = random_forest.set_params(**random_forest_optimal)
    # Finally, we can fit the model on the combo train and validation data 
    random_forest_optimal_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    ############ XGBoost Model ############
    # First, we can define the XGBoost model
    # Here we define the objective of the model which is binary classification as well as the number of parallel threads for faster processing
    xgboost = xgb.XGBClassifier(objective='binary:logistic', nthread = 4, seed = 42)
    # Like before, we can define the parameter grid for the XGBoost model
    xgboost_parameter_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'gamma': [0, 0.1, 0.2]
    }
    # Now we can perform the grid search process
    xgboost_grid_search = GridSearchCV(xgboost, xgboost_parameter_grid, scoring=make_scorer(accuracy_score), cv=5, verbose=1)
    # We can now fit the grid search model on the training data
    xgboost_grid_search.fit(X_train, y_train)
    # Now we can find the optimal parameters for the XGBoost model and then use those parameters to create the final model
    xgboost_optimal = xgboost_grid_search.best_params_
    # Here we can update the model with the optimal parameters
    xgboost_optimal_model = xgboost.set_params(**xgboost_optimal)
    # Finally, we can fit the model on the combto train and validation data
    xgboost_optimal_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    ############ Ensemble Model Predictions ############
    # Now we can average the predictions of the random forest and XGBoost models to create the final predictions
    # We need to get the probabilities of each class, however, we are only focused on the probability of the positive class. 
    # We will get the probabilities of the positive class for both models
    random_forest_probabilities = random_forest_optimal_model.predict_proba(test_data)[:,1]
    xgboost_probabilities = xgboost_optimal_model.predict_proba(test_data)[:,1]
    # Now we can average the probabilities of the positive class for both models
    ensemble_average = (random_forest_probabilities + xgboost_probabilities)/2
    # Finally, we can save the predictions to a csv file
    prediction_df = pd.DataFrame({'ID': test_data.index, 'Predicted': ensemble_average})
    if output_directory:
        prediction_df.to_csv(output_directory, index=False)
    return prediction_df

if __name__ == '__main__':
    # Here we load in the training and test data
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    # Here we load in the training and test data
    train_data = pd.read_csv(train_path)
    # Here we set the index to be the account.id column for the training data
    train_data.set_index('account.id', inplace=True)
    test_data = pd.read_csv(test_path)
    # Here we set the index to be the account.id column for the test data
    test_data.set_index('account.id', inplace=True)
    # Optionally, you can get the output_directory and categorical_columns from command line arguments as well
    output_dir = None if len(sys.argv) <= 3 else sys.argv[3]
    # Now we can run the modeling function
    prediction_df = ensemble_rf_xbg(train_data, test_data, categorical_columns= None, output_directory = output_dir)
    
