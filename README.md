
# Overview: Classical Music Meets Classical ML Fall 2023

The purpose of this project was to predict which of the previous patrons of an orchestra will likely purchase a season subscription for the 2014-2015 season. The scoring metric for this competition was AUROC, which means only soft predictions for each patron were generated. As a generic overview, prior to the modeling process the data was prepared via thorough pre-processing methodologies. During the modeling process, the XGBoost classifier and the Random Forest models were utilized to generate predictions. In addition, the hyperparameters for both models were tuned via a grid search methodology. The predictions of both the XGBoost and Random Forest models were averaged, producing an ensemble. 

# Data Preprocessing & Feature Engineering/Selection
This part of the data pipeline is responsible for processing and aggregating data to avoid duplicate primary keys. The python scripts associated with this step are _ and _. There are three primary functions that are used through both python scripts.  
1. load_data() : The purpose of this function is to load the various data sets into corresponding dataframes. 
2. training_data_processing(): The purpose of this function is to merge the dataframes together and also aggregate rows based on common account.id (primary key) values. One account.id may be linked to multiple values within a feature, therefore, for categorical variables the mode value within a feature corresponding to a patron was used. 
3. missing_values_filler(): The purpose of this function is to fill in any missing values within the training and test dataframes. Although models such as XGBoost can handle NaN values, to improve the model's performance missing values and non-recognizable characters were dealt with. The features that were chosen as part of the training and test sets were selected based on their presumed relevance to the topic of purchasing a subscription for the 2014-2015 season. 

# Modeling, Hyperparameter Tuning, and Ensemble Creation
Modeling - This part of the data pipeline is key for generating the predicted output for the patrons on the test set. The following code combines the strengths of both Random Forest and XGBoost models. Both models were used because they both have unique strengths that they can bring. For example, random forest is essentially just an ensemble of decision trees, therefore, it can easily reduce overfitting. On the other hand, XGBoost uses gradient boosting, which means that each tree is built one after another, while correcting errors on the prior tree. Furthermore, both provide a unique way to view feature importance. Each model is individually trained on the training set using optimal hyperparameters. 

Hyperparameter Tuning - Here a grid search process is used to try all the possible permutations of the hyperparameter values. The grid search process will find the optimal values of the hyperparameters, which can then be used to train each corresponding model. The cv parameter allows for cross-validation, which decreases the chance of introducing bias. 

Within the random forest model, the n_estimators parameter is used in order to increase the number of trees within the forest. By increasing the number of estimators, the model may perform better but could lead to overfitting. The max_depth parameter controls the depth of each tree, but having a smaller depth could introduce more overfitting. The min_samples_split and min_samples_leaf were also chosen because they control the number of samples required for a split and the number of samples required to be considered a leaf node correspondingly. Finally, we can control if we want to introduce bootstrap sampling or not with the final parameter. 

Within the XGBoost classifier model, the max_depth is used to set the maximum depth of the tree. By increasing the value, the model will become more complex leading to overfitting. The learning_rate parameter is used to set the learning_rate of the model. Typically, a lower learning rate will avoid overshooting any minimas and does a better job of converging at a better minima. The trade-off, however, is that this process could take a much longer time. The n_estimators is used to define the number of trees within the model. Finally, the gamma value is the threshold to make another partition on a leaf node. 

Ensemble Creation - As mentioned before, both models have strengths that need to be utilized, however, both have their fair share of weaknesses as well. Typically, by combining the strengths of multiple models, the overall accuracy should go up. Since both models are very complex in nature, it is important to combine their strengths to mitigate overfitting. After both models were trained using optimal hyperparameters, the final ensemble prediction was the average of predicted values of both models. This resulting data is then stored within a csv file at the output path of the user's choice. 

## Features Selected for the Training and Test Data 

- amount.donated.2013
- amount.donated.lifetime 
- no.donations.lifetime
- first.donated_ordinal 
- subscription_tier
- multiple.subs
- price.level   
- no.seats
- multiple.tickets 
- package
- section 
- TotalWages    
- set
- season_x 
- season_y 
- State'
- billing.city 
- concert.name_x 
- who_x
- location
- concert.name_y
- season
- who_y 
- what

