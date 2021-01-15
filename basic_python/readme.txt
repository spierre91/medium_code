## Churn Probability Prediction


## Random Foresst Classsfier
A random forest classifier is used to predict probability of churn. The  random forests  algorit
hm fits many decision tree classifiers on various subsamples of data and using averaging to improve accuracy and reduce overfitting.  
 
## Alpha.py
The alpha.py file contains a class, called Alpha, that contains an __init__ method, set_state method, fit method and predict method. The __init__ method is used to read in the data, define mod
el features and random forest parameters. Set state is used to filter our dataframe and construct our training data. Fit is used to fit the random forest the training data. Predict is used to
make new predictions on input. The predict method returns probability of churn. 

##Instructions
In a command line install wqpt utility. To run a local test run the command $(wqpt generate). To run a test on the platform on the researcher test set run the command wqpt -rf. To run a test on the platform on the hold out set you run the command wqpt -f. 
   
