# IA R ici

! In this POC we will use R + Python !

## Content

* data/paris.csv : the data
* web/app.R : The POC with shiny
* scrpit/main.R : The POC without shiny

## Goal

The goal is therefore to create a model and visualize the results with R.<br>
The data used in this POC will be the paris housing data : https://www.kaggle.com/mssmartypants/paris-housing-price-prediction

## requirement
You need this to make work the POC

* R
    * R
    * mlr3
    * mlr3verse
    * mlr3learners 
    * shiny (if you use app.R)
    * corrplot
    * reticulate
* Python
    * python 3.8
    * Sklearn
    * create a python environement.


## Technical
First set the random seed to `42` for reproctability.<br>
Second link R scrpit to python environement, and import `sklearn.preprocessing` from python.
### Import and pre processing the data
To import the data we used __R__ `red.csv` function.


The two column `"cityCode"` and `"made"` are droped, then the data is suffled.


After we declare a standard scaler with the `StandardScaler` methode from `sklearn.preprocessing` module who comes from python and apply the standardization to the column `"squareMeters"`,  `"numberOfRooms"`, `"floors"`, `"cityPartRange"`, `"numPrevOwners"`, `"basement"`, `"attic"`, `"garage"`, `"hasGuestRoom"`.


Next we created a mlr3 regression task with `mlr3::TaskRegr$new` and put the data inside, later we splited de data into `train_set` and `test_set`.


### Create the model
In the POC four models were created :
* Linear Model Regression (Linear Regression) with `lrn("regr.lm")`
* GLM with Elastic Net Regularization Regression (Penalized Linear Regression) with `lrn("regr.cv_glmnet")`
* Ranger Regression (Random Forest) with `lrn("regr.ranger")`
* Support Vector Machine (SVM) with `lrn("regr.svm")`

All models were created with the `mlr3learners` module form R.


### Evaluate the model

The metrics for ecaluation we be MSE with `msr("regr.mse")` then wit evaluation the score model with `prediction$score(measure)`

### R vs Python
R a better for data exploration, dataviz, math and stats. Python is better for ML and good for data manipulation.
For ML is better to use Python but you can use R + python with the `reticulate` module. 

