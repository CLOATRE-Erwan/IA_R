library(corrplot)
library(tidyverse)
library(reticulate)
library(mlr3)
library(mlr3verse)
library(mlr3learners)

# set seed for 
set.seed(42)

# use python
use_condaenv("py3.8", required = TRUE)

# import python module
skl_prep <- import("sklearn.preprocessing")

#preprocessing the data
houses <- read.csv(file = 'data/paris.csv')
houses$cityCode <- NULL
houses$made <- NULL

rows <- sample(nrow(houses))
houses_scaled <- houses[rows, ]

ohe = skl_prep$OneHotEncoder()
ss = skl_prep$StandardScaler()


houses['squareMeters'] <- ss$fit_transform(houses['squareMeters'])
houses['numberOfRooms'] <- ss$fit_transform(houses['numberOfRooms'])
houses['floors'] <- ss$fit_transform(houses['floors'])
houses['cityPartRange'] <- ss$fit_transform(houses['cityPartRange'])
houses['numPrevOwners'] <- ss$fit_transform(houses['numPrevOwners'])
houses['basement'] <- ss$fit_transform(houses['basement'])
houses['attic'] <- ss$fit_transform(houses['attic'])
houses['garage'] <- ss$fit_transform(houses['garage'])
houses['hasGuestRoom'] <- ss$fit_transform(houses['hasGuestRoom'])

# Craete a task with the data
task = mlr3::TaskRegr$new("paris", backend = houses, target = "price")

# Creata a learner
lr = lrn("regr.lm")
lr_cv <- lrn("regr.cv_glmnet")
lr_range <- lrn("regr.ranger")
svm <- lrn("regr.svm")

# Splite data to train set and test set
train_set = sample(task$nrow, 0.8 * task$nrow)
test_set = setdiff(seq_len(task$nrow), train_set)

# Train the model
lr$train(task, row_ids = train_set)
lr_cv$train(task, row_ids = train_set)
lr_range$train(task, row_ids = train_set)
svm$train(task, row_ids = train_set)

# Predicte
prediction_lr <- lr$predict(task, row_ids = test_set)
resp <- prediction_lr$response
truth <- prediction_lr$truth
lr_df <- data.frame(truth, resp)



prediction_lr_cv <- lr_cv$predict(task, row_ids = test_set)
prediction_lr_range <- lr_range$predict(task, row_ids = test_set)
prediction_svm <- svm$predict(task, row_ids = test_set)

# Eval
measure = msr("regr.mse")
score_lr <- prediction_lr$score(measure)
score_lr_cv <- prediction_lr_cv$score(measure)
score_lr_range <- prediction_lr_range$score(measure)
score_svm <- prediction_svm$score(measure)
