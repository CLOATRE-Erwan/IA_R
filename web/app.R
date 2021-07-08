#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.

library(corrplot)
library(reticulate)
library(mlr3)
library(mlr3verse)
library(mlr3learners)
library(shiny)

# set seed for 
set.seed(42)

# use python
use_condaenv("py3.8", required = TRUE)

# import python module
skl_prep <- import("sklearn.preprocessing")

# read data
houses <- read.csv(file = '../data/paris.csv')

# Correlation
mcor <- cor(houses)
infos <- summary(houses)

#preprocessing the data
houses$cityCode <- NULL
houses$made <- NULL

rows <- sample(nrow(houses))
houses <- houses[rows, ]

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

# Create a task with the data
task = mlr3::TaskRegr$new("paris", backend = houses, target = "price")

# Creata 4 learner
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
prediction_lr_cv <- lr_cv$predict(task, row_ids = test_set)
prediction_lr_range <- lr_range$predict(task, row_ids = test_set)
prediction_svm <- svm$predict(task, row_ids = test_set)

# Eval
measure = msr("regr.mse")
score_lr <- prediction_lr$score(measure)
score_lr_cv <- prediction_lr_cv$score(measure)
score_lr_range <- prediction_lr_range$score(measure)
score_svm <- prediction_svm$score(measure)

# pred to data frame
# lr
resp_lr <- prediction_lr$response
truth_lr <- prediction_lr$truth
df_lr <- data.frame(truth_lr, resp_lr)

# lr cv
resp_lr_cv <- prediction_lr$response
truth_lr_cv <- prediction_lr_cv$truth
df_lr_cv  <- data.frame(truth_lr_cv, resp_lr_cv)

# range
resp_lr_range <- prediction_lr$response
truth_lr_range <- prediction_lr_cv$truth
df_lr_range <- data.frame(truth_lr_range, resp_lr_range)

# svm
resp_svm <- prediction_lr$response
truth_svm <- prediction_lr_cv$truth
df_svm <- data.frame(truth_svm, resp_svm)



# Define UI for application
ui <- fluidPage(
   
   # Application title
   titlePanel("IA R ici"),
   p("Oui c'est long a lancer :)"),
   
   # Sidebar with text
   sidebarLayout(
     position = "right",
      sidebarPanel(
        h4("Tested 4 simple model on the Paris housing price data."),
        p("-- Linear Model Regression (Linear Regression)"),
        p("-- GLM with Elastic Net Regularization Regression (Penalized Linear Regression)"),
        p("-- Ranger Regression (Random Forest)"),
        p("-- Support Vector Machine (SVM)"),
        h6("spoiler : Best model = Linear Regression ^^")
      ),
      
      # Main panel with a plot and some text and table
      mainPanel(
        # corrplot section
        h2("Corrélation"),
        plotOutput("distPlot"),
        dataTableOutput("infos"),
        
        # Eval model section
        h2("Eval of Models (Metrics = mse) : "),
        h3("Linear Model Regression (Linear Regression)"),
        textOutput("score_lr"),
        plotOutput("plot_lr"),
        dataTableOutput("df_lr"),
        h3("GLM with Elastic Net Regularization Regression (Penalized Linear Regression)"),
        textOutput("score_lr_cv"),
        plotOutput("plot_lr_cv"),
        dataTableOutput("df_lr_cv"),
        h3("Ranger Regression (Random Forest)"),
        textOutput("score_lr_range"),
        plotOutput("plot_lr_range"),
        dataTableOutput("df_range"),
        h3("Support Vector Machine (SVM)"),
        textOutput("score_svm"),
        plotOutput("plot_svm"),
        dataTableOutput("df_svm")
      )
   )
)

# Define server logic
server <- function(input, output) {
   
  # draw coorplot
   output$distPlot <- renderPlot({
     corrplot(mcor, method="circle", type="upper", tl.col="black", diag = FALSE)
   })
   # summary
   output$infos <- renderDataTable(infos)
   
   # render score as text
   output$score_lr <- renderText({
     paste("Score linear regression :",score_lr)
   })
   output$score_lr_cv <- renderText({
     paste("Score Penalized Linear Regression :",score_lr_cv)
   })
   output$score_lr_range <- renderText({
     paste("Score Random Forest :",score_lr_range)
   })
   output$score_svm <- renderText({
     paste("Score SVM :",score_svm)
   })
   # render data frame
   output$df_lr <- renderDataTable(df_lr,options = list(pageLength = 5))
   output$df_lr_cv <- renderDataTable(df_lr_cv,options = list(pageLength = 5))
   output$df_lr_range <- renderDataTable(df_lr_range,options = list(pageLength = 5))
   output$df_svm <- renderDataTable(df_svm,options = list(pageLength = 5))
   
   output$plot_lr <- renderPlot({
     plot(df_lr, col = "blue")
   })
   output$plot_lr_cv <- renderPlot({
     plot(df_lr_cv, col = "blue")
   })
   output$plot_lr_range <- renderPlot({
     plot(df_lr_range, col = "blue")
   })
   output$plot_svm <- renderPlot({
     plot(df_svm, col = "blue")
   })
   
}

# Run the application 
shinyApp(ui = ui, server = server)

