library(faraway)
library(readr)
VDRUSA <- read_csv("VDR-USA.csv")
View(VDRUSA)
VDRUSA <- VDRUSA[,-1:-4]
View(VDRUSA)

library(mlbench)
library(caret)
library(corrplot)

validationIndex <- createDataPartition(VDRUSA$ratio, p = .8, list = FALSE)
validation <- VDRUSA[-validationIndex,]
vdrusa <- VDRUSA[validationIndex,]

dim(vdrusa)
sapply(vdrusa, class)
head(vdrusa)
levels(vdrusa$ratio)

percentage <- prop.table(table(vdrusa$ratio))*100
cbind(freq = table(vdrusa$ratio), percentage = percentage)
summary(vdrusa)

#split input and output
x <- vdrusa[,1:5]
y <- vdrusa[,6]

#boxplot for each attribute on one image
par(mfrow = c(1,5))
for(i in 1:5) {
  boxplot(x[,i], main = names(vdrusa)[i])
}

#barplot for class breakdown
plot(y)

#Run algorithms using 10-fold cross-validation
trainControl <- trainControl(method = "cv", number = 10)
metric <- "RMSE"

vdrusa <- vdrusa[,-5]

#split input and output again
x <- vdrusa[,1:4]
y <- vdrusa[,5]

#Linear Model
set.seed(7)
fit.lm <- train(ratio ~ ., data = vdrusa, method = "lm", metric = metric, trControl = trainControl)

#General Linear Model
set.seed(7)
fit.glm <- train(ratio ~ ., data = vdrusa, method = "glm", metric = metric, trControl = trainControl)

#Penalized Linear Model (GLMNET)#
set.seed(7)
fit.glmnet <- train(ratio ~ ., data = vdrusa, method = "glmnet", metric = metric, trControl = trainControl)

#Support Vector Machines#
set.seed(7)
fit.svm <- train(ratio ~ ., data = vdrusa, method = "svmRadial", metric = metric, trControl = trainControl)

#k-Nearest Neighbors#
set.seed(7)
fit.knn <- train(ratio ~ ., data = vdrusa, method = "knn", metric = metric, trControl = trainControl)

#Random Forest#
set.seed(7)
fit.rf <- train(ratio ~ ., data = vdrusa, method = "rf", metric = metric, trControl = trainControl)

#Stochastic Gradient Boosting Model#
set.seed(7)
fit.gbm <- train(ratio ~ ., data = vdrusa, method = "gbm", metric = metric, trControl = trainControl, verbose = FALSE)

#Cubist#
set.seed(7)
fit.cubist <- train(ratio ~ ., data = vdrusa, method = "cubist", metric = metric, trControl = trainControl)

outcome <- resamples(list(LM = fit.lm, GLM = fit.glm, GLMNET = fit.glmnet, SVM = fit.svm, KNN = fit.knn, RF = fit.rf, GBM = fit.gbm, CUBIST = fit.cubist))
summary(outcome)
dotplot(outcome)

#Linear Model Wins#

print(fit.lm)

#estimate skill of LM on the validation dataset
predictions <- predict(fit.lm, validation)
plot(predictions)

