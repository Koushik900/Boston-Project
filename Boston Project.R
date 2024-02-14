
library(MASS)
data(Boston)


library(class)    
library(rpart)    
library(rpart.plot) 
library(e1071)    
library(neuralnet) 
library(caTools)
library(dplyr)


split_ratio_knn <- 0.7
split_ratio_decision_tree <- 0.6
split_ratio_svm <- 0.8
split_ratio_nn <- 0.75


set.seed(123)
split_knn <- sample.split(Boston$medv, SplitRatio = split_ratio_knn)
training_data_knn <- Boston[split_knn, ]
testing_data_knn <- Boston[!split_knn, ]


set.seed(456)
split_decision_tree <- sample.split(Boston$medv, SplitRatio = split_ratio_decision_tree)
training_data_decision_tree <- Boston[split_decision_tree, ]
testing_data_decision_tree <- Boston[!split_decision_tree, ]


set.seed(789)
split_svm <- sample.split(Boston$medv, SplitRatio = split_ratio_svm)
training_data_svm <- Boston[split_svm, ]
testing_data_svm <- Boston[!split_svm, ]


set.seed(101)
split_nn <- sample.split(Boston$medv, SplitRatio = split_ratio_nn)
training_data_nn <- Boston[split_nn, ]
testing_data_nn <- Boston[!split_nn, ]


evaluate_and_visualize_tree <- function(data, max_depth) {
  decision_tree <- rpart(medv ~ ., data = data, method = "anova", control = rpart.control(maxdepth = max_depth))
  
 
  rpart.plot(decision_tree, type = 4, extra = 101)
  
  return(decision_tree)
}


X_train_knn <- training_data_knn[, -14]
Y_train_knn <- training_data_knn$medv
X_test_knn <- testing_data_knn[, -14]
k_value_knn <- 3
knn_model <- knn(train = X_train_knn, test = X_test_knn, cl = Y_train_knn, k = k_value_knn)
plot(testing_data_knn$medv, knn_model, 
     main = "KNN Regression: Actual vs. Predicted Values",
     xlab = "Actual Values", ylab = "Predicted Values",
     col = "blue")


abline(0, 1, col = "red")



max_depth <- 3
decision_tree <- evaluate_and_visualize_tree(training_data_decision_tree, max_depth)
predictions_decision_tree <- predict(decision_tree, testing_data_decision_tree)
predictions_decision_tree
cm <- table(testing_data_decision_tree$medv, predictions_decision_tree)
cm
accuracy <- sum(diag(cm)) / sum(cm) 
accuracy
decision_tree_mse <- sum((predictions_decision_tree - testing_data_decision_tree$medv)^2) / nrow(testing_data_decision_tree)
decision_tree_mse


svm_model <- svm(medv ~ ., data = training_data_svm, kernel = "linear")
predictions_svm <- predict(svm_model, testing_data_svm)
ym <- table(testing_data_svm$medv, predictions_svm)
ym
accuracy_svm <- sum(diag(ym)) / sum(ym) 
accuracy_svm
svm_mse <- sum((predictions_svm - testing_data_svm$medv)^2) / nrow(testing_data_svm)
svm_mse
plot(testing_data_svm$medv, predictions_svm, 
     main = "SVM Regression: Actual vs. Predicted Values",
     xlab = "Actual Values", ylab = "Predicted Values",
     col = "blue")


abline(0, 1, col = "red")



nn_model <- neuralnet(medv ~ ., data = training_data_nn, hidden = c(5, 5))
plot(nn_model)
predictions_nn <- predict(nn_model, testing_data_nn[, -14])
xm <- table(testing_data_nn$medv, predictions_nn)
xm
accuracy_nn <- sum(diag(xm)) / sum(xm) 
accuracy_nn
nn_mse <- sum((predictions_nn - testing_data_nn$medv)^2) / nrow(testing_data_nn)
nn_mse

cat("SVM MSE:", svm_mse, "\n")
cat("Neural Network MSE:", nn_mse, "\n")
cat("Decision-Tree MSE:", decision_tree_mse, "\n" )



