## Setup
library(tree)
library(randomForest)
library(VIM)
library(caret)
library(e1071)
library(stringr)
library(mice)

set.seed(0)

## Data Loading
train <- read.csv("train.csv")
test <- read.csv("test.csv")
train$Survived <- as.factor(train$Survived)
head(train)
head(test)


## Data Exploration

### Summary

summary(train)


### Checking Uniques

# check that every entry is indeed unique
print(length(unique(train$PassengerId)))
print(length(train$PassengerId))


### Checking Missingness

na_count_train <- sapply(train, function(y) sum(length(which(is.na(y)))))
na_count_train <- data.frame(na_count_train)
print(na_count_train)
# visualize missing data
aggr(train)
matrixplot(train[order(train$Survived),])


na_count_test <- sapply(test, function(y) sum(length(which(is.na(y)))))
na_count_test <- data.frame(na_count_test)
print(na_count_test)
# visualize missing data
aggr(test)


# plot survival frequency
survival <- table(train$Survived)
barplot(survival, xlab = "Frequency", ylab = "Survival (1 = Yes, 0 = No)", main = "Survival", 
        col = "cornflowerblue", border = NA, horiz = TRUE)


## Imputation

# perform multiple imputations on missing data
nrow(train)
nrow(test)
combined_imp <- rbind(train[, -c(2)], test)

m = 5
age <- mice(combined_imp, m = m, method = "cart")
# pull out a more complete data set
combined_imp <- list()
for (i in 1:m){
  combined_imp[[i]] <- complete(age,i)
}

combined_imp <- data.frame(combined_imp[5])


combined_imp$Sex <- as.factor(combined_imp$Sex)


train_complete <- combined_imp[1:891,] 
test_complete <- combined_imp[892:1309,]

train_complete$Survived <- train$Survived




## Feature extraction 

# family size
train_complete$FamSize <- (train_complete$SibSp + train_complete$Parch + 1)
test_complete$FamSize <- (test_complete$SibSp + test_complete$Parch + 1)


# title
train_complete$Title <- str_extract(train_complete$Name, "(Mr.|Ms.|Dr.|Sr.|Mrs.|Miss.|Master.|Don.|Rev.|Major.|Col.|Capt.)")
# replace NAs in title column
train_complete[is.na(train_complete)] <- "None"

train_complete[train_complete=="Mr."] <- 0
train_complete[train_complete=="Mr "] <- 0
train_complete[train_complete=="Mr"] <- 0
train_complete[train_complete=="Mrs"] <- 1
train_complete[train_complete=="Miss."] <- 1
train_complete[train_complete=="Ms."] <- 1
train_complete[train_complete=="Master."] <- 2
train_complete[train_complete=="Don."] <- 3
train_complete[train_complete=="Col."] <- 3
train_complete[train_complete=="Rev."] <- 3
train_complete[train_complete=="Dr."] <- 3
train_complete[train_complete=="Major."] <- 3
train_complete[train_complete=="Capt."] <- 3
train_complete[train_complete=="None"] <- 3
train_complete[train_complete=="Dono"] <- 3
train_complete[train_complete=="Dona"] <- 3
train_complete[train_complete=="Colb"] <- 3
train_complete[train_complete=="Dre"] <- 3
train_complete[train_complete=="Dri"] <- 3
train_complete[train_complete=="Dra"] <- 3
train_complete[train_complete=="Coll"] <- 3
train_complete[train_complete=="Cole"] <- 3

train_complete$Title <- as.factor(train_complete$Title)

test_complete$Title <- str_extract(test_complete$Name, "(Mr.|Ms.|Dr.|Sr.|Mrs.|Miss.|Master.|Don.|Rev.|Major.|Col.|Capt.)")
# replace NAs in title column
test_complete[is.na(test_complete)] <- "None"

test_complete[test_complete=="Mr."] <- 0
test_complete[test_complete=="Mr"] <- 0
test_complete[test_complete=="Mrs"] <- 1
test_complete[test_complete=="Miss."] <- 1
test_complete[test_complete=="Ms."] <- 1
test_complete[test_complete=="Master."] <- 2
test_complete[test_complete=="Don."] <- 3
test_complete[test_complete=="Col."] <- 3
test_complete[test_complete=="Rev."] <- 3
test_complete[test_complete=="Dr."] <- 3
test_complete[test_complete=="Major."] <- 3
test_complete[test_complete=="Capt."] <- 3
test_complete[test_complete=="None"] <- 3
test_complete[test_complete=="Dono"] <- 3
test_complete[test_complete=="Dona"] <- 3
test_complete[test_complete=="Colb"] <- 3
test_complete[test_complete=="Dre"] <- 3
test_complete[test_complete=="Dri"] <- 3
test_complete[test_complete=="Dra"] <- 3
test_complete[test_complete=="Coll"] <- 3
test_complete[test_complete=="Cole"] <- 3

test_complete$Title <- as.factor(test_complete$Title)



summary(train_complete)
summary(test_complete)


## Modeling

table(is.na(train_complete))
table(is.na(test_complete))


head(train_complete)
head(train_complete[, -c(1,12)])


drop <- c("AgeGrp","SibSp", "Name", "Parch")
train_complete = train_complete[,!(names(train_complete) %in% drop)]
test_complete = test_complete[,!(names(test_complete) %in% drop)]
head(train_complete[, -c(1,9)])



# use k-fold cross validation

repeatedCV <- trainControl(method="cv", number=5)




rf_grid <- expand.grid(mtry = seq(from = 2, to = ncol(train_complete[, -c(1,9)]) - 1, by = 1))

rf_model_2 <- train(x = train_complete[, -c(1,9)],
                    y = train_complete$Survived,
                    method = "rf", 
                    trControl = repeatedCV, 
                    importance = TRUE, 
                    tuneGrid = rf_grid)






rf_model_2

paste("The maximum accuracy was", round(max(rf_model_2$results$Accuracy), 5))



varImp(rf_model_2)




## Prediction


nrow(test_complete)


res <- predict(rf_model_2, test_complete)

res <- data.frame(res)

final_sub <- data.frame(cbind(test_complete$PassengerId, res))

head(final_sub)


colnames(final_sub) <- c("PassengerId", "Survived")
head(final_sub)



write.csv(final_sub, "submission.csv", row.names = F)
