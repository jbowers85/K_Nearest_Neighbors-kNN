#___________________________________________________________________________________________________________________________________
# This is an example workflow that predicts wine quality using the K Nearest Neighbor (kNN) algorithm.
#
# By: James Bowers
# 
# The data used in this code was taken from the UCI Machine Learning Repository; https://archive.ics.uci.edu/ml/datasets/wine+quality.
# 
# Source Data Citation: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
# 
#___________________________________________________________________________________________________________________________________

library(GGally)
library(caret)
library(kknn)


## 1.) K NEAREST NEIGHBORS - ORDINAL DEPENDENT VARIABLE ####

### Get Data ####

#### get both URLs
redUrl <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
whiteUrl <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

#### store and inspect
redData <- read.csv(redUrl, header=TRUE , sep=';')
head(redData)
whiteData <- read.csv(whiteUrl, header=TRUE , sep=';')
head(whiteData)


### Cleanse ####

#### add a column that identifies the color
redData$color <- "red"
whiteData$color <- "white"

#### reorder columns so that color is first
redData <- redData[c(13,1:12)]
whiteData <- whiteData[c(13,1:12)]

head(redData)
head(whiteData)

#### combine colors in a single data frame
allData <- rbind(redData, whiteData)

##### check to make sure all data was inserted into allData
(nrow(redData) + nrow(whiteData)) == nrow(allData)

#### convert data types
allData$color <- as.factor(allData$color)
allData$quality <- factor(allData$quality, ordered=TRUE)

### Profile ####
sapply(allData, function(x) sum(length(which(is.na(x))))) # NA count

dim(allData)
str(allData)
summary(allData)

ggpairs(allData) # pkg: GGally

#### identify collinear features for possible removal
corrMatrix <- cor(allData[2:12])
corrMatrix
highlyCorr <- findCorrelation(corrMatrix, cutoff = 0.7,verbose=T, exact=T)

colnames(corrMatrix)[highlyCorr] # print the columns to remove according to cutoff


### Split Data ####

#### split data by color
whiteData <- allData[which(allData$color == 'white'),]
redData <- allData[which(allData$color == 'red'),]

#### create train and test set 
##### Red
redTrainIndex <- createDataPartition(redData$quality, p=0.8, list=FALSE) # pkg: caret
redTrainIndex
redTrainData <- redData[redTrainIndex,]
redTestData <- redData[-redTrainIndex,]

nrow(redTestData)/nrow(redTrainData)

##### White
whiteTrainIndex <- createDataPartition(whiteData$quality, p=0.8, list=FALSE) # pkg: caret
whiteTrainIndex
whiteTrainData <- whiteData[whiteTrainIndex,]
whiteTestData <- whiteData[-whiteTrainIndex,]

nrow(whiteTestData)/nrow(whiteTrainData)

#### verify split proportions
par(mfrow=c(2,2)) 
barplot(table(redTrainData$quality) / nrow(redTrainData), col='indianred3', main="Red: Train") # proportion of quality
barplot(table(redTestData$quality) / nrow(redTestData), col='indianred3', main="Red: Test") # proportion of quality

barplot(table(whiteTrainData$quality) / nrow(whiteTrainData), col='peachpuff', main="White: Train") # proportion of quality
barplot(table(whiteTestData$quality) / nrow(whiteTestData), col='peachpuff', main="White: Test") # proportion of quality
par(mfrow=c(1,1)) 

#### reset factors for red since red has no '9' quality scores
redTrainData$quality <- factor(redTrainData$quality)
redTestData$quality <- factor(redTestData$quality)


### Build Model ####

#### Red: weighted kNN using kknn package

##### apply preprocessing to training and testing sets independently, we don't want to "leak" information to our training data
redTrainData.processed <- predict(preProcess(redTrainData, method=c("center", "scale")), redTrainData)
summary(redTrainData.processed)

redTestData.processed <- predict(preProcess(redTestData, method=c("center", "scale")), redTestData)
summary(redTestData.processed)

kknn_fit_red <- train.kknn(quality ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + chlorides + 
                   free.sulfur.dioxide + total.sulfur.dioxide + density  + pH + sulphates + alcohol + color,
                  data = redTrainData.processed, 
                  kmax=25, distance=1, kernel=c("triangular", "gaussian", "rank", "rectangular", "epanechnikov", "optimal"), 
                  contrasts = c(unordered = "contr.dummy", ordered = "contr.ordinal"))

kknn_fit_red
plot(kknn_fit_red)


#### White: weighted kNN using kknn package

##### apply preprocessing to training and testing sets independently, we don't want to "leak" information to our training data
whiteTrainData.processed <- predict(preProcess(whiteTrainData, method=c("center", "scale")), whiteTrainData)
summary(whiteTrainData.processed)

whiteTestData.processed <- predict(preProcess(whiteTestData, method=c("center", "scale")), whiteTestData)
summary(whiteTestData.processed)

kknn_fit_white <- train.kknn(quality ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + chlorides + 
                         free.sulfur.dioxide + total.sulfur.dioxide + density  + pH + sulphates + alcohol + color,
                       data = whiteTrainData.processed, 
                       kmax=25, distance=1, kernel=c("triangular", "gaussian", "rank", "rectangular", "epanechnikov", "optimal"), 
                       contrasts = c(unordered = "contr.dummy", ordered = "contr.ordinal"))

kknn_fit_white
plot(kknn_fit_white)


#### Alternatively: non-weighted kknn using caret package
# 
##### define controls and fit model
# control <- trainControl(method = "repeatedcv", number=10, repeats=3)
# 
# knn_fit <- train(quality ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + chlorides + 
#                  free.sulfur.dioxide + total.sulfur.dioxide + density  + pH + sulphates + alcohol + color,
#                  data=redTrainData, 
#                  method="knn", 
#                  preProcess=c("center", "scale"),
#                  trControl=control,
#                  tuneLength=25)
# 
# knn_fit
# plot(knn_fit)
# 
# knnPredict <- predict(knn_fit, redTestData[,c(-13,-14,-15)])
# cm.knn <- confusionMatrix(knnPredict, redTestData$quality)
# cm.knn
# cm.knn$table
# 
# acc.knn <- cm.knn$overall[1]
# acc.knn



### Evaluate ####

#### Red
cm.kknn.red <- table(predict(kknn_fit_red, redTestData.processed[,-13]), redTestData$quality)
cm.kknn.red

acc.kknn.red <- sum(diag(cm.kknn.red))/sum(cm.kknn.red)
acc.kknn.red

redTestData$prediction <- predict(kknn_fit_red, redTestData.processed[,-13])
redTestData$pred_diff <- as.numeric(redTestData$quality) - as.numeric(redTestData$prediction)
barplot(table(redTestData$pred_diff))

#### White
cm.kknn.white <- table(predict(kknn_fit_white, whiteTestData.processed[,-13]), whiteTestData$quality)
cm.kknn.white

acc.kknn.white <- sum(diag(cm.kknn.white))/sum(cm.kknn.white)
acc.kknn.white

whiteTestData$prediction <- predict(kknn_fit_white, whiteTestData.processed[,-13])
whiteTestData$pred_diff <- as.numeric(whiteTestData$quality) - as.numeric(whiteTestData$prediction)
barplot(table(whiteTestData$pred_diff))

### Predict ####

#### define new record
newData <- data.frame(color=c("red","white"), fixed.acidity=0, volatile.acidity=0, citric.acid=0, residual.sugar=0, chlorides=0, 
                  free.sulfur.dioxide=0, total.sulfur.dioxide=0, density=0, pH=0, sulphates=0, alcohol=0)
newData


#### standardize using column means and st dev from original non-normalized dataset 

str(newData)

predict(kknn_fit_red,newData)
predict(kknn_fit_white,newData)

head(testData.processed[,-13])

options('contrasts')