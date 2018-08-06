# data wrangling
library(tidyverse)
library(forcats)
library(stringr)
library(caTools)

# data assessment/visualizations
library(DT)
library(data.table)
library(pander)
library(ggplot2)
library(scales)
library(grid)
library(gridExtra)
library(corrplot)
library(VIM) 
library(knitr)
library(vcd)
library(caret)
library(lubridate)


# model
library(MLmetrics)
library(randomForest) 
library(rpart)
library(rpart.plot)
library(car)
library(e1071)
library(ROCR)
library(pROC)
library(glmnet)
library(tidyverse)
library(xgboost)
library(magrittr)
library(caret)
library(h2o)
library(irlba)
library(moments)
library(readr)
library(data.table)



#Useful data quality function for missing values, I learned them from kaggle
checkColumn = function(df,colname){
  
  testData = df[[colname]]
  numMissing = max(sum(is.na(testData)|is.nan(testData)|testData==''),0)
  
  
  if (class(testData) == 'numeric' | class(testData) == 'Date' | class(testData) == 'difftime' | class(testData) == 'integer'){
    list('col' = colname,'class' = class(testData), 'num' = length(testData) - numMissing, 'numMissing' = numMissing, 'numInfinite' = sum(is.infinite(testData)), 'avgVal' = mean(testData,na.rm=TRUE), 'minVal' = round(min(testData,na.rm = TRUE)), 'maxVal' = round(max(testData,na.rm = TRUE)))
  } else{
    list('col' = colname,'class' = class(testData), 'num' = length(testData) - numMissing, 'numMissing' = numMissing, 'numInfinite' = NA,  'avgVal' = NA, 'minVal' = NA, 'maxVal' = NA)
  }
  
}
checkAllCols = function(df){
  resDF = data.frame()
  for (colName in names(df)){
    resDF = rbind(resDF,as.data.frame(checkColumn(df=df,colname=colName)))
  }
  resDF
}

get_dup <- function(x) lapply(x, c) %>% duplicated %>% which 

#-------------------------------------------------

#read data

train <- read_csv("Auto1-DS-TestData.csv") 
#test <- read_csv("RAcredit_test.csv")
numerical <- c("price", "wheel-base", "length", "width","height", "normalized-losses", "curb-weight","engine-size", "bore", "stroke", "compression-ratio", "horsepower",
               "peak-rpm", "city-mpg", "highway-mpg")

train[, which(names(train) %in% numerical)] <- sapply(train[, which(names(train) %in% numerical)], as.numeric)
train$symboling <- as.character(train$symboling)

str(train)
# Unique values per column
len <- t(data.frame(lapply(train, function(x) length(unique(x)))))

#Check for Missing values
missing_values <- train %>% summarize_all(funs(sum(is.na(.))/n()))
#missing_values <- test %>% summarize_all(funs(sum(is.na(.))/n()))

missing_values <- gather(missing_values, key="feature", value="missing_pct")
missing_values %>% 
  ggplot(aes(x=reorder(feature,-missing_pct),y=missing_pct)) +
  geom_bar(stat="identity",fill="red")+
  coord_flip()+theme_bw() + labs(x='features', y='% missing', title='Percent missing data by feature')


#nice one to analyze the data
datatable(checkAllCols(train), style="bootstrap", class="table-condensed", options = list(dom = 'tp',scrollX = TRUE))

cols <- sapply(train, class)

numeric_cols <- rownames(data.frame(cols[which(cols != "character")]))
char_cols <- rownames(data.frame(cols[which(cols == "character")]))

#Replace missings with the related column means
for (name in numeric_cols) {
  train[[name]][is.na(train[[name]])] <-  mean(train[[name]], na.rm=TRUE)
}

#seperated the data
char_train <- train[, -which(names(train) %in% numerical)]
num_train <- train[, which(names(train) %in% numerical)]

#here nice EDA functions that makes my world easier and clearer

#Categorical Variables
plotHist <- function(datam, i) {
  datam <- as.data.frame(char_train[,i])
  p <- ggplot(data = datam, aes(x = factor(unlist(datam)))) + stat_count() + xlab(colnames(char_train[i])) + theme_light() + theme(axis.text.x = element_text(angle = 90, hjust =1))
  return (p)
}

doPlots <- function(datam, fun, ii, ncol=3) {
  pp <- list()
  for (i in ii) {
    p <- fun(datam = char_train, i=i)
    pp <- c(pp, list(p))
  }
  do.call("grid.arrange", c(pp, ncol=ncol))
}


doPlots(datam, fun = plotHist, ii = 1:6, ncol = 2)
doPlots(datam, fun = plotHist, ii  = 7:11, ncol = 2)
train[["num-of-doors"]][train[["num-of-doors"]] == "?"] <-  "four"



plotDen <- function(datam, i){
  datam <- as.data.frame(num_train[,i])
  p <- ggplot(data = datam) + geom_line(aes(x = datam), 
                                        stat = 'density', size = 1,alpha = 1.0) + xlab(paste0((colnames(num_train)[i]), 
                                                                                              '\n', 'Skewness: ',round(skewness(num_train[,i], na.rm = TRUE), 2))) + theme_light() 
  return(p)
  
}


doPlots(datam, fun = plotDen, ii = 1:7, ncol = 2)
doPlots(datam, fun = plotDen, ii = 8:15, ncol = 2)


get_dup(train)

#-------Extract the columns has 0 variance
vars <- sapply(train[, which(!(names(train) %in% char_cols))], function(x) var(x, na.rm  = T))
extract_vars <- rownames(data.frame(vars[which(vars < 0.01)]))
#train <- train[ , -which(names(train) %in% extract_vars)]

char_train <- train[, -which(names(train) %in% numerical)]
num_train <- train[, which(names(train) %in% numerical)]



#----------------------------------------
library(factoextra)
n_pca <- 50
m_pca <- prcomp(na.omit(num_train), scale. = T)
summary(m_pca)

auto_kmm <- kmeans(m_pca$x, centers = 5)
p <- fviz_pca_var(m_pca)
# Add supplementary active variables
fviz_add(p, m_pca$rotation, color ="blue", geom="arrow")

fviz_cluster(auto_kmm, num_train)


#-------------------------------------------LINEAR REGRESSION-----------

set.seed(1234)
parts <- createDataPartition(train$price, p = 0.8, list = F) %>% c()
train[, -which(names(train) %in% numerical)] <- lapply(train[, -which(names(train) %in% numerical)], as.factor)
new_train <- train[parts,]
new_test <- train[-parts,]
price <- train$price


reg_auto <- glm(price~., new_train, family = "gaussian")
summary(reg_auto)
plot(reg_auto)

reg_auto$xlevels[["num-of-cylinders"]] <- union(reg_auto$xlevels[["num-of-cylinders"]], levels(new_test$`num-of-cylinders`))
reg_auto$xlevels[["fuel-system"]] <- union(reg_auto$xlevels[["fuel-system"]], levels(new_test$`fuel-system`))

pred1 <- predict(reg_auto, newdata = new_test)
plot(pred1, type = "l")
lines(new_test$price, type ="l", col = "red")

#linear regression halts 

#--------------------------RANDOM FOREST-------------------------
library(randomForest)

rf_train <- train[parts,]
rf_test <- train[-parts,]
set.seed(1000)
names <- names(new_train)
names <- gsub(x = names, "-", "")
colnames(rf_train) <- names
colnames(rf_test) <- names


control <- trainControl(method="repeatedcv", number=10, repeats=3, search = "grid")
set.seed(123)
mtry <- 7
tunegrid <- expand.grid(.mtry=c(1:15))
rf_gridsearch <- train(price~., data=rf_train, method="rf", metric="RMSE", tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)
auto_rf <- randomForest(price ~ ., data = rf_train, importance = T)
print(auto_rf)
importance(auto_rf)
plot(auto_rf)

#Rebuilding the model after analyzing the graphs and parameters
auto_rf <- randomForest(price ~ ., data = rf_train, importance = T, mtry = 10, ntree = 60)
print(auto_rf)
importance(auto_rf)
plot(auto_rf)

fitForest <- predict(auto_rf, newdata = rf_test)

fitForest

pred_vs_actuals_rf <- data.frame(cbind(fitForest, rf_test$price)) %>% cbind(error = fitForest - rf_test$price)

library(plotly)
rf_graph <- plotly::plot_ly(pred_vs_actuals_rf, y = fitForest, mode = "lines", type = "scatter", name = "Prediction-rf") %>% 
  add_trace(y = pred_vs_actuals_rf$V2, type = "scatter", mode = "lines", name = "Actuals-rf") %>%
  add_trace(y = pred_vs_actuals_rf$error, type = "scatter", mode = "lines", name = "Error-rf")  

rf_graph
#The Graphs looks pretty good :)

importance    <- importance(auto_rf)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'%IncMSE'],2))
library(ggthemes)
# Create a rank variable based on importance
rankImportance <- varImportance %>% mutate(Rank = paste0('#',dense_rank(desc(Importance))))

#visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()


#------------------------------XGB-------------------------------


#create correlation matrix
m <- model.matrix(~.-1, num_train) %>% cor(method = "pearson")

library(ggcorrplot)
ggcorrplot(m, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="square", 
           colors = c("red", "white", "springgreen3"), 
           title="Correlogram of data", 
           ggtheme=theme_bw)

#off the variables over 90% corr amount
cor_var <- findCorrelation(m, cutoff = 0.90, names = TRUE) %>% gsub("`", "", .)
train %<>% select(-one_of(cor_var))

rm(m, num_train, missing_values, char_train, len)

sapply(train[, which(!(names(train) %in% char_cols))], class)

train[, -which(names(train) %in% numerical)] <- lapply(train[, -which(names(train) %in% numerical)], as.factor)

set.seed(1234)
parts <- createDataPartition(train$price, p = 0.8, list = F) %>% c()
new_train <- train[parts,]
new_test <- train[-parts,]
price <- train$price

new_train <- model.matrix(~.+0,new_train[,-c(25)]) 
new_test <- model.matrix(~.+0,data = new_test[,-c(25)])

xgbtest <- xgb.DMatrix(data = data.matrix(new_test))
xgbtrain <- xgb.DMatrix(data = data.matrix(new_train), label = price[parts])

All_rmse<- c()
Param_group<-c()
system.time(
  for (iter in 1:20) {
  param <- list(objective = "reg:linear",
                eval_metric = "rmse",
                booster = "gbtree",
                max_depth = sample(6:10, 1),
                eta = runif(1, 0.01, 0.3),
                gamma = runif(1, 0.0, 0.2), 
                subsample = runif(1, 0.6, 0.9),
                colsample_bytree = runif(1, 0.5, 0.8)
                
  )
  cv.nround = 500
  cv.nfold = 5
  mdcv <- xgb.cv(data=xgbtrain, params = param, nthread=10, 
                 nfold=cv.nfold, nrounds=cv.nround,verbose = TRUE)
  # Least Mean_Test_RMSE as Indicator # 
  min_rmse<- min(mdcv$evaluation_log[,test_rmse_mean])
  All_rmse<-append(All_rmse,min_rmse)
  Param_group<-append(Param_group,param)
  # Select Param
  param<-Param_group[((which.min(All_rmse)-1)*8+1):((which.min(All_rmse)-1)*8+8)]
}
)


xgbcv <- xgb.cv( params = param, data = xgbtrain, nrounds = 100, nfold = 5, 
                 showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)

xgb1 <- xgb.train (params = param, data = xgbtrain, nrounds = 47, nthread = 6,
                   watchlist = list(train=xgbtrain), print_every_n = 10, early_stopping_rounds = 10, 
                   maximize = F )

prediction <- predict(xgb1, xgbtest)
preds_vs_actuals <- data.frame(cbind(prediction, price[-parts])) %>% cbind(error = prediction - price[-parts])

library(plotly)
xgb_graph <- plotly::plot_ly(preds_vs_actuals, y = prediction, mode = "lines", type = "scatter", name = "Prediction-xgb") %>% 
  add_trace(y = preds_vs_actuals$V2, type = "scatter", mode = "lines", name = "Actuals-xgb") %>%
  add_trace(y = preds_vs_actuals$error, type = "scatter", mode = "lines", name = "Error-xgb")
xgb_graph

xgb.importance(names(xgb1$feature_names), model = xgb1) %>% xgb.plot.importance(top_n = 30)

#-----------COMPARISON---------

subplot(rf_graph, xgb_graph)

rf_graph
xgb_graph

sum(pred_vs_actuals_rf$error)
sum(preds_vs_actuals$error)

#SO THE RANDOM FORESTS WINS !!!
