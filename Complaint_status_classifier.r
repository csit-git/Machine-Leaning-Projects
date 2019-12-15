setwd("C:/Users/Chandra/Desktop/Brainwaves/ML/classification")
train_data_stock<-read.csv("train.csv",na.strings = c("",NA),encoding = "UTF-8",stringsAsFactors = F)
train_data_stock<-train_data_stock[,-c(1)]
train_data_stock$date_diff <- as.numeric(as.Date(as.character(train_data_stock$Date.sent.to.company), format="%m/%d/%Y")-
  as.Date(as.character(train_data_stock$Date.received), format="%m/%d/%Y"))
DateCol<-train_data_stock[,'Date.received']
train_data_stock$Company.response<-ifelse(is.na(train_data_stock$Company.response),"missing",train_data_stock$Company.response)
train_data_stock$Consumer.disputes<-ifelse(is.na(train_data_stock$Consumer.disputes),"missing",train_data_stock$Consumer.disputes)

#data prepration
train_data_stock<-train_data_stock[ , -which(names(train_data_stock) %in% c("Date.received","Date.sent.to.company"))]
train_data_stock$Transaction.Type<-as.factor(train_data_stock$Transaction.Type)
train_data_stock$Complaint.reason<-as.factor(train_data_stock$Complaint.reason)
train_data_stock$Company.response<-as.factor(train_data_stock$Company.response)
train_data_stock$Complaint.Status<-as.factor(train_data_stock$Complaint.Status)
train_data_stock$Consumer.disputes<-as.factor(train_data_stock$Consumer.disputes)

library(tm)
library(RTextTools)
library(SnowballC)
library(textstem)

#Preprocess train_data_stock customer.complaint data 
train_data_stock_customer_complaint_df <- data.frame(train_data_stock$Consumer.complaint.summary)
names(train_data_stock_customer_complaint_df) <-"text"
train_data_stock_text.corpus <- Corpus(VectorSource(train_data_stock_customer_complaint_df$text))
train_data_stock_text.corpus <- tm_map(train_data_stock_text.corpus, tolower)
removeUrl  <- function(x) gsub("http[[:alnum:]]*|http[^[:space:]]*", "", x)
train_data_stock_text.corpus <- tm_map(train_data_stock_text.corpus, removeUrl)
train_data_stock_text.corpus <- tm_map(train_data_stock_text.corpus, content_transformer(gsub), pattern="x*?x",replace=" ")
train_data_stock_text.corpus <- tm_map(train_data_stock_text.corpus, removePunctuation)
train_data_stock_text.corpus <- tm_map(train_data_stock_text.corpus, removeNumbers)
train_data_stock_text.corpus <- tm_map(train_data_stock_text.corpus, content_transformer(gsub), pattern="[^[:alpha:]]",replace=" ")
train_data_stock_text.corpus <-tm_map(train_data_stock_text.corpus,stripWhitespace)
stopwords.tot<- c(stopwords(kind="SMART"),stopwords(kind="fr"),stopwords(kind="es"))
train_data_stock_text.corpus <- tm_map(train_data_stock_text.corpus, removeWords, stopwords.tot)
train_data_stock_text.corpus <-tm_map(train_data_stock_text.corpus,stripWhitespace)
train_data_stock_text.corpus <- tm_map(train_data_stock_text.corpus, stemDocument, language = "english")
train_data_stock_text.corpus <- tm_map(train_data_stock_text.corpus, stemDocument, language = "french")
train_data_stock_text.corpus <- tm_map(train_data_stock_text.corpus, stemDocument, language = "spanish")
#generate document term matrix
train_data_stock_dtm <- tm::DocumentTermMatrix(train_data_stock_text.corpus)
train_data_stock_dtm.tfidf <- tm::weightTfIdf(train_data_stock_dtm)                                 
train_data_stock.tfidf<-removeSparseTerms(train_data_stock_dtm.tfidf,sparse=0.70)
inspect(train_data_stock.tfidf)
train_data_stock.matrix<-as.data.frame(as.matrix(train_data_stock.tfidf))
train_data_stock <- data.frame(cbind(train_data_stock,train_data_stock.matrix))
# drop text column customer complain summary
train_data_stock<-train_data_stock[ , -which(names(train_data_stock) %in% c("Consumer.complaint.summary"))]

#dropping columns  
train_data_stock<-train_data_stock[,-c(2,3,6,9)]

#create dummy variables
train_data_stock_new <- createDummyFeatures (obj = train_data_stock[,-2])
train_data_stock_new<- data.frame(cbind(train_data_stock_new,train_data_stock[,2]))

names(train_data_stock_new)[24]<-"target"


#dropping missing feature
train_data_stock_new<-train_data_stock_new[,-c(21)]



#generating train and test 
set.seed(123)
splitIndex <-sample.int(n = nrow(train_data_stock_new), size = floor(.80*nrow(train_data_stock_new)), replace = F)
train<-train_data_stock_new[splitIndex,]
test<-train_data_stock_new[-splitIndex,]


#running xgboost
library(data.table)
library(mlr)
library(xgboost)  # the main algorithm
library(archdata) # for the sample dataset
library(caret)    # for the confusionmatrix() function (also needs e1071 package)
library(dplyr)    # for some data preperation
 

# Create numeric labels with one-hot encoding
train_labs <- as.numeric(train$target) - 1
test_labs <- as.numeric(test$target) - 1

new_train <- model.matrix(~.+0, data = train[, -c(23)])
new_test <- model.matrix(~.+0, data = test[, -c(23)])

# Prepare matrices
xgb_train <- xgb.DMatrix(data = new_train, label = train_labs)
xgb_test <- xgb.DMatrix(data = new_test, label = test_labs)



# Set parameters(default)
params <- list(booster = "gbtree", objective = "multi:softprob", num_class = 5, eval_metric = "mlogloss", max_depth=4, min_child_weight=2.97, subsample=0.621, colsample_bytree=0.757)

# Calculate # of folds for cross-validation
xgbcv <- xgb.cv(params = params, data = xgb_train, nrounds = 100, nfold = 5, showsd = TRUE, stratified = TRUE, print_every_n = 10, early_stop_round = 20, maximize = FALSE, prediction = TRUE)

# Mutate xgb output to deliver hard predictions
xgb_train_preds <- data.frame(xgbcv$pred) %>% mutate(max = max.col(., ties.method = "last"), label = train_labs + 1)

# Examine output
head(xgb_train_preds)


# Confustion Matrix
# Automated confusion matrix using "caret"
xgb_conf_mat_2 <- confusionMatrix(factor(xgb_train_preds$max),
                                  factor(xgb_train_preds$label),
                                  mode = "everything")
print(xgb_conf_mat_2)

# Create the model
xgb_model <- xgb.train(params = params, data = xgb_train, nrounds = 33,watchlist = list(val=xgb_test,train=xgb_train), print.every.n = 10, early.stop.round = 10, maximize = F)

# Predict for test set
xgb_test_preds <- predict(xgb_model, newdata = xgb_test)

xgb_test_out <- matrix(xgb_test_preds, nrow = 5, ncol = length(xgb_test_preds) / 5) %>% 
  t() %>%
  data.frame() %>%
  mutate(max = max.col(., ties.method = "last"), label = test_labs + 1) 

# Confustion Matrix
xgb_test_conf2 <- confusionMatrix(factor(xgb_test_out$max),
                                 factor(xgb_test_out$label),
                                 mode = "everything")
print(xgb_test_conf2)


# compute feature importance matrix
imp_mat = xgb.importance(feature_names = colnames(new_train), model = xgb_model)
xgb.plot.importance (importance_matrix = imp_mat[1:20])
head(imp_mat,20)


#Tuning hyper parameter

#create tasks
train_task <- makeClassifTask (data = train,target = "target")
test_task <- makeClassifTask (data = test,target = "target")

train_task <- createDummyFeatures (obj = train_task)
test_task <- createDummyFeatures (obj = test_task)

#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="multi:softprob", num_class = 5, eval_metric = "mlogloss", nrounds=100L, eta=0.1)

#set parameter space
params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree")), makeIntegerParam("max_depth",lower = 3L,upper = 10L), makeNumericParam("min_child_weight",lower = 1L,upper = 10L), makeNumericParam("subsample",lower = 0.5,upper = 1), makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

#search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)

#set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

#parameter tuning
mytune <- tuneParams(learner = lrn, task = train_task, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)

#set hyperparameters
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

#train model
xgmodel <- mlr::train(lrn_tune,train_task)

xgpred <- predict(xgmodel,test_task)

confusionMatrix(xgpred$data$response,xgpred$data$truth)


#Correlations
library(pcaplot)
library(stats)
train_pca <- prcomp(train_data_stock_new[, -187], center = TRUE, scale = TRUE) 

summary(train_pca)
biplot(train_pca)
biplot(train_pca, xlabs = rep("", nrow(train_data_stock_new)))
pcaCharts(train_pca)

plot(train_pca$x[,1:2], col = train_data_stock_new[,187])

print(train_pca)

train_data_stock$Transaction.Type<-as.numeric(train_data_stock$Transaction.Type)
train_data_stock$Complaint.Status<-as.numeric(train_data_stock$Complaint.Status)
train_data_stock$Complaint.reason<-as.numeric(train_data_stock$Complaint.reason)
train_data_stock$Company.response<-as.numeric(train_data_stock$Company.response)
train_data_stock$Consumer.disputes<-as.numeric(train_data_stock$Consumer.disputes)

install.packages("Hmisc")
res <- cor(train_data_stock)
library("Hmisc")
res2 <- rcorr(as.matrix(train_data_stock))
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}
flattenCorrMatrix(res2$r, res2$P)

library(corrplot)
corrplot(res, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)
corrplot(res2$r, type="upper", order="hclust", 
         p.mat = res2$P, sig.level = 0.01, insig = "blank")


#Preparing test stock data
test_stock<-read.csv("test.csv",na.strings = c("",NA),encoding = "UTF-8",stringsAsFactors = F)
test_data_stock<-test_stock[,-c(1)]
test_data_stock$date_diff <- as.numeric(as.Date(as.character(test_data_stock$Date.sent.to.company), format="%m/%d/%Y")-
                                          as.Date(as.character(test_data_stock$Date.received), format="%m/%d/%Y"))
DateCol<-test_data_stock[,'Date.received']
test_data_stock$Company.response<-ifelse(is.na(test_data_stock$Company.response),"missing",test_data_stock$Company.response)
test_data_stock$Consumer.disputes<-ifelse(is.na(test_data_stock$Consumer.disputes),"missing",test_data_stock$Consumer.disputes)
test_data_stock<-test_data_stock[ , -which(names(test_data_stock) %in% c("Date.received","Date.sent.to.company"))]
test_data_stock$Transaction.Type<-as.factor(test_data_stock$Transaction.Type)
test_data_stock$Complaint.reason<-as.factor(test_data_stock$Complaint.reason)
test_data_stock$Company.response<-as.factor(test_data_stock$Company.response)
test_data_stock$Consumer.disputes<-as.factor(test_data_stock$Consumer.disputes)

#Preprocess train_data_stock customer.complaint data 
test_data_stock_customer_complaint_df <- data.frame(test_data_stock$Consumer.complaint.summary)
names(test_data_stock_customer_complaint_df) <-"text"
test_data_stock_text.corpus <- Corpus(VectorSource(test_data_stock_customer_complaint_df$text))
test_data_stock_text.corpus <- tm_map(test_data_stock_text.corpus, tolower)
removeUrl  <- function(x) gsub("http[[:alnum:]]*|http[^[:space:]]*", "", x)
test_data_stock_text.corpus <- tm_map(test_data_stock_text.corpus, removeUrl)
test_data_stock_text.corpus <- tm_map(test_data_stock_text.corpus, content_transformer(gsub), pattern="x*?x",replace=" ")
test_data_stock_text.corpus <- tm_map(test_data_stock_text.corpus, removePunctuation)
test_data_stock_text.corpus <- tm_map(test_data_stock_text.corpus, removeNumbers)
test_data_stock_text.corpus <- tm_map(test_data_stock_text.corpus, content_transformer(gsub), pattern="[^[:alpha:]]",replace=" ")
test_data_stock_text.corpus <-tm_map(test_data_stock_text.corpus,stripWhitespace)
stopwords.tot<- c(stopwords(kind="SMART"),stopwords(kind="fr"),stopwords(kind="es"))
test_data_stock_text.corpus <- tm_map(test_data_stock_text.corpus, removeWords, stopwords.tot)
test_data_stock_text.corpus <-tm_map(test_data_stock_text.corpus,stripWhitespace)
test_data_stock_text.corpus <- tm_map(test_data_stock_text.corpus, stemDocument, language = "english")
test_data_stock_text.corpus <- tm_map(test_data_stock_text.corpus, stemDocument, language = "french")
test_data_stock_text.corpus <- tm_map(test_data_stock_text.corpus, stemDocument, language = "spanish")
#generate document term matrix
test_data_stock_dtm <- tm::DocumentTermMatrix(test_data_stock_text.corpus)
test_data_stock_dtm.tfidf <- tm::weightTfIdf(test_data_stock_dtm)                                 
test_data_stock.tfidf<-removeSparseTerms(test_data_stock_dtm.tfidf,sparse=0.70)
inspect(test_data_stock.tfidf)
test_data_stock.matrix<-as.data.frame(as.matrix(test_data_stock.tfidf))
test_data_stock <- data.frame(cbind(test_data_stock,test_data_stock.matrix))
# drop text column customer complain summary
test_data_stock<-test_data_stock[ , -which(names(test_data_stock) %in% c("Consumer.complaint.summary"))]

#drop columsn as per training model
test_data_stock<-test_data_stock[,-c(2,3,5,6)]
test_data_stock_new <- createDummyFeatures (obj = test_data_stock)
test_data_stock_new<-test_data_stock_new[,-c(21)]

#Train on full data
# Create numeric labels with one-hot encoding
train_labs <- as.numeric(train_data_stock_new$target) - 1

new_train <- model.matrix(~.+0, data = train_data_stock_new[, -c(23)])

# Prepare matrices
xgb_train <- xgb.DMatrix(data = new_train, label = train_labs)

# Calculate # of folds for cross-validation
xgbcv <- xgb.cv(params = params, data = xgb_train, nrounds = 100, nfold = 5, showsd = TRUE, stratified = TRUE, print_every_n = 10, early_stop_round = 20, maximize = FALSE, prediction = TRUE)

# Create the model
xgb_model <- xgb.train(params = params, data = xgb_train, nrounds = 39,watchlist = list(val=xgb_test,train=xgb_train), print.every.n = 10, early.stop.round = 10, maximize = F)


#apply model on test data
new_test_stock <- model.matrix(~.+0, data = test_data_stock_new)

# Prepare matrices
xgb_test_stock <- xgb.DMatrix(data = new_test_stock)
xgb_model$feature_names<-names(test_data_stock_new)

# Predict for test set
xgb_test_stock_preds <- predict(xgb_model, newdata = xgb_test_stock)

xgb_test_stock_out <- matrix(xgb_test_stock_preds, nrow = 5, ncol = length(xgb_test_stock_preds) / 5) %>% 
  t() %>%
  data.frame() %>%
  mutate(max = max.col(., ties.method = "last"))

head(xgb_test_stock_out)


test_result<-data.frame(cbind(test_stock$Complaint.ID,xgb_test_stock_out$max))

lables<-data.frame(cbind(train_stock_labs+1,as.character(train_data_stock_new$target)))
lables<-lables[!duplicated(lables), ]

test_result$X2 <- lables$X2[match(test_result$X2, lables$X1)]

names(test_result)[1]<-"Complaint-ID"
names(test_result)[2]<-"Complaint-Status"

# writing output files
write.csv(test_result,file="Complaint_Status_predictions3.csv",row.names=FALSE,quote = FALSE)

