setwd("C:/Users/Chandra/Desktop/Brainwaves/ML/clustering/dataset")
news_data<-read.csv("news.csv",h=TRUE)
news_data$alltext<-paste(news_data$headline,news_data$text)
news_df<-data.frame(news_data$alltext)
names(news_df)<-"text"
library(tm)
library(RTextTools)
library(SnowballC)
library(textstem)
library(proxy)
library(ggplot2)
library(cluster)
library(clValid)

# preprocessing
news.corpus <- Corpus(VectorSource(news_df$text))
news.corpus <- tm_map(news.corpus, tolower)
removeUrl  <- function(x) gsub("http[[:alnum:]]*|http[^[:space:]]*", "", x)
news.corpus <- tm_map(news.corpus, removeUrl)
news.corpus <- tm_map(news.corpus, content_transformer(gsub), pattern="<.*?>",replace=" ")
news.corpus <- tm_map(news.corpus, removePunctuation)
news.corpus <- tm_map(news.corpus, removeNumbers)
news.corpus <- tm_map(news.corpus, content_transformer(gsub), pattern="[^a-zA-Z]",replace=" ")
news.corpus <-tm_map(news.corpus,stripWhitespace)
news.corpus <- tm_map(news.corpus, removeWords, stopwords(kind = "SMART"))
news.corpus <-tm_map(news.corpus,stripWhitespace)
news.corpus <- tm_map(news.corpus, stemDocument, language = "english")

#generate distance matrix
dtm <- tm::DocumentTermMatrix(news.corpus,control = list(minWordLength = 2))
dtm.tfidf <- tm::weightTfIdf(dtm)

#Feature reduction
news.tfidf<-removeSparseTerms(dtm.tfidf,sparse=0.60)
inspect(news.tfidf)
news.matrix<-as.matrix(news.tfidf)
distmatrix<-dist(news.matrix,method="euclidean")
dist.matrix<-as.matrix(distmatrix)

#choosing the algorithm
intern <- clValid(dist.matrix, nClust = 2:10, 
                  clMethods = c("hierarchical","kmeans","pam"),
                  validation = "internal")
summary(intern)


#using  for evaluation
silhouette_score <- function(x){
  hr <- hclust(distmatrix,method="ward.D2")
  cut<-cutree(hr, k = x)
  ss <- silhouette(cut, distmatrix)
  mean(ss[,3])
}
avg_sil <- sapply(2:10, silhouette_score)
plot(2:10, type='b', avg_sil, xlab='Number of clusters', ylab='Average Silhouette Scores', frame=FALSE)

cut_news<-cutree(news.h, k = 2)

sil1 = silhouette(cut_news, distmatrix)
plot(sil1)

#Applying the hierarchcal clusterting algorithm
news.h<-hclust(distmatrix,method="ward.D2")
plot(news.h,cex=0.1,hang=-1,which.plots = 2,main="Word cluster Dendogram")


#plotting hirarchical
news_with_cluster_h <- data.frame(cbind(news_data,cut_news))

points <- cmdscale(distmatrix, k = 2)

plot(points, main = 'Hierarchical clustering', col = as.factor(cut_news), 
     mai = c(0, 0, 0, 0), mar = c(0, 0, 0, 0),  
     xaxt = 'n', yaxt = 'n', xlab = '', ylab = '')


# writing output files
#cluster file
news_cluster<-data.frame(news_with_cluster_h$id,news_with_cluster_h$cut_news-1)
colnames(news_cluster)[1] <- "id"
colnames(news_cluster)[2] <- "cluster"
write.csv(news_cluster,file="news_cluster_chandra.csv",row.names=FALSE)


#distance matrix
news.matrix<-formatC(news.matrix, format = "e", digits = 18)
write.table(news.matrix, file = "news_matrix_chandra.txt",row.names=FALSE, col.names=FALSE,quote = FALSE)


