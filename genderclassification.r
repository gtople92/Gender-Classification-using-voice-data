# Load the required libraries
library(dplyr)
library(Amelia)
library(ggplot2)
library(corrgram)
library(corrplot)
library(caTools)
library(caret)
library(gains)
library(class)
library(randomForest)
library(e1071)
library(psych)
library(neuralnet)
library(pROC)
library(gmodels)
library(tuneR)
library(psycho)
library(warbleR)

#Load the files
voice.df <- read.csv("voice.csv")

#Exploratory Data Analysis
head(voice.df)
str(voice.df)
summ <- summary(voice.df[,-21])
any(is.na(voice.df))
missmap(voice.df, main="Voice Data - Missings Map",col=c("yellow", "black"), legend=FALSE)
print(summ)
ggplot(voice.df, aes(meanfreq, fill = label)) + geom_histogram( color="black",alpha=0.3 ,bins = 30)
ggplot(voice.df, aes(mode)) + geom_histogram( color= "black",alpha=0.3 ,bins = 30)
smf <- summarise(group_by(voice.df,label),mean(mode))
voice.df[which(voice.df$mode == 0 & voice.df$label == "male"), "mode"] <-smf$`mean(mode)`[2]
voice.df[which(voice.df$mode == 0 & voice.df$label == "female"), "mode"] <- smf$`mean(mode)`[1]
ggplot(voice.df, aes(mode, fill = label)) + geom_histogram( color= "black",alpha=0.3 ,bins = 30)
ggplot(voice.df, aes(modindx, fill = label)) + geom_histogram( color= "black",alpha=0.3 ,bins = 30)
ggplot(voice.df, aes(dfrange, fill = label)) + geom_histogram( color= "black",alpha=0.3 ,bins = 30)

#Seperating Numerical columns
num.cols <- sapply(voice.df, is.numeric)
#Plotting Correlation plot
corr.data <- cor(voice.df[,num.cols])
corrplot(corr.data,method='number')
corrgram(voice.df,order=TRUE, lower.panel=panel.shade,upper.panel=panel.pie, text.panel=panel.txt)
#Function to Assign 1 to male and 0 to female
val <- function(lab){
temp <- 1:length(lab)
for (i in 1:length(lab)) {
if(lab[i] == "male"){
temp[i] <- 1
}
else{
temp[i] <- 0
}
}
return(temp)}

#Principal Component Analysis (PCA)
voice.pca<- prcomp(voice.df[,-21],scale. = T)
summary(voice.pca)
pc_var <- (voice.pca$sdev^2)/sum(voice.pca$sdev^2)
plot(pc_var, xlab = "Principal Component", ylab = "Proportion of Variance Explained", type = "b")
plot(voice.pca, main = "Principal Component Analysis")
voice.pca.imp<-as.data.frame(voice.pca$x[,1:10])
voice.pca.imp$label <- voice.df$label

#Split the data into training and validation data
set.seed(101)
split = sample.split(voice.pca.imp$label, SplitRatio = 0.7)
voice.train <- subset(voice.pca.imp,split== TRUE)
voice.test <- subset(voice.pca.imp,split== FALSE)

#Train and Build logistic regression model using PCA scores
start_time<-Sys.time()
logmodel <- glm(label ~ ., family = binomial(link = 'logit'),data = voice.train)
summary(logmodel)

# Predict the data
fitted.probability <- predict(logmodel,newdata = voice.test[,-11],type = 'response')
end_time<-Sys.time()
time.taken.logit <-end_time-start_time
time.taken.logit <- round(as.numeric(time.taken.logit),2)
fitted.results <- as.factor(ifelse(fitted.probability > 0.5,1,0))
logit.con <- confusionMatrix(as.factor(ifelse(fitted.results=="1", "male", "female")), voice.test[,11])
ct <-as.factor(ifelse(fitted.results=="1", "male", "female"))
CrossTable(ct, voice.test[,11])
print(logit.con$table)
accuracy.logit <- round(logit.con$overall[[1]] * 100 ,2)
print(paste("Accuracy :",accuracy.logit,"%"))
# KNN
start_time<-Sys.time()
acc <- 1:100
for(i in 1:100){
set.seed(101)
predicted.gender.knn <- knn(voice.train[,-11],voice.test[,-11],voice.train[,11],k=i)
c <- confusionMatrix(predicted.gender.knn, voice.test[,11])
acc[i] <- c$overall[[1]] * 100
}

acc <- as.data.frame(acc)
acc$knn <- 1:100
acc$err <- 100- acc$acc
ggplot(acc,aes(x=knn,y=err)) + geom_point()+ geom_line(lty="dotted",color='red')
set.seed(101)
predicted.gender.knn <- knn(voice.train[,-11],voice.test[,-11],voice.train[,11],k=1)
end_time<-Sys.time()
time.taken.knn <-end_time-start_time
time.taken.knn <- round(as.numeric(time.taken.knn),2)
print(time.taken.knn)
conknn <- confusionMatrix(predicted.gender.knn, voice.test[,11])
CrossTable(predicted.gender.knn, voice.test[,11])
print(conknn)
accuracy.knn <- round(conknn$overall[[1]] * 100 ,2)
print(paste("Accuracy :",accuracy.knn,"%"))

#Random Forest
start_time<-Sys.time()
voice.randf.model <- randomForest(label ~ ., data = voice.train, ntree = 500)
print(voice.randf.model)
voice.randf.model$confusion
voice.randf.model$importance
predictedresults <- predict(voice.randf.model,voice.test[,-11])
end_time<-Sys.time()
time.taken.rdf <-end_time-start_time
time.taken.rdf <- round(as.numeric(time.taken.rdf),2)
print(time.taken.rdf)
plot(voice.randf.model)
conrdf <- confusionMatrix(predictedresults,voice.test[,11])
CrossTable(predictedresults,voice.test[,11])
print(conrdf)
accuracy.rdf <- round(conrdf$overall[[1]] * 100 ,2)
print(paste("Accuracy :",accuracy.rdf,"%"))

# SVM algorithm
start_time<-Sys.time()
voice.svm.model <- svm(label ~ ., data= voice.train)
summary(voice.svm.model)
svm.predicted.values <- predict(voice.svm.model,voice.test[,-11],type="class")
end_time<-Sys.time()
time.taken.svm <-end_time-start_time
time.taken.svm <- round(as.numeric(time.taken.svm),2)
print(time.taken.svm)

confsvm <- confusionMatrix(svm.predicted.values,voice.test[,11])
CrossTable(svm.predicted.values,voice.test[,11])
accuracy.svm <- round(confsvm$overall[[1]] * 100 ,2)
print(paste("Accuracy :",accuracy.svm,"%"))

# Neural Network
start_time<-Sys.time()
# Get column names
f <- as.formula(paste("label ~", paste(n[!n %in% "label"], collapse = " + ")))
nn.voice <- neuralnet(f,data= voice.train)
plot(nn.voice, rep = "best")
summary(nn.voice)
pred.voice <- compute(nn.voice,voice.test[,-11])
predicted.class=apply(pred.voice$net.result,1,which.max)-1
predicted.class <- as.factor(predicted.class)
end_time<-Sys.time()
time.taken.nn <-end_time-start_time
time.taken.nn <- round(as.numeric(time.taken.nn),2)
print(time.taken.nn)
confnn <- confusionMatrix(as.factor(ifelse(predicted.class=="1", "male", "female")),voice.test[,11])
print(confnn)
accuracy.nn <- round(confnn$overall[[1]] * 100 ,2)
print(paste("Accuracy :",accuracy.nn,"%"))


# New Data
#Loading the voice file: male voice
snap <- readWave("Recording.wav")
print(snap)
plot(snap@left[30700:31500], type = "l", main = "Snap",xlab = "Time", ylab = "Frequency")
summary(snap)
ad <- autodetec(threshold = 5, env = "hil", ssmooth = 300, power=1,bp=c(0,22), xl = 2, picsize = 2, res = 200, flim= c(1,11), osci = TRUE, wl = 300, ls = FALSE, sxrow = 2, rows = 4, mindur = 0.1, maxdur = 1, set = TRUE)
c <- specan(ad,bp=c(0,1),pd= F)
#Loading the voice file: female voice
snap2 <- readWave("FemaleRecord.wav")
print(snap2)
plot(snap2@left[30700:31500], type = "l", main = "Snap",xlab = "Time", ylab = "Frequency")
summary(snap2)

ad2 <- autodetec(threshold = 5, env = "hil", ssmooth = 300, power=1,bp=c(0,22), xl = 2, picsize = 2, res = 200, flim= c(1,11), osci = TRUE,wl = 300, ls = FALSE, sxrow = 2, rows = 4, mindur = 0.1, maxdur = 1, set = TRUE)
c2 <- specan(ad,bp=c(0,1),pd= F)
#Consolidating the male and female data
newdata <- rbind(c,c2)
#adjusting variable names
newdata$median <- c$freq.median
newdata$Q25 <- c$freq.Q25
newdata$Q75 <- c$freq.Q75
newdata$IQR <- c$freq.IQR
newdata <- newdata[names(newdata) %in% names(voice.df)]
#Mean Imputation of missing data
smf <- summarise(group_by(voice.df,label),mean(maxfun))
newdata$maxfun[1:3] <- round(smf[2,2],7)
newdata$maxfun[4:6] <- round(smf[1,2],7)
newdata$maxfun <- as.numeric(newdata$maxfun)
smf <- summarise(group_by(voice.df,label),mean(minfun))
newdata$minfun[1:3] <- round(smf[2,2],7)
newdata$minfun[4:6] <- round(smf[1,2],7)
newdata$minfun <- as.numeric(newdata$minfun)
smf <- summarise(group_by(voice.df,label),mean(meanfun))
newdata$meanfun[1:3] <- round(smf[2,2],7)
newdata$meanfun[4:6] <- round(smf[1,2],7)
newdata$meanfun <- as.numeric(newdata$meanfun)
smf <- summarise(group_by(voice.df,label),mean(centroid))
newdata$centroid[1:3] <- round(smf[2,2],7)
newdata$centroid[4:6] <- round(smf[1,2],7)
newdata$centroid <- as.numeric(newdata$centroid)
smf <- summarise(group_by(voice.df,label),mean(mode))
newdata$mode[1:3] <- round(smf[2,2],7)
newdata$mode[4:6] <- round(smf[1,2],7)
newdata$mode <- as.numeric(newdata$mode)
newdata$label <- factor("male",levels = c("male","female"))
newdata$label[4:6] <- factor("female",levels = c("male","female"))
new.svm.model <- svm(label ~ ., data= voice.df)
#Predicting the gender of new data
new.predicted.values <- predict(new.svm.model,newdata[,-21],type="class")
#Performance evaluation
confusionMatrix(as.factor(new.predicted.values), newdata[,21])