## Load the libraries

library(dplyr)
library(ggplot2)
library(stringr)
library(randomForest)
library(caret)


## Loading the training data

imageSeg<-read.table("segmentation.test",skip=3,header=TRUE,sep=",",row.names=NULL)

##Changing the column names
#Replacing "." to "_" in the column names
#Naming the first column as "Class"

names(imageSeg)
col<-names(imageSeg)
col<-str_replace_all(col,"[.]","_")
col[1]<-"Class"
names(imageSeg)<-col
names(imageSeg)
imageSeg$Class = as.factor(imageSeg$Class)

### Fetching the numeric columns
lapply(imageSeg,class)
imagesegnum<-imageSeg[,-1]
dim(imagesegnum)

### Scaling and centering the data and remove the 
#zero variance and near zero variance features from the data set

preprocimageseg<-preProcess(imagesegnum,method=c("center","scale","zv","nzv"))
imageSegTrans<-predict(preprocimageseg,imagesegnum)
dim(imageSegTrans)
head(imageSegTrans)

###after transformation, binding the class column
Class<-imageSeg$Class
dim(imageSegTrans)

imageSegTranscl<-cbind(Class,imageSegTrans)
dim(imageSegTranscl)


#loading, preprocessing the test data
test<-read.table("segmentation.data", skip=3, header=TRUE, sep=",", 
                 row.names=NULL)
col1=col
col1[1]<-"testClass"
names(test)<-col1
testnum=test[,-1]
testClass<-test[,1]
preproctest<-preProcess(testnum,method=c("center","scale","zv","nzv"))
testTrans<-predict(preproctest,testnum)
testTranscl<-cbind(testClass,testTrans)

#RandomForest model
model.rf = randomForest(Class ~ ., data = imageSeg, nodesize = 25, ntree = 4000)
print(model.rf)

#Error rate is pretty high. Trying with different node sizes and ntree values.
model.rf2 = randomForest(Class ~ ., data=imageSegTranscl, nodesize=20, ntree=1000,importance = TRUE)
print(model.rf2)
round(importance(model.rf2),2)

model.rf3 = randomForest(Class ~ ., data=imageSegTranscl, ntree=500, nodesize=10,
                         importance = TRUE)
print(model.rf3)

model.rf4 = randomForest(Class ~ ., data=imageSegTranscl, ntree=500,
                         importance = TRUE)
print(model.rf4)
#model.rf3 has a lower error rate of 2.24%
round(importance(model.rf2),2)

#prediction on test data
test.rf3 = randomForest(testClass ~ .,data=testTranscl,ntree=500)
print(test.rf3)

#RandomForest with cross validation
fitcontrol = trainControl(method="cv",number=10)
imageForestCV = train(Class~., data=imageSegTranscl, method="rf", trControl=fitcontrol)
print(imageForestCV)
predForestCV = predict(imageForestCV, newdata = imageSegTranscl)
table(imageSegTranscl$Class,predForestCV)

predictForestCV = predict(imageForestCV, newdata = testTranscl)
table(testTranscl$testClass,predictForestCV)
#accuracy is
(30+30+30+30+30+30+27)/210
#98%
