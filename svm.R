## Load the libraries

library(dplyr)
library(ggplot2)
library(stringr)

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

### Scaling and centering the data and remove the zero variance and near zero variance features from the data set

preprocimageseg<-preProcess(imagesegnum,method=c("center","scale","zv","nzv"))
imageSegTrans<-predict(preprocimageseg,imagesegnum)
dim(imageSegTrans)
head(imageSegTrans)

###after transformation, binding the class column
Class<-imageSeg$Class
dim(imageSegTrans)

imageSegTranscl<-cbind(Class,imageSegTrans)
dim(imageSegTranscl)

#SVM

library(e1071) 

### Separating the independent variables and classification variable

x<-subset(imageSegTranscl,select=-Class)
y<-imageSegTranscl$Class

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


# create a model

svm.model<-svm(Class~.,data=imageSegTranscl)
summary(svm.model)

#testing the model using predict
pred<-predict(svm.model,x)
table(pred,y) #confusion matrix

### Accuracy of this svm.model is 95%

#tuning the SVM model
svm_tune<-tune(svm,train.x=x,train.y=y,kernel="radial",
               ranges=list(cost=10^(-2:3),gamma=c(.5,1,1.5,2)))

print(svm_tune)

#creating a better model based on tuning
svm.model.tuned<-svm(Class~., data=imageSegTranscl,cost=10,gamma=0.5)
summary(svm.model.tuned)

#accuracy is 99%


#validating tuned model on test data

y2<-testClass
x2<-subset(testTranscl,select=-testClass)

predtest<-predict(svm.model.tuned,x2)
table(predtest,y2)
#accuracy is 99%
