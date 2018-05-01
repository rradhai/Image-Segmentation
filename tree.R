#Tree without cross validation

## Load the libraries

library(dplyr)
library(ggplot2)
library(stringr)
library(rpart)
library(rpart.plot)
library(caret)
library(e1071)

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


#creating the tree model

imageTree = rpart(Class ~ .,data=imageSegTranscl, method="class",
                  control= rpart.control(minbucket=25))
prp(imageTree)
#The tree has picked quite a set of independent variables, of which HUE_MEAN and RAWRED_MEAN has been used more than once.

#using tree model to predict the test data

predicttree = predict(imageTree,newdata = testTranscl,type="class" )
table(testTranscl$testClass, predicttree)

#accuracy is 88%, that is (29+24+15+30+30+30+27)/210 #.88

#tree with cross validatation
fitcontrol = trainControl(method="cv",number=10)
cartgrid = expand.grid(.cp=(1:50)*.01)
train(Class~.,data=imageSegTranscl, method="rpart", 
      trControl=fitcontrol, tuneGrid=cartgrid)

#Optimal model chosen based on the accuracy with cp=.02.

imageTreeCV = rpart(Class ~ . , method = "class", 
                    data = imageSegTranscl, control=rpart.control(cp=.02))

#predicting the testdata using this model
predictTreeCV= predict(imageTreeCV, newdata=testTranscl, type= "class")
table(testTranscl$testClass,predictTreeCV)

# accuracy is again 88%, (29+24+15+30+30+30+27)/210 #.88
#Cross validation does not seem to improve decision tree's prediction.

#model with repeated cv
trctrl = trainControl(method = "repeatedcv", number=10, repeats = 3)
set.seed(3333)
dtree.fit = train(Class ~ ., data = imageSegTranscl, method = "rpart",
                  parms = list(split= "information"),
                  trControl = trctrl,
                  tuneLength = 10)
dtree.fit
#plotting the tree
prp(dtree.fit$finalModel, box.palette = "Reds", tweak = 1.2)
#predicting the test data using this tree model
pred = predict(dtree.fit, newdata = testTranscl)
confusionMatrix(pred, testTranscl$testClass)
