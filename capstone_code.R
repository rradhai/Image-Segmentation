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


# creating svm model

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

#GLM

library(ISLR)
library(boot)
library(caTools)

#numeric classification variable created
data<-imageSeg%>% mutate(cl=if_else(Class=="BRICKFACE",1,+
                                      if_else(Class=="SKY",2,+
                                                if_else(Class=="FOLIAGE",3,+
                                                          if_else(Class=="CEMENT",4,+
                                                                    if_else(Class=="WINDOW",5,+
                                                                              if_else(Class=="PATH",6,7)))))))
#remove the default class variable
datanum<-data[,-1]
glimpse(datanum)

#removing the near zero variance variables that do not help much in classification
nzv<-nearZeroVar(datanum)
datanumnzv<-datanum[,-nzv]

#sample.split to split the data into training and testing sets
set.seed(1000)
split=sample.split(datanumnzv$cl,SplitRatio = 0.65)
traini=subset(datanumnzv, split == TRUE)
testi=subset(datanumnzv, split==FALSE)

#creating with glm
gl.model = glm(cl ~ .,data = traini)
summary(gl.model)

#creating the glm based on the significant features alone
gl.model1 = glm(cl ~ REGION_CENTROID_COL + REGION_CENTROID_ROW + VALUE_MEAN + SATURATION_MEAN + HUE_MEAN +HEDGE_MEAN + HEDGE_SD + INTENSITY_MEAN, data = traini)
summary(gl.model1)

#comparing both the models
predglm = predict(gl.model, type="response", newdata=testi)
table(testi$cl,predglm>0.5)
predglm1 = predict(gl.model1, type="response", newdata=testi)
table(testi$cl,predglm1>0.5)

#gl.model1 shows a better accuracy
test<-read.table("segmentation.data",skip=3,header=TRUE,sep=",",row.names=NULL)
col1=col
col1[1]<-"testClass"
names(test)<-col1
testdata<-test%>% mutate(cl=if_else(testClass=="BRICKFACE",1,+
                                      if_else(testClass=="SKY",2,+
                                                if_else(testClass=="FOLIAGE",3,+
                                                          if_else(testClass=="CEMENT",4,+
                                                                    if_else(testClass=="WINDOW",5,+
                                                                              if_else(testClass=="PATH",6,7)))))))

testdatanum<-testdata[,-1]
glimpse(testdatanum)
nzv<-nearZeroVar(testdatanum)
testdatanumnzv<-testdatanum[,-nzv]

predtestglm = predict(gl.model1, type="response", newdata=testdatanumnzv)
table(testdatanumnzv$cl,predtestglm>0.95)

#The test data prediction has also shown a very low error rate 

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


#RandomForest model
model.rf = randomForest(Class ~ ., data = imageSeg, 
                        nodesize = 25, ntree = 4000)
print(model.rf)

#Error rate is pretty high. Trying with different node sizes and ntree values.
model.rf2 = randomForest(Class ~ ., data=imageSegTranscl, nodesize=10, 
                         ntree=1000, importance =TRUE)
print(model.rf2)

model.rf3 = randomForest(Class ~ ., data=imageSegTranscl, ntree=500, nodesize=25,
                         importance = TRUE, proximity = TRUE)
print(model.rf3)

#model.rf3 has a lower error rate of 2.24%
round(importance(model.rf2),2)

#prediction on test data
test.rf3 = randomForest(testClass ~ .,data=testTranscl,ntree=1000)
print(test.rf3)

#RandomForest with cross validation
fitcontrol = trainControl(method="cv",number=10)
imageForestCV = train(Class~., data=imageSegTranscl, method="rf", trControl=fitcontrol)
print(imageForestCV)

predictForestCV = predict(imageForestCV, newdata = testTranscl)
table(testTranscl$testClass,predictForestCV)
#accuracy is
(30+30+30+30+30+30+27)/210
#98%
