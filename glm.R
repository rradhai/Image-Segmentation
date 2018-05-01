

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