# Image Segmentation

##Image Segmentation - Project Goal 

The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze.

Segmentation is generally the first stage in any attempt to analyze or interpret an image automatically. Segmentation bridges the gap between low-level image processing and high-level image processing.

This has application in various areas:
 Industrial inspection
 Optical character recognition (OCR) 
 Tracking of objects in a sequence of images 
 Classification of terrains visible in satellite images. 
 Detection and measurement of bone, tissue, etc., in medical images.


##Dataset
I downloaded the data available in the following link for machine learning purpose:
[Image Segmentation](https://archive.ics.uci.edu/ml/datasets/Image+Segmentation)

##My approach in classifying this data:
1. Data Cleaning
  The dataset does not have null values. Most of the field values are pre-processed and hence does not seem to need an extensive data cleaning.
  The classification field does not have a field name, the field names has "." character. I need to do correct them.
2. Pre-processing:
  Use the R functions like: summary, cor, plot, lm, preprocessing commands in "caret" package to preprocess and understand the relationship between different columns and the Class variable.
3. Create a model that will classify the data accurately into seven classes.
4. Test the model on the test data. I tried with SVM (Support Vector Machine) algorithm, GLM, Tree and RandomForest models with cross validation technique for this classification problem.

