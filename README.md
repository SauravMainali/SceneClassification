## Scene Classification

This project is mainly about image classification. The goal here is to classify Scenes around the globe into one of six possible classes using Deep Neural Networks. The application of scene classification could range from organization of photos in Smartphones, assist in growth of country’s economy through Tourism Planning and so on.

Dataset is the intel image classification from Kaggle. (https://www.kaggle.com/puneet6060/intel-image-classification) 
There are 25,000 images in total, and 17,000 of them are labelled into the 6 categories, including Buildings, Forest, Glacier, Mountain, Sea and Street. Models are trained using images in the training set and predict the categories of images in the prediction set.

Both self-trained model and pre-trained model were experimented with in this project.The performance of all these models are compared and analyzed. 



#### Instructions on how to run the code :----

data_prep.py -- This file allow us to load the data.

vgg16.py vgg19.py res.py inception.py inceptionRes.py -- These five files use pre-trained networks without data augmentation.

aug_plots.py -- This file plots how selected picture is augmented.

vgg16_da.py vgg19_da.py ResNet152_da.py InceptionV3_da.py InceptionResV2_da.py -- These five files use pre-trained networks with data augmentation.

Self_trained_cnn.py -- This file performs our self-trained network.

vgg_16__Prediction.py vgg_19__Prediction.py ResNet__Prediction.py Self_trained_Prediction.py -- These files give us the results of prediction.
