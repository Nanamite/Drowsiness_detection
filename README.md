# Drowsiness_detection - (Project - EE2802, PRML)

***************************************
K R Nandakishore\n
K R Nandakishore

EE21BTECH11027
***************************************

Dataset is a modified version of a kaggle dataset. Link to the kaggle dataset:
https://www.kaggle.com/datasets/kutaykutlu/drowsiness-detection/data
Haar XML files obtained from: https://github.com/opencv/opencv/tree/master/data/haarcascades

The main implementation can be executed by running main.py. utilities.py consists of supporting functions for main.py. 

The face and eye detection module is coded within detect_eye.py. Here OpenCV's Cascade Classifiers are used. 

#### To train and validate models:

model_train_validate.py is used to train and validate a model with the provided dataset. It also gives the test accuracy of the best weights obtained in the entire run.
