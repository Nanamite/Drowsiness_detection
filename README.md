# Drowsiness_detection - (Project - EE2802, PRML)

***************************************
K R Nandakishore\n
K R Nandakishore

EE21BTECH11027
***************************************

Dataset is a modified version of a kaggle dataset. Link to the kaggle dataset:
https://www.kaggle.com/datasets/kutaykutlu/drowsiness-detection/data
Haar XML files obtained from: https://github.com/opencv/opencv/tree/master/data/haarcascades.

### To train and validate models:

model_train_validate.py is used to train and validate a model with the provided dataset. It also gives the test accuracy of the best weights obtained in the entire run. Copy the following code onto the terminal to run it.
```
python model_train_validate.py --model mobile_net --pretrained True --batch_size 64 --epochs 10 --learning_rate 1e-4
```
Only mobile_net and squeeze_net is supported for training here, to include other models necessary changes are to be made in the source code. The program will save all the weights and summary file in a folder. Here it will save in mobile_net_saves. It will save all other multiple runs within the same folder with appropriate indexing. The first run will be saved in model_name_saves\1 and so on.

### To run example implementation

main.py can be executed to run the example implementation. Example runs have been recorded and are saved in implementation_recording.

Copy the following code snippet onto terminal to run the example implementation:
```
python main.py --model squeeze_net --weights squeeze_net_best_weights\best_model.pth
```
