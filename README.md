# Drowsiness_detection - (Project - EE2802, PRML)

***************************************
K R Nandakishore\n
K R Nandakishore

EE21BTECH11027
***************************************

Dataset is a modified version of a kaggle dataset. Link to the kaggle dataset:
https://www.kaggle.com/datasets/kutaykutlu/drowsiness-detection/data

Only the processed dataset is available in this repository

Haar XML files obtained from: https://github.com/opencv/opencv/tree/master/data/haarcascades.

### To train and validate models:

model_train_validate.py is used to train and validate a model with the provided dataset. It also gives the test accuracy of the best weights obtained in the entire run. Copy the following code onto the terminal to run it.
```
python model_train_validate.py --model squeeze_net --pretrained True --batch_size 64 --epochs 10 --learning_rate 1e-4 --optimizer Adam
```
Only mobile_net and squeeze_net is supported for training here, to include other models necessary changes are to be made in the source code. The optimizers supported are 'Adam' and 'SGD'. The program will save all the weights and summary file in a folder. Here it will save in mobile_net_saves. It will save all other multiple runs within the same folder with appropriate indexing. The first run will be saved in model_name_saves\1 and so on.

### To run example implementation

main.py can be executed to run the example implementation. Example runs have been recorded and are saved in implementation_recording.

Copy the following code snippet onto terminal to run the example implementation:
```
python main.py --model squeeze_net --weights squeeze_net_best_weights\best_model.pth
```

### Other file details

#### beep.py:

This contains a code snippet that makes a beep sound

#### detect_eye.py

This contains the code snippet that detects the face, left eye and right eye within a frame and returns it

#### generate_model.py

This generates and returns the required pytorch model

#### model_description.py

This generates the summary of mobile_net and squeeze_net

#### organize_Data.py

This was used to organize the kaggle dataset to the one seen in the folder data

#### prepare_data.py

It creates dataloaders and returns it

#### test_acc.py

It tests the accuracy of the given model with the given weights and returns the test accuracy, precision and recall

Copy the following code onto terminal to run it
```
python test_acc.py --model squeeze_net --weights squeeze_net_best_weights\best_model.pth
```

#### utilities.py

Contain supporting code snippets for main.py

#### visualize_dataset.ipynb

A notebook that visualizes the organized dataset

#### visualize_model_output.ipynb

Shows the prediction of model with the respective inference time. Showcases the effect of warmup too
