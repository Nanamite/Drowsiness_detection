import os
import shutil

closed_dir = r'kaggle_dataset\closed_eye'
open_dir = r'kaggle_dataset\open_eye'

train_dir = r'data\train'
val_dir = r'data\val'
test_dir = r'data\test'

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)


num_train = 0
num_val = 0
num_test = 0

for idx, img_name in enumerate(os.listdir(closed_dir)): 
    if idx < 15120:
        new_name = f'{num_train}_0.png'
        num_train += 1
        shutil.copyfile(os.path.join(closed_dir, img_name), os.path.join(train_dir, new_name))
    elif idx < 21600:
        new_name = f'{num_val}_0.png'
        num_val += 1
        shutil.copyfile(os.path.join(closed_dir, img_name), os.path.join(val_dir, new_name))
    else:
        new_name = f'{num_test}_0.png'
        num_test += 1
        shutil.copyfile(os.path.join(closed_dir, img_name), os.path.join(test_dir, new_name))

for idx, img_name in enumerate(os.listdir(open_dir)):
    if idx < 15120:
        new_name = f'{num_train}_1.png'
        num_train += 1
        shutil.copyfile(os.path.join(open_dir, img_name), os.path.join(train_dir, new_name))
    elif idx < 21600:
        new_name = f'{num_val}_1.png'
        num_val += 1
        shutil.copyfile(os.path.join(open_dir, img_name), os.path.join(val_dir, new_name))
    else:
        new_name = f'{num_test}_1.png'
        num_test += 1
        shutil.copyfile(os.path.join(open_dir, img_name), os.path.join(test_dir, new_name))