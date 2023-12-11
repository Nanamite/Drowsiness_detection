import torch
import torchvision.transforms as transform
import cv2 as cv
import numpy as np

def conv_image(img):
    if(len(img.shape) == 3):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    tensor = transform.ToTensor()
    img = tensor(img)
    normalize = transform.Normalize(mean = [0.335], std = [0.0132])
    img = normalize(img)
    img = img.unsqueeze(0)
    return img

def predict(img, model):
    output = model(img)
    return (torch.argmax(output, dim = 1).item())

def warmup(model):
    print("warming up")
    for i in range(200):
        img = np.random.randint(0, 255, (52, 52), dtype= np.uint8)
        img = conv_image(img)
        predict(img, model)
    print("warmup done")