import torch
import torchvision.transforms as transform
from generate_model import *
from detect import detect
import time
import cv2 as cv
import numpy as np

# if(torch.cuda.is_available()):
#     device = 'cuda'
# else:
#     device = 'cpu'

device = 'cuda'

model = gen_mobile_net(False)
model.load_state_dict(torch.load(r'mobile_net_best_weights\best_model.pth'))
model.to(device)
model.eval()

def conv_image(img, device):
    if(len(img.shape) == 3):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    tensor = transform.ToTensor()
    img = tensor(img)
    normalize = transform.Normalize(mean = [0.335], std = [0.0132])
    img = normalize(img)
    img = img.unsqueeze(0).to(device)
    return img

def predict(img, model):
    output = model(img)
    return (torch.argmax(output).item())

def warmup(model):
    for i in range(100):
        img = np.random.randint(0, 255, (52, 52), dtype= np.uint8)
        img = conv_image(img, device)
        predict(img, model)

#img = cv.imread('eye_open_left.jpg', cv.IMREAD_GRAYSCALE)

warmup(model)

# img = conv_image(img, device)
# stime = time.time()
# print(predict(img, model))
# print(time.time() - stime)

vid = cv.VideoCapture(0)

while True:
    _, frame = vid.read()
    cv.imshow('frame', frame)

    ret, eye_left, eye_right, face = detect(frame)

    if ret:
        eye_left = conv_image(eye_left, device)
        eye_right = conv_image(eye_right, device)
        stime = time.time()
        print(predict(eye_left, model), predict(eye_right, model))
        print(time.time() - stime)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()