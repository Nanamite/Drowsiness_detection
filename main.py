import torch
import torchvision.transforms as transform
import cv2 as cv
import numpy as np
from detect_eye import detect
from utilities import *
from generate_model import *
from beep import make_beep
import time
import argparse as ap

device = 'cpu'

parser = ap.ArgumentParser()

parser.add_argument('--model', type = str, default = 'mobile_net')
parser.add_argument('--weights', type = str)

args = parser.parse_args()

if args.model == 'mobile_net':
    model = gen_mobile_net(False)
elif args.model == 'squeeze_net':
    model = gen_squeeze_net(False)
else:
    print('model not supported')

model = model.to(device)
model.load_state_dict(torch.load(args.weights))
model.eval()

warmup(model)

vid = cv.VideoCapture(0)
frame_duration = 10
frames_observed = 500/frame_duration

left_eye_preds = []
right_eye_preds = []
alarm = 0
beep_on = 0

print('starting')
print('***********************')

while True:
    ret, frame = vid.read()
    cv.imshow('frame', frame)

    if ret:
        detected, eye_left, eye_right, face = detect(frame)
        if detected:
            eye_left = conv_image(eye_left).to(device)
            eye_right = conv_image(eye_right).to(device)

            eye_left_pred = predict(eye_left, model)
            eye_right_pred = predict(eye_right, model)

            left_eye_preds.append(eye_left_pred)
            right_eye_preds.append(eye_right_pred)

            if(len(left_eye_preds) > frames_observed):
                left_eye_preds.pop(0)
                right_eye_preds.pop(0)  

                if(np.mean(left_eye_preds) < 0.8 and np.mean(right_eye_preds) < 0.8):
                    if not alarm:
                        alarm = 1
        
                if(np.mean(left_eye_preds) > 0.8 and np.mean(right_eye_preds) > 0.8):
                    if alarm:
                        alarm = 0

                if alarm:
                        if not beep_on:
                            beep_on = 1
                            beep_on = make_beep(beep_on)
                if not alarm:
                    if beep_on:
                        beep_on = 0
                print("alarm status: ", alarm, " | left and right eye average status: ", np.mean(left_eye_preds), ", ", np.mean(right_eye_preds))    



    if cv.waitKey(frame_duration) & 0xFF == ord('q'):
        break;



