import torch
from torchvision import models
import torch.nn as nn

def gen_mobile_net(pretrained = True):
    if pretrained == True:
        mobile_net = models.mobilenet_v2(pretrained = pretrained)
    else:
        mobile_net = models.mobilenet_v2()
        
    in_conv_layer = nn.Conv2d(1, 3, 3, padding= 1)
    mobile_net.features = nn.Sequential(in_conv_layer, *list(mobile_net.features))
    mobile_net.classifier[1] = nn.Linear(mobile_net.classifier[1].in_features, 2)
    return mobile_net

def gen_vgg(pretrained = True):
    if pretrained == True:
        vgg = models.vgg11_bn(pretrained = True)
    else:
        vgg = models.vgg11_bn()

    in_conv_layer = nn.Conv2d(1, 3, 3, padding= 1)
    vgg.features = nn.Sequential(in_conv_layer, *list(vgg.features))
    vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, 2)
    return vgg

def gen_squeeze_net(pretrained = True):
    if pretrained:
        squeeze_net = models.squeezenet1_1(pretrained = True)
    else:
        squeeze_net = models.squeezenet1_1()

    in_conv_layer = nn.Conv2d(1, 3, 3, padding= 1)
    squeeze_net.features = nn.Sequential(in_conv_layer, *list(squeeze_net.features))
    
    squeeze_net.classifier[1] = nn.Conv2d(512, 2, (1, 1))
    return squeeze_net