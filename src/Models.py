# ====================================================
# Library
# ====================================================


import scipy as sp
import numpy as np
import pandas as pd

from PIL import ImageFile
# sometimes, you will have images without an ending bit
# this takes care of those kind of (corrupt) images
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn as nn

import timm
import CFG

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





sigmoid = torch.nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
swish = Swish.apply

class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)
swish_layer = Swish_module()

# Model using metadata
class EfficientNetwork(nn.Module):
    def __init__(self,CFG, pretrained=True):
        super(EfficientNetwork, self).__init__()
        self.base_model = timm.create_model(CFG.model_name, pretrained=pretrained)
        #self.dropout = nn.Dropout(0.5)
        
        self.csv = nn.Sequential(nn.Linear(9, 250),
                                 nn.BatchNorm1d(250),
                                 Swish_module(),
                                 nn.Dropout(p=0.2),
                                 
                                 nn.Linear(250, 128),
                                 nn.BatchNorm1d(128),
                                 Swish_module(),
                                 nn.Dropout(p=0.2))
        
        self.classification = nn.Linear(in_features=1000 + 128, out_features=9, bias=True)
        
    def forward(self, image, csv_data):
        out_image = self.base_model(image)
        out_csv = self.csv(csv_data)
        image_csv_data = torch.cat((out_image, out_csv), dim=1)
        out = self.classification(image_csv_data)

        return out

# images only
class CustomMolde(nn.Module):
    def __init__(self,CFG, pretrained=True):
        super(CustomMolde, self).__init__()
        self.base_model = timm.create_model(CFG.model_name, pretrained=pretrained)
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(in_features, out_features=9, bias=True)
        
    def forward(self, image):
        out = self.base_model(image)
        return out