from __future__ import print_function
from collections import namedtuple
from pathlib import Path
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
from torchvision import transforms as trans
import numpy as np
import os
import torch
import time
    
class Config:
    def __init__(self):
        self.net_output = 512
        self.net_depth = 50
        self.drop_ratio = 0.4
        self.net_mode = 'ir_se'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False),
                BatchNorm2d(depth))
            
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3,3), (1,1),1 ,bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3,3), stride, 1 ,bias=False),
            BatchNorm2d(depth),
            SEModule(depth,16)
            )
        
    def forward(self,x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        #res = x
        #for module in self.res_layer:
        #    res = module(res)
        return res + shortcut

class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir', output=512):
        super(Backbone, self).__init__()
        assert num_layers in [34, 50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        
        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, output),
                                       BatchNorm1d(output))
        
        self.body = torch.nn.ModuleList()
        for x in range(3):
            if x == 0:
                self.body.append(bottleneck_IR_SE(64, 64, 2))
            else:
                self.body.append(bottleneck_IR_SE(64, 64, 1))
        
        for x in range(4):
            if x == 0:
                self.body.append(bottleneck_IR_SE(64, 128, 2))
            else:
                self.body.append(bottleneck_IR_SE(128, 128, 1))
                               
        for x in range(14):
            if x == 0:
                self.body.append(bottleneck_IR_SE(128, 256, 2))
            else:
                self.body.append(bottleneck_IR_SE(256, 256, 1))
        
        for x in range(3):
            if x == 0:
                self.body.append(bottleneck_IR_SE(256, 512, 2))
            else:
                self.body.append(bottleneck_IR_SE(512, 512, 1))

    def forward(self, x):
        x = self.input_layer(x)
        for module in self.body:
            x = module(x)
         
        x = self.output_layer(x)
        return l2_norm(x)

def l2_norm(input):
    norm = torch.norm(input,2,1,True)
    output = torch.div(input, norm)
    return output

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks(num_layers):
    if num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

class TestNet(object):
    def __init__(self):
        self.conf = Config()
        self.model = Backbone(self.conf.net_depth, self.conf.drop_ratio, self.conf.net_mode, self.conf.net_output).to(self.conf.device)
        print('{}_{} model generated'.format(self.conf.net_mode, self.conf.net_depth))
        self.threshold = self.conf.threshold

    def load_state(self, path_str):
        self.model.load_state_dict(torch.load(path_str, map_location='cpu'))
        
    def infer(self, img):    
        with torch.no_grad():
            return self.model(self.conf.test_transform(img).to(self.conf.device).unsqueeze(0))