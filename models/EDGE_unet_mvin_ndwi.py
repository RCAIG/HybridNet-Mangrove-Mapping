import torch
import torch.nn as nn
from typing import List
from collections import OrderedDict

import _util as utils

import numpy as np
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
import timm
import functools
import torch.utils.model_zoo as model_zoo

from .modeling import VisionTransformer, CONFIGS


class EncoderMixin:
    """Add encoder functionality such as:
    - output channels specification of feature tensors (produced by encoder)
    - patching first convolution for arbitrary input channels
    """

    _output_stride = 32

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)

    def set_in_channels(self, in_channels, pretrained=True):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        utils.patch_first_conv(model=self, new_in_channels=in_channels, pretrained=pretrained)

    def get_stages(self):
        """Override it in your implementation"""
        raise NotImplementedError

    def make_dilated(self, output_stride):

        if output_stride == 16:
            stage_list = [
                5,
            ]
            dilation_list = [
                2,
            ]

        elif output_stride == 8:
            stage_list = [4, 5]
            dilation_list = [2, 4]

        else:
            raise ValueError("Output stride should be 16 or 8, got {}.".format(output_stride))

        self._output_stride = output_stride

        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            utils.replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )
            
from copy import deepcopy

class ResNetEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


##get_encoder in smp && _resnet in torch.models.resnet
def get_encoder(in_channels= 3, weights= "ImageNet", **kwargs):
    params = {"out_channels": (3, 64, 128, 128, 256, 512),
    "block": BasicBlock,
    "layers": [3, 4, 6, 3]}
    model = ResNetEncoder(**params)
    
    if weights is not None:
        try:
            settings =  {
            "url": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
            }
        except KeyError:
            raise KeyError(
                "Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights
                )
            )
    model.load_state_dict(model_zoo.load_url(settings["url"],progress=True))

    model.set_in_channels(in_channels, pretrained=weights is not None)
    
    return model
    


class EDGE_net(nn.Module):
    def __init__(self, in_channels = 3):
        super().__init__()

        self.in_channels = in_channels
        resnet34 = get_encoder(in_channels = self.in_channels)        
        #######Multi stage feature #########


        

        
    def forward(self, x):

        
        feature_out = self.resnet34(x)
        #feature_out = self.up_scale4(edge_out)
        
        return feature_out
      
        
        
           
        
         
    
if __name__ == "__main__":


    input = torch.rand(size= (1, 5, 256, 256))
