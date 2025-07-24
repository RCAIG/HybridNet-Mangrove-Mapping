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

        #layer1
        self.conv1 = resnet34.conv1
        self.bn1 =resnet34.bn1
        self.relu = resnet34.relu
        
        self.layer1_conv = nn.Conv2d(5, 64, 3, padding = 1)
        self.layer1_bn = nn.BatchNorm2d(64)
        self.layer1_relu = nn.ReLU(inplace = True)
        
        self.layer1 = resnet34.layer1
        
        #layer2
        self.layer2 = resnet34.layer2
        
        #layer3
        self.layer3 = resnet34.layer3
        
        #layer4
        self.layer4 = resnet34.layer4
        
        #layer5
        #self.layer5 = resnet34.layer5
        self.pool_layer5 = nn.MaxPool2d(2, 2, ceil_mode= True)
        self.layer5_1 = BasicBlock(512, 512)
        self.layer5_2 = BasicBlock(512,512)
        self.layer5_3 = BasicBlock(512, 512)
        
        input_size=[128 * 128, 64*64, 32*32, 16*16, 8*8]
        dims=[64, 128, 256, 512, 512]
        proj_size =[64, 64,64,64,32]
        num_heads = 4
        transformer_dropout_rate = 0.1
        

        self.stages_en = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(5):
            stage_blocks = []
            stage_blocks.append(utils.TransformerBlock(input_size=input_size[i], hidden_size=dims[i],
                                                 proj_size=proj_size[i], num_heads=num_heads,
                                                 dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages_en.append(*stage_blocks)
        
        #####################Edge###########
        self.up_conv = nn.Conv2d(512, 128, 3, dilation=8, padding= 8) 
        self.up_bn = nn.BatchNorm2d(128)
        
        self.edge_conv = nn.Conv2d(128, 128, 3, padding =1)
        self.edge_bn = nn.BatchNorm2d(128)
        self.edge_relu = nn.ReLU(inplace = True)
        
        ##############upsampling#############
        self.up_scale64 = nn.Upsample(scale_factor=64, mode="nearest")
        self.up_scale32 = nn.Upsample(scale_factor =32, mode="nearest")
        self.up_scale16 = nn.Upsample(scale_factor =16, mode ="nearest")
        self.up_scale8 =  nn.Upsample(scale_factor =8, mode ="nearest")
        self.up_scale4 = nn.Upsample(scale_factor = 4, mode="nearest")
        self.up_scale2 = nn.Upsample(scale_factor =2 , mode="nearest")
        
        ###########deep feature learning#####
        ####encoder_channels = [ 64, 128, 128, 256, 512]decoder_channels = [(256, 128, 64, 32, 16)]
        
        ############EDGE DIM = 128
        #layer5 
        self.pool5 = nn.MaxPool2d(8, 8, ceil_mode=True) # 8
        
        self.linconv5 = nn.Conv2d(512 + 128 , 512, 3, padding = 1)
        self.linbn5 = nn.BatchNorm2d(512)
        self.linrelu5 = nn.ReLU(inplace = True)
        
        self.conv5_1 = nn.Conv2d(512, 256, 3, padding = 1)
        self.bn5_1 = nn.BatchNorm2d(256)
        self.relu5_1 = nn.ReLU(inplace = True)
        
        self.conv5_2 = nn.Conv2d(256, 256, 3, padding = 1)
        self.bn5_2 = nn.BatchNorm2d(256)
        self.relu5_2 = nn.ReLU(inplace = True)
        
        
        #layer4 
        self.pool4 = nn.MaxPool2d(4, 4, ceil_mode= True)#16 
        
        self.conv4_1 = nn.Conv2d(512 + 128 + 256, 128, 3, padding = 1)
        self.bn4_1 = nn.BatchNorm2d(128)
        self.relu4_1 = nn.ReLU(inplace = True)
        
        self.conv4_2 = nn.Conv2d(128, 128, 3, padding = 1)
        self.bn4_2 = nn.BatchNorm2d(128)
        self.relu4_2 = nn.ReLU(inplace = True)
        
        
        #layer3 
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode= True) #32
        self.conv3_1 = nn.Conv2d(256+ 128 + 128, 64, 3, padding = 1)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.relu3_1 = nn.ReLU(inplace = True)
        
        self.conv3_2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.relu3_2 = nn.ReLU(inplace = True)        
        
        #layer2 
        self.conv2_1 = nn.Conv2d(64+ 128 + 128, 32, 3, padding = 1)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.relu2_1 = nn.ReLU(inplace = True)
        
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding = 1)
        self.bn2_2 = nn.BatchNorm2d(32)
        self.relu2_2 = nn.ReLU(inplace = True) 


        #layer1
        self.up1 = nn.Upsample(scale_factor = 2, mode="nearest")
        self.conv1_1 = nn.Conv2d(32 + 128 + 64, 16, 3, padding = 1)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.relu1_1 = nn.ReLU(inplace = True)
        
        self.conv1_2 = nn.Conv2d(16, 16, 3, padding = 1)
        self.bn1_2 = nn.BatchNorm2d(16)
        self.relu1_2 = nn.ReLU(inplace = True) 
        
        #############label ###########

        self.conv0_1 = nn.Conv2d(16 + 128 + self.in_channels, 16, 3, padding = 1)
        self.bn0_1 = nn.BatchNorm2d(16)
        self.relu0_1 = nn.ReLU(inplace = True)
        
        self.conv0_2 = nn.Conv2d(16, 16, 3, padding = 1)
        self.bn0_2 = nn.BatchNorm2d(16)
        self.relu0_2 = nn.ReLU(inplace = True)         
        
        self.dense = nn.Conv2d(16, 2, 1, padding= 0)
        
        #############edge ############
        ##nn.BCELoss
        self.edge_dense = nn.Conv2d(128, 1, 1, padding = 0)
        
        
        
        #############MVI#############        
        self.mconv0 = nn.Conv2d(2, 3, 1)
        self.mbn0 = nn.BatchNorm2d(3)  
        self.mrelu0 = nn.ReLU(inplace = True) #B, 3, H ,W

        
        ##dual branch
        config = CONFIGS["ViT-B_16"] #B-BASE; L-LARGE; H-HUGE
        self.vit = VisionTransformer(config, num_classes=16, zero_head=False, img_size=256)
        self.mconv1 = nn.Conv2d(16, 16, 3, padding = 1)
        self.mbn1 = nn.BatchNorm2d(16)  
        self.mrelu1 = nn.ReLU(inplace = True)  
        
        self.mconv2 = nn.Conv2d(16, 16, 3,  padding = 1)
        self.mbn2 = nn.BatchNorm2d(16)  
        self.mrelu2 = nn.ReLU(inplace = True) 
        

        
    def forward(self, x):
        B, C, H, W = x.shape
        # MVI = (NIR - G) / (SWIR- G)
        mvi = (torch.abs(x[:,4,...] - x[:,1,...])) / (torch.abs(x[:,3,...] - x[:,1,...]) + 1e-8 ) # B 1 H W
        mvi = mvi.view(B, 1, H, W)

        #NDWI = (GREEN-SWIR)/(GREEN+SWIR)
        ndwi = (x[:,4,...] - x[:,3,...]) / (x[:,4,...] + x[:,3,...] + 1e-8)
        ndwi = ndwi.view(B, 1, H, W)

        mvi = torch.cat([mvi, ndwi], dim=1) # B 2 H W
        
        ###feature embedding###
        l_1feature = self.relu(self.bn1(self.conv1(x)))
        
        l_1feature = self.layer1(l_1feature)
        #l_1feature = self.stages_en[0](l_1feature)
        
        l_2feature = self.layer2(l_1feature)
        #l_2feature = self.stages_en[1](l_2feature)        
        
        l_3feature = self.layer3(l_2feature)
        #l_3feature = self.stages_en[2](l_3feature)  
        
        l_4feature = self.layer4(l_3feature)
        #l_4feature = self.stages_en[3](l_4feature)          

        #l_5feature = self.layer5(l_5feature)
        l_5feature = self.pool_layer5(l_4feature)
        l_5feature = self.layer5_1(l_5feature)
        l_5feature = self.layer5_2(l_5feature)
        l_5feature = self.layer5_3(l_5feature)
        #l_5feature = self.stages_en[4](l_5feature)
        
        #print(l_4feature.shape, l_5feature.shape)        
        ########Edge###########
        l_5feature_up = self.up_scale8(l_5feature)

        l_5feature_up = self.up_bn(self.up_conv(l_5feature_up))
        #print(l_5feature_up.shape, l_2feature.shape)        
        feature_edge = l_2feature + l_5feature_up
        feature_edge = self.edge_relu(self.edge_bn(self.edge_conv(feature_edge)))
        
        ############deep feature learning#####
        ##layer5
        t = self.pool5(feature_edge)
        t = torch.cat([t, l_5feature], dim=1)
        t = self.linrelu5(self.linbn5(self.linconv5(t)))
        
        l5_feature_d = self.relu5_1(self.bn5_1(self.conv5_1(t)))
        l5_feature_d = self.relu5_2(self.bn5_2(self.conv5_2(l5_feature_d)))        
        
        l4_feature_d = self.up_scale2(l5_feature_d)
        
        ##layer4
        t = self.pool4(feature_edge)
        t = torch.cat([t, l4_feature_d, l_4feature], dim=1)   

        l4_feature_d = self.relu4_1(self.bn4_1(self.conv4_1(t)))
        l4_feature_d = self.relu4_2(self.bn4_2(self.conv4_2(l4_feature_d))) 

        l3_feature_d = self.up_scale2(l4_feature_d)
        
        ##layer3
        t = self.pool3(feature_edge)
        t = torch.cat([t, l3_feature_d, l_3feature], dim=1)   

        l3_feature_d = self.relu3_1(self.bn3_1(self.conv3_1(t)))
        l3_feature_d = self.relu3_2(self.bn3_2(self.conv3_2(l3_feature_d)))   

        l2_feature_d = self.up_scale2(l3_feature_d)
        
        ##layer2
        t = torch.cat([feature_edge, l2_feature_d, l_2feature], dim=1)
        
        #print(l2_feature_d.shape, l_2feature.shape)
        l2_feature_d = self.relu2_1(self.bn2_1(self.conv2_1(t)))
        l2_feature_d = self.relu2_2(self.bn2_2(self.conv2_2(l2_feature_d)))
        
        l1_feature_d = self.up_scale2(l2_feature_d)
        
        ##layer1
        t = self.up1(feature_edge)
        t = torch.cat([t, l1_feature_d, l_1feature], dim = 1)
        
        l1_feature_d = self.relu1_1(self.bn1_1(self.conv1_1(t)))
        l1_feature_d = self.relu1_2(self.bn1_2(self.conv1_2(l1_feature_d)))
        
        l0_feature_d = self.up_scale2(l1_feature_d)
        ###final###
        
        t = self.up_scale4(feature_edge)
        t = torch.cat([t, l0_feature_d, x], dim = 1) 
        
        l0_feature_out = self.relu0_1(self.bn0_1(self.conv0_1(t)))
        
        
        ##MVI attention###
        mvi = self.mrelu0(self.mbn0(self.mconv0( mvi)))
        logits, atten_wei = self.vit( mvi) # logits B, HW, NUM_CLASS
        _, n_patch, hidden = logits.shape
        h_s, w_s = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        logits = logits.permute(0, 2, 1)
        logits = logits.contiguous().view(B, hidden, h_s, w_s)
        logits_up16 = self.up_scale16(logits) 
        
        
        
        logits_up16 = self.mrelu1(self.mbn1(self.mconv1( logits_up16)))
        logits_up16 = self.mrelu2(self.mbn2(self.mconv2( logits_up16)))        
        
        #atten_wei = self.atten_wei(self.mrelu3(self.mbn3(self.mconv3( atten_wei))))
        
        l0_feature_out = l0_feature_out * logits_up16
        
        
        l0_feature_out = self.relu0_2(self.bn0_2(self.conv0_2(l0_feature_d)))
        
        l0_feature_out = self.dense(l0_feature_out)
        
        ###edge_output###
        edge_out = self.edge_dense(feature_edge)
        edge_out = self.up_scale4(edge_out)
        
        return torch.sigmoid(edge_out), l0_feature_out
      
        
        
           
        
         
    
if __name__ == "__main__":
    model = EDGE_net(5)

    input = torch.rand(size= (1, 5, 256, 256))
    edge, output = model(input)

    #print(output)
    print(edge.shape)
    print(output.shape)