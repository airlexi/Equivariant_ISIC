from typing import Tuple
import torch
import torch.nn.functional as F
import copy
import numpy as np
import torchvision
import math
import e2cnn.nn as enn
from e2cnn.nn import init
from e2cnn import gspaces
import torch.nn as nn
from collections import OrderedDict
   
    
def regular_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = True):
    """ build a regular feature map with the specified number of channels"""
    assert gspace.fibergroup.order() > 0
    
    N = gspace.fibergroup.order()
    
    if fixparams:
        planes *= math.sqrt(N)
    
    planes = planes / N
    planes = int(planes)
    
    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def trivial_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = True):
    """ build a trivial feature map with the specified number of channels"""
    
    if fixparams:
        planes *= math.sqrt(gspace.fibergroup.order())
        
    planes = int(planes)
    return enn.FieldType(gspace, [gspace.trivial_repr] * planes)


FIELD_TYPE = {
    "trivial": trivial_feature_type,
    "regular": regular_feature_type,
}


def conv7x7(in_type: enn.FieldType, out_type: enn.FieldType, stride=2, padding=1,
            dilation=1, bias=False):
    """3x3 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      #initialize = False,
                      )

def conv3x3(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=1,
            dilation=1, bias=False):
    """3x3 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      #initialize = False,
                      )


def conv1x1(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=0,
            dilation=1, bias=False):
    """1x1 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 1,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      #initialize= False
                      )



class BottleneckBlock(enn.EquivariantModule):
    """BottleneckBlock for DenseNets, consisting of batchnormalization -> ReLU -> 1x1 Convolution -> batchnormalization -> ReLU -> 3x3 Convolution, and dropout with probability of 0.5 added as well"""
    
    
    def __init__(self, in_type, inner_type, out_type, stride=1):
        super(BottleneckBlock, self).__init__()
        
        self.bn1 = enn.InnerBatchNorm(in_type)
        self.relu1 = enn.ReLU(in_type,inplace=True)
        self.conv1 = conv1x1(in_type, inner_type)
        self.bn2 = enn.InnerBatchNorm(inner_type)
        self.relu2 = enn.ReLU(inner_type,inplace=True)
        self.conv2 = conv3x3(inner_type, out_type)
        
        self.dropout = enn.PointwiseDropout(inner_type, p=0.5)
        self.dropout2 = enn.PointwiseDropout(out_type, p=0.5)
        
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.dropout2(out)
        out = enn.tensor_directsum([x,out])
        return out
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape
    

class TransitionBlock(enn.EquivariantModule):
    """TransitionBlock of DenseNets to increase the size of channels between blocks"""
    
    
    def __init__(self, in_type, out_type, gspace):
        super(TransitionBlock, self).__init__()
        self.gspace = gspace
        self.in_type = FIELD_TYPE["regular"](self.gspace, in_type, fixparams=False)
        self.out_type = FIELD_TYPE["regular"](self.gspace, out_type, fixparams=False)
        
        self.bn1 = enn.InnerBatchNorm(self.in_type)
        self.relu1 = enn.ReLU(self.in_type,inplace=True)
        self.conv1 = conv1x1(self.in_type,self.out_type)
        self.avgpool = enn.PointwiseAvgPool(self.out_type, kernel_size=2)
        
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.avgpool(out)
        return out
        
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape
        
        
        
        
class DenseBlock(enn.EquivariantModule):
    """DenseBlock, creating dense blocks as proposed in the paper"""
    
    
    def __init__(self, in_type, growth_rate, gspace, list_layer):
        super(DenseBlock, self).__init__()
        self.gspace = gspace
        
        self.layer = self._make_layer(in_type, growth_rate, list_layer)
    
    def _make_layer(self, in_type, growth_rate, num_layers):
        layers = []
        
        for i in range(num_layers):
            layers.append(BottleneckBlock(FIELD_TYPE["regular"](self.gspace, i*growth_rate+in_type, fixparams=False),
                                          FIELD_TYPE["regular"](self.gspace, 4*growth_rate, fixparams=False),
                                          FIELD_TYPE["regular"](self.gspace, growth_rate, fixparams=False)
                                         )
                         )
        layers = torch.nn.Sequential(*layers)
        return layers
    
    def forward(self, x):
        return self.layer(x)
    
    
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape

        
        
        
class DenseNet(torch.nn.Module):
    """equivariant DenseNet implementation as proposed by Huang, Liu et al. in 'Densely Connected Convolutional Networks'(https://arxiv.org/abs/1608.06993) adapted from https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py"""
    
    
    def __init__(self, growth_rate, list_layer, n_meta_features):
        super(DenseNet, self).__init__()
        
        self.gspace = gspaces.Rot2dOnR2(N=8)
        
        in_type = 2*growth_rate
        
        self.conv1 = conv7x7(FIELD_TYPE["trivial"](self.gspace, 3, fixparams=False), 
                             FIELD_TYPE["regular"](self.gspace, in_type, fixparams=False))
        
        self.relu1 = enn.ELU(FIELD_TYPE["regular"](self.gspace, in_type, fixparams=False), inplace = True)
        self.pool1 = enn.PointwiseMaxPool(FIELD_TYPE["regular"](self.gspace, in_type, fixparams=False),
                                          kernel_size=3, stride=2, padding=1)
        
        
        #1st block
        self.block1 = DenseBlock(in_type, growth_rate, self.gspace, list_layer[0])
        in_type = in_type +list_layer[0]*growth_rate
        self.trans1 = TransitionBlock(in_type, int(in_type/2), self.gspace)
        in_type = int(in_type/2)
        
        #2nd block
        self.block2 = DenseBlock(in_type, growth_rate, self.gspace, list_layer[1])
        in_type = in_type +list_layer[1]*growth_rate
        self.trans2 = TransitionBlock(in_type, int(in_type/2), self.gspace)
        in_type = int(in_type/2)
        
        #3rd block
        self.block3 = DenseBlock(in_type, growth_rate, self.gspace, list_layer[2])
        in_type = in_type +list_layer[2]*growth_rate
        self.trans3 = TransitionBlock(in_type, int(in_type/2), self.gspace)
        in_type = int(in_type/2)
        
        #4th block
        self.block4 = DenseBlock(in_type, growth_rate, self.gspace, list_layer[3])
        in_type = in_type +list_layer[3]*growth_rate
        
        
        self.bn = enn.InnerBatchNorm(FIELD_TYPE["regular"](self.gspace, in_type, fixparams=False))
        self.relu = enn.ReLU(FIELD_TYPE["regular"](self.gspace, in_type, fixparams=False),inplace=True)
        self.pool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(in_type, 500)
        
        #this part consists of fully connected layers to fit the one-hot encoded meta-data
        self.meta = torch.nn.Sequential(torch.nn.Linear(n_meta_features, 500),
                                  torch.nn.BatchNorm1d(500),
                                  torch.nn.ReLU(),
                                  torch.nn.Dropout(p=0.2),
                                  torch.nn.Linear(500, 250),  
                                  torch.nn.BatchNorm1d(250),
                                  torch.nn.ReLU(),
                                  torch.nn.Dropout(p=0.2))
        self.output = torch.nn.Linear(500+250,1)
        
        
    def forward(self, inputs):
        x, meta = inputs
        x = enn.GeometricTensor(x, enn.FieldType(self.gspace, 3*[self.gspace.trivial_repr]))
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.trans1(out)
        
        out = self.block2(out)
        out = self.trans2(out)
        
        out = self.block3(out)
        out = self.trans3(out)
        
        out = self.block4(out)
        out = self.relu(self.bn(out))
        out = out.tensor
        
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        meta_features = self.meta(meta)
        features = torch.cat((out, meta_features), dim=1)
        output = self.output(features)
        return output


    
    
    
class ResBlock(enn.EquivariantModule):
    """Residual Block consisting of 1x1 convolution -> batchnormalization -> ReLU -> 3x3 convolution -> batchnormalization -> ReLU -> 1x1 convolution -> batchnormalization -> ReLU and a skip-connection"""
    
    
    def __init__(self, in_type,inner_type, out_type, stride=1, dropout_rate=0.0):
        super(ResBlock, self).__init__()
        
        self.in_type = in_type
        self.inner_type = inner_type
        self.out_type = out_type
        
        self.conv1 = conv1x1(self.in_type, self.inner_type, stride = 1, bias = False)
        self.bn1 = enn.InnerBatchNorm(self.inner_type)
        self.relu1 = enn.ReLU(self.inner_type)
        
        self.conv2 = conv3x3(self.inner_type, self.inner_type, padding=1, stride = stride, bias = False)
        self.bn2 = enn.InnerBatchNorm(self.inner_type)
        self.relu2 = enn.ReLU(self.inner_type, inplace=True)
        
        self.conv3 = conv1x1(self.inner_type, self.out_type, stride = 1, bias = False)
        self.bn3 = enn.InnerBatchNorm(self.out_type)
        self.relu3 = enn.ReLU(self.out_type, inplace=True)
        
        
        self.shortcut = None
        if stride != 1 or self.in_type != self.out_type:
            self.shortcut = enn.R2Conv(self.in_type, self.out_type, kernel_size=1, stride=stride, bias=False)
            
    def forward(self, x):
        
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity   
        out = self.relu3(out)
        return out
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape
        
        

class ResNet(torch.nn.Module):
    """equivariant Resnet implementation as proposed by He, Zhang et al. in 'Deep Residual Learning for Image Recognition' (https://arxiv.org/abs/1512.03385) adapted from: """
    
    
    def __init__(self, list_layer, n_meta_features, p_drop=0.0):
        super(ResNet, self).__init__()
        
        self.gspace = gspaces.Rot2dOnR2(N=8)
        
        reg_field64 = FIELD_TYPE["regular"](self.gspace, 64, fixparams=False)
        reg_field256 = FIELD_TYPE["regular"](self.gspace, 256, fixparams=False)
        reg_field128 = FIELD_TYPE["regular"](self.gspace, 128, fixparams=False)
        reg_field512 = FIELD_TYPE["regular"](self.gspace, 512, fixparams=False)
        reg_field1024 = FIELD_TYPE["regular"](self.gspace, 1024, fixparams=False)
        reg_field2048 = FIELD_TYPE["regular"](self.gspace, 2048, fixparams=False)
        
        self.conv1 = enn.R2Conv(FIELD_TYPE["trivial"](self.gspace, 3, fixparams=False), reg_field64, kernel_size=7, stride=2, padding=3)
        
        self.bn1 = enn.InnerBatchNorm(reg_field64)
        self.relu1 = enn.ReLU(reg_field64)
        self.maxpool1 = enn.PointwiseMaxPool(reg_field64, kernel_size=3, stride=2, padding=1)
        
        layer1 = []
        layer1.append(ResBlock(stride=2, in_type = reg_field64, inner_type = reg_field64, out_type = reg_field256, dropout_rate=p_drop))
        for i in range(1, list_layer[0]):
            layer1.append(ResBlock(stride=1, in_type = reg_field256, inner_type = reg_field64, out_type = reg_field256, dropout_rate=p_drop))
        self.layer1 = torch.nn.Sequential(*layer1)
        
        layer2 = []
        layer2.append(ResBlock(stride=2, in_type = reg_field256, inner_type = reg_field128, out_type = reg_field512, dropout_rate=p_drop))
        for i in range(list_layer[1]):
            layer2.append(ResBlock(stride=1, in_type = reg_field512, inner_type = reg_field128, out_type = reg_field512, dropout_rate=p_drop))
        self.layer2 = torch.nn.Sequential(*layer2)
        
        layer3 = []
        layer3.append(ResBlock(stride=2, in_type = reg_field512, inner_type = reg_field256, out_type = reg_field1024, dropout_rate=p_drop))
        for i in range(list_layer[2]):
            layer3.append(ResBlock(stride=1, in_type = reg_field1024, inner_type = reg_field256, out_type = reg_field1024,dropout_rate=p_drop))
        self.layer3 = torch.nn.Sequential(*layer3)
        
        layer4 = []
        layer4.append(ResBlock(stride=2, in_type = reg_field1024, inner_type = reg_field512, out_type = reg_field2048, dropout_rate=p_drop))
        for i in range(list_layer[3]):
            layer4.append(ResBlock(stride=1, in_type = reg_field2048, inner_type = reg_field512, out_type = reg_field2048, dropout_rate=p_drop))
        self.layer4 = torch.nn.Sequential(*layer4)
        
        self.pool = enn.PointwiseAdaptiveAvgPool(reg_field2048, (1, 1))
        self.fc = torch.nn.Linear(2048, 500)
        
        #this part consists of fully connected layers to fit the one-hot encoded meta-data
        self.meta = torch.nn.Sequential(torch.nn.Linear(n_meta_features, 500),
                                  torch.nn.BatchNorm1d(500),
                                  torch.nn.ReLU(),
                                  torch.nn.Dropout(p=0.2),
                                  torch.nn.Linear(500, 250),  # FC layer output will have 250 features
                                  torch.nn.BatchNorm1d(250),
                                  torch.nn.ReLU(),
                                  torch.nn.Dropout(p=0.2))
        self.output = torch.nn.Linear(500+250,1)
        
        
    def forward(self, inputs):
        x, meta = inputs
        x = enn.GeometricTensor(x, enn.FieldType(self.gspace, 3*[self.gspace.trivial_repr]))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.tensor
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        meta_features = self.meta(meta)
        features = torch.cat((x, meta_features), dim=1)
        output = self.output(features)
        return output
        
        
class BasicConv2d(enn.EquivariantModule):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.gspace = gspaces.Rot2dOnR2(N=8)
        self.conv = enn.R2Conv(in_channels, out_channels, bias=False, **kwargs)
        self.bn = enn.InnerBatchNorm(out_channels)
        self.relu = enn.ReLU(out_channels, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape
    
    
    
class Inception3(torch.nn.Module):

    def __init__(self, num_classes=1, aux_logits=False, 
                 inception_blocks=None):
        super(Inception3, self).__init__()
        
        self.gspace = gspaces.Rot2dOnR2(N=8)
        
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        reg_field32 = FIELD_TYPE["regular"](self.gspace, 32, fixparams=False)
        reg_field64 = FIELD_TYPE["regular"](self.gspace, 64, fixparams=False)
        reg_field80 = FIELD_TYPE["regular"](self.gspace, 80, fixparams=False)
        reg_field192 = FIELD_TYPE["regular"](self.gspace, 192, fixparams=False)
        reg_field256 = FIELD_TYPE["regular"](self.gspace, 256, fixparams=False)
        reg_field288 = FIELD_TYPE["regular"](self.gspace, 288, fixparams=False)
        reg_field768 = FIELD_TYPE["regular"](self.gspace, 768, fixparams=False)
        reg_field160 = FIELD_TYPE["regular"](self.gspace, 160, fixparams=False)
        reg_field128 = FIELD_TYPE["regular"](self.gspace, 128, fixparams=False)
        reg_field160 = FIELD_TYPE["regular"](self.gspace, 160, fixparams=False)
        reg_field192 = FIELD_TYPE["regular"](self.gspace, 192, fixparams=False)
        reg_field1280 = FIELD_TYPE["regular"](self.gspace, 1280, fixparams=False)
        reg_field2048 = FIELD_TYPE["regular"](self.gspace, 2048, fixparams=False)
        
        
        
        self.aux_logits = aux_logits
        self.Conv2d_1a_3x3 = conv_block(FIELD_TYPE["trivial"](self.gspace, 3, fixparams=False),
                                        reg_field32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(reg_field32, reg_field32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(reg_field32, reg_field64, kernel_size=3, padding=1)
        self.maxpool1 = enn.PointwiseMaxPool(reg_field64, kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(reg_field64, reg_field80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(reg_field80, reg_field192, kernel_size=3)
        self.maxpool2 = enn.PointwiseMaxPool(reg_field192,kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(reg_field192, pool_features=reg_field32)
        self.Mixed_5c = inception_a(reg_field256, pool_features=reg_field64)
        self.Mixed_5d = inception_a(reg_field288, pool_features=reg_field64)
        self.Mixed_6a = inception_b(reg_field288)
        self.Mixed_6b = inception_c(reg_field768, channels_7x7=reg_field128)
        self.Mixed_6c = inception_c(reg_field768, channels_7x7=reg_field160)
        self.Mixed_6d = inception_c(reg_field768, channels_7x7=reg_field160)
        self.Mixed_6e = inception_c(reg_field768, channels_7x7=reg_field192)
        if aux_logits:
            self.AuxLogits = inception_aux(reg_field768, num_classes)
        self.Mixed_7a = inception_d(reg_field768)
        self.Mixed_7b = inception_e(reg_field1280)
        self.Mixed_7c = inception_e(reg_field2048)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout()
        self.fc = torch.nn.Linear(2048, num_classes)
        

    

    def _forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux = self.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        
        x=x.tensor
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x, aux):
        # type: (Tensor, Optional[Tensor]) -> InceptionOutputs
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x

    def forward(self, x):
        x = enn.GeometricTensor(x, enn.FieldType(self.gspace, 3*[self.gspace.trivial_repr]))
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape    
        
        
    
class InceptionA(enn.EquivariantModule):

    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        
        self.gspace = gspaces.Rot2dOnR2(N=8)
        
        reg_field32 = FIELD_TYPE["regular"](self.gspace, 32, fixparams=False)
        reg_field48 = FIELD_TYPE["regular"](self.gspace, 48, fixparams=False)
        reg_field64 = FIELD_TYPE["regular"](self.gspace, 64, fixparams=False)
        reg_field96 = FIELD_TYPE["regular"](self.gspace, 96, fixparams=False)
        reg_field80 = FIELD_TYPE["regular"](self.gspace, 80, fixparams=False)
        reg_field192 = FIELD_TYPE["regular"](self.gspace, 192, fixparams=False)
        reg_field256 = FIELD_TYPE["regular"](self.gspace, 256, fixparams=False)
        reg_field288 = FIELD_TYPE["regular"](self.gspace, 288, fixparams=False)
        reg_field768 = FIELD_TYPE["regular"](self.gspace, 768, fixparams=False)
        reg_field160 = FIELD_TYPE["regular"](self.gspace, 160, fixparams=False)
        reg_field128 = FIELD_TYPE["regular"](self.gspace, 128, fixparams=False)
        reg_field160 = FIELD_TYPE["regular"](self.gspace, 160, fixparams=False)
        reg_field192 = FIELD_TYPE["regular"](self.gspace, 192, fixparams=False)
        reg_field1280 = FIELD_TYPE["regular"](self.gspace, 1280, fixparams=False)
        reg_field2048 = FIELD_TYPE["regular"](self.gspace, 2048, fixparams=False)
        
        
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, reg_field64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, reg_field48, kernel_size=1)
        self.branch5x5_2 = conv_block(reg_field48, reg_field64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, reg_field64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(reg_field64, reg_field96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(reg_field96, reg_field96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)
        self.branch_pool_ = enn.PointwiseAvgPool(in_channels, kernel_size=3, stride=1, padding=1)
        
        
    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool_(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return enn.tensor_directsum(outputs)
    
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape
    
    
    

class InceptionB(enn.EquivariantModule):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        self.gspace = gspaces.Rot2dOnR2(N=8)
        reg_field64 = FIELD_TYPE["regular"](self.gspace, 64, fixparams=False)
        reg_field96 = FIELD_TYPE["regular"](self.gspace, 96, fixparams=False)
        reg_field384 = FIELD_TYPE["regular"](self.gspace, 384, fixparams=False)
        
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, reg_field384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, reg_field64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(reg_field64, reg_field96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(reg_field96, reg_field96, kernel_size=3, stride=2)
        
        self.pool = enn.PointwiseMaxPool(in_channels, kernel_size=3, stride=2)
        
    def _forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.pool(x)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return enn.tensor_directsum(outputs)

    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape
    
    

class InceptionC(enn.EquivariantModule):

    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.gspace = gspaces.Rot2dOnR2(N=8)
        reg_field192 = FIELD_TYPE["regular"](self.gspace, 192, fixparams=False)
        
        self.branch1x1 = conv_block(in_channels, reg_field192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_3 = conv_block(c7, reg_field192, kernel_size=7, padding=3)

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=7, padding=3)
        self.branch7x7dbl_5 = conv_block(c7, reg_field192, kernel_size=7, padding=3)

        self.branch_pool = conv_block(in_channels, reg_field192, kernel_size=1)
        self.pool = enn.PointwiseAvgPool(in_channels, kernel_size=3, stride=1, padding=1)
        
        
    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        #branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        #branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        #branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return enn.tensor_directsum(outputs)

    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape
    
    

class InceptionD(enn.EquivariantModule):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.gspace = gspaces.Rot2dOnR2(N=8)    
        reg_field192 = FIELD_TYPE["regular"](self.gspace, 192, fixparams=False)
        reg_field320 = FIELD_TYPE["regular"](self.gspace, 320, fixparams=False)
        
        
        self.branch3x3_1 = conv_block(in_channels, reg_field192, kernel_size=1)
        self.branch3x3_2 = conv_block(reg_field192, reg_field320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, reg_field192, kernel_size=1)
        self.branch7x7x3_3 = conv_block(reg_field192, reg_field192, kernel_size=7, padding=3)
        self.branch7x7x3_4 = conv_block(reg_field192, reg_field192, kernel_size=3, stride=2)
        
        self.pool = enn.PointwiseMaxPool(in_channels, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        #branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.pool(x)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return enn.tensor_directsum(outputs)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape
    
    
    
    
class InceptionE(enn.EquivariantModule):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.gspace = gspaces.Rot2dOnR2(N=8)   
        reg_field384 = FIELD_TYPE["regular"](self.gspace, 384, fixparams=False)
        reg_field320 = FIELD_TYPE["regular"](self.gspace, 320, fixparams=False)
        reg_field192 = FIELD_TYPE["regular"](self.gspace, 192, fixparams=False)
        reg_field448 = FIELD_TYPE["regular"](self.gspace, 448, fixparams=False)
        
        
        self.branch1x1 = conv_block(in_channels, reg_field320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, reg_field384, kernel_size=1)
        self.branch3x3_2a = conv_block(reg_field384, reg_field384, kernel_size=3, padding=1)

        self.branch3x3dbl_1 = conv_block(in_channels, reg_field448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(reg_field448, reg_field384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(reg_field384, reg_field384, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, reg_field192, kernel_size=1)
        
        self.pool = enn.PointwiseAvgPool(in_channels, kernel_size=3, stride=1, padding=1)
        
    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2a(branch3x3),
        ]
        branch3x3 = enn.tensor_directsum(branch3x3)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3a(branch3x3dbl),
        ]
        branch3x3dbl = enn.tensor_directsum(branch3x3dbl)

        branch_pool = self.pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return enn.tensor_directsum(outputs)
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape

    
    
    
    
class InceptionAux(torch.nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.gspace = gspaces.Rot2dOnR2(N=8)    
        reg_field128 = FIELD_TYPE["regular"](self.gspace, 128, fixparams=False)
        reg_field768 = FIELD_TYPE["regular"](self.gspace, 768, fixparams=False)
        
        
        self.gspace = gspaces.Rot2dOnR2(N=8)
        self.conv0 = conv_block(in_channels, reg_field128, kernel_size=1)
        self.conv1 = conv_block(reg_field128, reg_field768, kernel_size=5)
        #self.conv1.stddev = 0.01
        self.fc = torch.nn.Linear(768, num_classes)
        self.fc.stddev = 0.001
        
        self.pool = enn.PointwiseAvgPool(in_channels, kernel_size=5, stride=3)
        
    def forward(self, x):
        # N x 768 x 17 x 17
        x = self.pool(x)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        
        x = x.tensor
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x

    
    
    
class ResBottleBlock(enn.EquivariantModule):
    def __init__(self, in_planes, bottleneck_width=4, stride=1, expansion=1):
        super(ResBottleBlock, self).__init__()
        
        self.gspace = gspaces.Rot2dOnR2(N=4)
        
        self.conv0=enn.R2Conv(FIELD_TYPE["regular"](self.gspace, in_planes, fixparams=False), FIELD_TYPE["regular"](self.gspace, bottleneck_width, fixparams=False),kernel_size=1,stride=1,bias=False)
        self.bn0 = enn.InnerBatchNorm(FIELD_TYPE["regular"](self.gspace, bottleneck_width, fixparams=False))
        self.conv1= enn.R2Conv(FIELD_TYPE["regular"](self.gspace, FIELD_TYPE["regular"](self.gspace, bottleneck_width, fixparams=False), fixparams=False),bottleneck_width,3,stride=stride,padding=1,bias=False)
        self.bn1=enn.InnerBatchNorm(FIELD_TYPE["regular"](self.gspace, bottleneck_width, fixparams=False))
        self.conv2=enn.R2Conv(FIELD_TYPE["regular"](self.gspace, bottleneck_width, fixparams=False), FIELD_TYPE["regular"](self.gspace, in_planes*expansion, fixparams=False),1,bias=False)
        self.bn2=enn.InnerBatchNorm(FIELD_TYPE["regular"](self.gspace, in_planes*expansion, fixparams=False))
        
        self.shortcut=nn.Sequential()
        
        if stride!=1 or expansion!=1:
            self.shortcut=nn.Sequential(
                nn.Conv2d(FIELD_TYPE["regular"](self.gspace, in_planes, fixparams=False),FIELD_TYPE["regular"](self.gspace, in_planes*expansion, fixparams=False),1,stride=stride,bias=False)
            )
        self.relu1 = enn.ReLU(FIELD_TYPE["regular"](self.gspace, bottleneck_width, fixparams=False), inplace = True)
        self.relu2 = enn.ReLU(FIELD_TYPE["regular"](self.gspace, in_planes*expansion, fixparams=False), inplace=True)
        self.relu3 = enn.ReLU(FIELD_TYPE["regular"](self.gspace, in_planes*expansion, fixparams=False), inplace=True)
        
        
    def forward(self, x):
        out = self.relu1(self.bn0(self.conv0(x)))
        out = self.relu2(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape
        
        
        

class BasicBlock_A(enn.EquivariantModule):
    def __init__(self, in_planes, num_paths=32, bottleneck_width=4, expansion=1, stride=1):
        super(BasicBlock_A,self).__init__()
        self.gspace = gspaces.Rot2dOnR2(N=4)
        self.num_paths = num_paths
        for i in range(num_paths):
            setattr(self,'path'+str(i),self._make_path(in_planes,bottleneck_width,stride,expansion))

        # self.paths=self._make_path(in_planes,bottleneck_width,stride,expansion)
        self.conv0=enn.R2Conv(FIELD_TYPE["regular"](self.gspace, in_planes*expansion, fixparams=False), FIELD_TYPE["regular"](self.gspace, in_planes*expansion, fixparams=False),1,stride=1,bias=False)
        self.bn0 = enn.InnerBatchNorm(FIELD_TYPE["regular"](self.gspace, in_planes*expansion, fixparams=False))
        
        self.shortcut = nn.Sequential()
        if stride != 1 or expansion != 1:
            self.shortcut = nn.Sequential(
                enn.R2Conv(FIELD_TYPE["regular"](self.gspace, in_planes, fixparams=False), FIELD_TYPE["regular"](self.gspace, in_planes*expansion, fixparams=False), 1, stride=stride, bias=False)
            )
        self.relu = enn.ReLU(FIELD_TYPE["regular"](self.gspace, in_planes*expansion, fixparams=False),inplace=True)
        
        
    def forward(self, x):
        out = self.path0(x)
        for i in range(1,self.num_paths):
            if hasattr(self,'path'+str(i)):
                out+getattr(self,'path'+str(i))(x)
            # out+=self.paths(x)
            # getattr
        # out = torch.sum(out, dim=1)
        out = self.bn0(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

    def _make_path(self, in_planes, bottleneck_width, stride, expansion):
        layers = []
        layers.append(ResBottleBlock(
            in_planes, bottleneck_width, stride, expansion))
        return nn.Sequential(*layers)


    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape
        
        

class BasicBlock_C(enn.EquivariantModule):
    """
    increasing cardinality is a more effective way of 
    gaining accuracy than going deeper or wider
    """

    def __init__(self, in_planes, bottleneck_width=4, cardinality=32, stride=1, expansion=2):
        super(BasicBlock_C, self).__init__()
        self.gspace = gspaces.Rot2dOnR2(N=4)
        inner_width = cardinality * bottleneck_width
        self.expansion = expansion
        self.basic = nn.Sequential(OrderedDict(
            [
                ('conv1_0', enn.R2Conv(FIELD_TYPE["regular"](self.gspace, in_planes, fixparams=False), FIELD_TYPE["regular"](self.gspace, inner_width, fixparams=False), 1, stride=1, bias=False)),
                ('bn1', enn.InnerBatchNorm(FIELD_TYPE["regular"](self.gspace, inner_width, fixparams=False))),
                ('act0', enn.ReLU(FIELD_TYPE["regular"](self.gspace, inner_width, fixparams=False))),
                ('conv3_0', enn.R2Conv(FIELD_TYPE["regular"](self.gspace, inner_width, fixparams=False), FIELD_TYPE["regular"](self.gspace, inner_width, fixparams=False), 3, stride=stride, padding=1, groups=cardinality, bias=False)),
                ('bn2', enn.InnerBatchNorm(FIELD_TYPE["regular"](self.gspace, inner_width, fixparams=False))),
                ('act1', enn.ReLU(FIELD_TYPE["regular"](self.gspace, inner_width, fixparams=False))),
                ('conv1_1', enn.R2Conv(FIELD_TYPE["regular"](self.gspace, inner_width, fixparams=False), FIELD_TYPE["regular"](self.gspace, inner_width*self.expansion, fixparams=False), 1, stride=1, bias=False)),
                ('bn3', enn.InnerBatchNorm(FIELD_TYPE["regular"](self.gspace, inner_width*self.expansion, fixparams=False)))
            ]
        ))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != inner_width * self.expansion:
            self.shortcut = nn.Sequential(
                enn.R2Conv(FIELD_TYPE["regular"](self.gspace, in_planes, fixparams=False), FIELD_TYPE["regular"](self.gspace, inner_width*self.expansion, fixparams=False), 1, stride=stride, bias=False)
            )
        self.bn0 = enn.InnerBatchNorm(FIELD_TYPE["regular"](self.gspace, inner_width*self.expansion, fixparams=False))
        self.relu_out = enn.ReLU(FIELD_TYPE["regular"](self.gspace, inner_width*self.expansion, fixparams=False))
        
        
    def forward(self, x):
        out = self.basic(x)
        out += self.shortcut(x)
        out = self.relu_out(self.bn0(out))
        return out
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape
    
    
    
    
class ResNeXt(torch.nn.Module):
    """implementation of equivariant ResneXt Network (https://arxiv.org/abs/1611.05431), adapted from: https://github.com/Hsuxu/ResNeXt"""
    def __init__(self, num_blocks, cardinality, bottleneck_width, expansion=2, num_classes=1):
        super(ResNeXt, self).__init__()
        self.gspace = gspaces.Rot2dOnR2(N=4)
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.expansion = expansion
        
        self.conv0 = enn.R2Conv(FIELD_TYPE["trivial"](self.gspace, 3, fixparams=False), FIELD_TYPE["regular"](self.gspace, self.in_planes, fixparams=False), kernel_size=7, stride=2, padding=3)
        self.bn0 = enn.InnerBatchNorm(FIELD_TYPE["regular"](self.gspace, self.in_planes, fixparams=False))
        self.pool0 = enn.PointwiseMaxPool(FIELD_TYPE["regular"](self.gspace, self.in_planes, fixparams=False), kernel_size=3, stride=2, padding=1)
        self.layer1=self._make_layer(num_blocks[0],1)
        self.layer2=self._make_layer(num_blocks[1],2)
        self.layer3=self._make_layer(num_blocks[2],2)
        self.layer4=self._make_layer(num_blocks[3],2)
        self.linear = nn.Linear(self.cardinality * self.bottleneck_width, num_classes)
        
        self.relu = enn.ReLU(FIELD_TYPE["regular"](self.gspace, 64, fixparams=False))
        
        
    def forward(self, x):
        x = enn.GeometricTensor(x, enn.FieldType(self.gspace, 3*[self.gspace.trivial_repr]))
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)
        out = self.pool0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.tensor
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_C(self.in_planes, self.bottleneck_width, self.cardinality, stride, self.expansion))
            self.in_planes = self.expansion * self.bottleneck_width * self.cardinality
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)