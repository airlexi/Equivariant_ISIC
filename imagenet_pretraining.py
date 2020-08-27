from typing import Tuple
import torch
import torch.nn.functional as F
import copy
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import e2cnn.nn as enn
from e2cnn.nn import init
from e2cnn import gspaces
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tqdm.autonotebook import tqdm


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



class BasicBlock(enn.EquivariantModule):
    def __init__(self, in_type, out_type):
        super(BasicBlock, self).__init__()
        self.in_type = in_type
        self.out_type = out_type
        
        self.bn1 = enn.InnerBatchNorm(self.in_type)
        self.relu1 = enn.ReLU(self.in_type, inplace=True)
        self.conv1 = conv3x3(self.in_type, self.out_type)
        
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        return enn.tensor_directsum([x,out])

    

class BottleneckBlock(enn.EquivariantModule):
    def __init__(self, in_type, inner_type, out_type, stride=1):
        super(BottleneckBlock, self).__init__()
        
        self.bn1 = enn.InnerBatchNorm(in_type)
        self.relu1 = enn.ReLU(in_type,inplace=True)
        self.conv1 = conv1x1(in_type, inner_type)
        self.bn2 = enn.InnerBatchNorm(inner_type)
        self.relu2 = enn.ReLU(inner_type,inplace=True)
        self.conv2 = conv3x3(inner_type, out_type)
        
    def forward(self, x):
        out = x
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = enn.tensor_directsum([x,out])
        return out
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape
    

class TransitionBlock(enn.EquivariantModule):
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
        out = x
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.avgpool(out)
        return out
        
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape
        
        
        
        
class DenseBlock(enn.EquivariantModule):
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

        
        
        
class DenseNet161(torch.nn.Module):
    def __init__(self, growth_rate, list_layer, nclasses):
        super(DenseNet161, self).__init__()
        
        self.gspace = gspaces.Rot2dOnR2(N=8)
        
        in_type = 2*growth_rate
        
        self.conv1 = conv7x7(FIELD_TYPE["trivial"](self.gspace, 3, fixparams=False), 
                             FIELD_TYPE["regular"](self.gspace, in_type, fixparams=False))
        
        self.pool1 = enn.PointwiseMaxPool(FIELD_TYPE["regular"](self.gspace, in_type, fixparams=False),
                                          kernel_size=2, stride=2)
        
        
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
        self.classifier = torch.nn.Linear(in_type, nclasses)
        
        
    def forward(self, x):
        out = enn.GeometricTensor(x, enn.FieldType(self.gspace, 3*[self.gspace.trivial_repr]))
        out = self.conv1(out)
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
        out = self.classifier(out)
        
        return out

    
    
class ResBlock(enn.EquivariantModule):
    
    def __init__(self, in_type,inner_type, out_type, stride=1):
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
        
        
class ResNet50(torch.nn.Module):
    def __init__(self, nclasses=1):
        super(ResNet50, self).__init__()
        self.gspace = gspaces.Rot2dOnR2(N=8)
        
        reg_field64 = FIELD_TYPE["regular"](self.gspace, 64, fixparams=False)
        reg_field256 = FIELD_TYPE["regular"](self.gspace, 256, fixparams=False)
        reg_field128 = FIELD_TYPE["regular"](self.gspace, 128, fixparams=False)
        reg_field512 = FIELD_TYPE["regular"](self.gspace, 512, fixparams=False)
        reg_field1024 = FIELD_TYPE["regular"](self.gspace, 1024, fixparams=False)
        reg_field2048 = FIELD_TYPE["regular"](self.gspace, 2048, fixparams=False)
        
        self.conv1 = enn.R2Conv(FIELD_TYPE["trivial"](self.gspace, 3, fixparams=False),
                                reg_field64, kernel_size=7, stride=2, padding=3)
        self.bn1 = enn.InnerBatchNorm(reg_field64)
        self.relu1 = enn.ELU(reg_field64)
        self.maxpool1 = enn.PointwiseMaxPoolAntialiased(reg_field64, kernel_size=2)
        
        layer1 = []
        layer1.append(ResBlock(stride=2, in_type = reg_field64, inner_type = reg_field64, out_type = reg_field256))
        layer1.append(ResBlock(stride=1, in_type = reg_field256, inner_type = reg_field64, out_type = reg_field256))
        layer1.append(ResBlock(stride=1, in_type = reg_field256, inner_type = reg_field64, out_type = reg_field256))
        self.layer1 = torch.nn.Sequential(*layer1)
        
        layer2 = []
        layer2.append(ResBlock(stride=2, in_type = reg_field256, inner_type = reg_field128, out_type = reg_field512))
        layer2.append(ResBlock(stride=1, in_type = reg_field512, inner_type = reg_field128, out_type = reg_field512))
        layer2.append(ResBlock(stride=1, in_type = reg_field512, inner_type = reg_field128, out_type = reg_field512))
        layer2.append(ResBlock(stride=1, in_type = reg_field512, inner_type = reg_field128, out_type = reg_field512))
        self.layer2 = torch.nn.Sequential(*layer2)
        
        layer3 = []
        layer3.append(ResBlock(stride=2, in_type = reg_field512, inner_type = reg_field256, out_type = reg_field1024))
        layer3.append(ResBlock(stride=1, in_type = reg_field1024, inner_type = reg_field256, out_type = reg_field1024))
        layer3.append(ResBlock(stride=1, in_type = reg_field1024, inner_type = reg_field256, out_type = reg_field1024))
        layer3.append(ResBlock(stride=1, in_type = reg_field1024, inner_type = reg_field256, out_type = reg_field1024))
        layer3.append(ResBlock(stride=1, in_type = reg_field1024, inner_type = reg_field256, out_type = reg_field1024))
        layer3.append(ResBlock(stride=1, in_type = reg_field1024, inner_type = reg_field256, out_type = reg_field1024))
        self.layer3 = torch.nn.Sequential(*layer3)
        
        layer4 = []
        layer4.append(ResBlock(stride=2, in_type = reg_field1024, inner_type = reg_field512, out_type = reg_field2048))
        layer4.append(ResBlock(stride=1, in_type = reg_field2048, inner_type = reg_field512, out_type = reg_field2048))
        layer4.append(ResBlock(stride=1, in_type = reg_field2048, inner_type = reg_field512, out_type = reg_field2048))
        self.layer4 = torch.nn.Sequential(*layer4)
        
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(2048, nclasses)
        
        
    def forward(self, x):
        x = enn.GeometricTensor(x, enn.FieldType(self.gspace, 3*[self.gspace.trivial_repr]))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.tensor
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
        

def run_epoch(model, optimizer, dataloader, train, len_dataloader):
  
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_top5 = 0.0
    length_dataset = 0
    for xb, yb in tqdm((dataloader), total=int(len_dataloader)):
    #for xb, yb in (dataloader):
        xb, yb = xb.to(device), yb.to(device)
        
        length_dataset += xb.shape[0]
        
        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss = F.cross_entropy(pred, yb)
            top1 = torch.argmax(pred, dim=1)
            top5 = torch.topk(pred, 5)
            top5sum = torch.sum((top5[1]==yb.unsqueeze(1)))
            ncorrect = torch.sum(top1 == yb)

            # backward + optimize only if in training phase
            if train:
                loss.backward()
                optimizer.step()

        # statistics
        epoch_loss += loss.item()
        epoch_acc += ncorrect.item()
        epoch_top5 += top5sum.item()
        
    
    epoch_acc /= length_dataset
    epoch_top5 /= length_dataset
    return epoch_loss, epoch_acc, epoch_top5



def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    #initializing the model you want to train
    model = ResNet50(nclasses=1000).to(device)
    print("model initialized")
    
    #building the dataset
    train_dir =  "../../../storage/group/dataset_mirrors/old_common_datasets/imagenet/Data/train"
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ]))
    print("train_dataset made")
    
    #training on random subset and with random validation set
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [250000,len(train_dataset)- 250000])
    val_dataset, _ = torch.utils.data.random_split(val_dataset, [50000,len(val_dataset)-50000])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                                   shuffle=True, num_workers=4,
                                                   pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64,
                                                   shuffle=False, num_workers=4,
                                                   pin_memory=True)

    #pretraining is done with stochastic gradient decent with learning rate 0.1 and momentum 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    logs = []
    best_loss = None
    max_epochs = 90

    
    logs = []
    best_loss = None
    max_epochs = 90
    for epoch in range(max_epochs):
        
        print(f"Epoch {epoch + 1}/{max_epochs}")
        print("")
        print("training...")
        
        epoch_loss, train_acc, train_top5 = run_epoch(model, optimizer, train_loader, True, len(train_loader))
       
        print(f"Train loss: {epoch_loss}")
        print(f"ACC on training set: {train_acc}")
        print(f"Top5 ACC on training set: {train_top5}")
        print("")
        
        lr_scheduler.step()
        logs.append(("EPOCH "+str(epoch)+
                    ": train loss: "+str(epoch_loss)+
                    ", train acc: "+str(train_acc)+", train top5 acc: "+str(train_top5)))
        
        
        print("validating...")
        val_loss, val_acc, val_top5 = run_epoch(model, None, val_loader, False, len(val_loader))
        
        print(f"Test loss: {val_loss}")
        print(f"Val Acc: {val_acc}")
        print(f"Top5 ACC on val set: {val_top5}")
        print("")
        
        logs.append(("EPOCH "+str(epoch)+
                    ": val loss: "+str(val_loss)+
                    ", val acc: "+str(val_acc)+", val top5 acc: "+str(val_top5)))
        
        with open("ResNet50_results.json", "w") as file:
                    json.dump(logs, file)
                
        if best_loss == None:
            best_loss = val_loss
            
        if val_loss <= best_loss:
            
            best_epoch = epoch
            best_loss = val_loss
            best_acc = val_acc
            best_top5 = val_top5
            
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "../../../storage/group/pretrained/rotation-equivariant/pretrained_ResNet50")
            
            
    logs.append(("EPOCH "+str(best_epoch)+
                    ": best loss: "+str(best_loss)+
                    ", best acc: "+str(best_acc)+", best top5 acc: "+str(best_top5)))
    with open("ResNet50_results.json", "w") as file:
                    json.dump(logs, file)
    print("finished")
    

if __name__ == '__main__':
    main()

    