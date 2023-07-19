from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
__all__ = ['resnet18_rpp']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,

    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, FCN=False, T=1, dim = 256):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.FCN=FCN
        self.T = T
        self.reduce_dim = dim
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

#==========================add dilation=============================#
        if self.FCN:
            for mo in self.base.layer4[0].modules():
                if isinstance(mo, nn.Conv2d):
                    mo.stride = (1,1)
#================append conv for FCN==============================#
            self.num_features = num_features
            self.num_classes = num_classes
            self.dropout = dropout
            self.out_planes = self.base.fc.in_features
            self.local_conv = nn.Conv2d(self.out_planes, self.num_features, kernel_size=1,padding=0,bias=False)
            init.kaiming_normal(self.local_conv.weight, mode= 'fan_out')
#            init.constant(self.local_conv.bias,0)
            self.feat_bn2d = nn.BatchNorm2d(self.num_features) #may not be used, not working on caffe
            init.constant(self.feat_bn2d.weight,1) #initialize BN, may not be used
            init.constant(self.feat_bn2d.bias,0) # iniitialize BN, may not be used

##---------------------------stripe1----------------------------------------------#
            self.instance0 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance0.weight, std=0.001)
            init.constant(self.instance0.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance1 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance1.weight, std=0.001)
            init.constant(self.instance1.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance2 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance2.weight, std=0.001)
            init.constant(self.instance2.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance3 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance3.weight, std=0.001)
            init.constant(self.instance3.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance4 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance4.weight, std=0.001)
            init.constant(self.instance4.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance5 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance5.weight, std=0.001)
            init.constant(self.instance5.bias, 0)
##---------------------------stripe1----------------------------------------------#
            self.drop = nn.Dropout(self.dropout)
            self.local_mask = nn.Conv2d(self.reduce_dim, 3 , kernel_size=1,padding=0,bias=True)
            init.kaiming_normal(self.local_mask.weight, mode= 'fan_out')
            init.constant(self.local_mask.bias,0)

#===================================================================#

        elif not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features, bias=False)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
#                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
#                self.classifier = nn.Linear(self.num_features, self.num_classes)
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x
#=======================FCN===============================#
        if self.FCN:
            T = self.T         
            y = self.drop(x).unsqueeze(1)
            stride = int(self.out_planes/self.reduce_dim)
            y = F.avg_pool3d(y,kernel_size=(stride,1,1),stride=(stride,1,1)).squeeze(1)
            center = F.avg_pool2d(y,(y.size(2),y.size(3)))
            y = y-center.expand_as(y)
            local_mask = self.local_mask(y)#B,(6,24,8)
            local_mask = F.softmax(T*local_mask)   #using softmax mode

            lw = local_mask.chunk(3,1)
            x = x*3
            f0 = x*(lw[0].expand_as(x))#B,1,24,8 -> B, 256, 24, 8
            f1 = x*(lw[1].expand_as(x))
            f2 = x*(lw[2].expand_as(x))
            f0 = F.avg_pool2d(f0,kernel_size=(f0.size(2),f0.size(3)))  
            f1 = F.avg_pool2d(f1,kernel_size=(f1.size(2),f1.size(3)))  
            f2 = F.avg_pool2d(f2,kernel_size=(f2.size(2),f2.size(3)))  
            x = torch.cat((f0,f1,f2,),2)
            feat = torch.cat((f0,f1,f2,),2)
            
            out0 = feat/feat.norm(2,1).unsqueeze(1).expand_as(feat)
            
            x = self.drop(x)
            x = self.local_conv(x)

            out1 = x.view(x.size(0),-1)
            out1 = x/x.norm(2,1).unsqueeze(1).expand_as(x)
            
            x = self.feat_bn2d(x)
            out1 = x/x.norm(2,1).unsqueeze(1).expand_as(x)
            x = F.relu(x) # relu for local_conv feature
            x = x.chunk(3,2)
            x0 = x[0].contiguous().view(x[0].size(0),-1)
            x1 = x[1].contiguous().view(x[1].size(0),-1)
            x2 = x[2].contiguous().view(x[2].size(0),-1)
            c0 = self.instance0(x0)
            c1 = self.instance1(x1)
            c2 = self.instance2(x2)
            return out0, (c0, c1, c2),local_mask

#==========================================================#


        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        out1 = x
        out1 = x / x.norm(2,1).unsqueeze(1).expand_as(x)
        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
            out2 = x/ x.norm(2,1).unsqueeze(1).expand_as(x)
        if self.norm:
            x = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)


        return out2, x
	

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


class EfficientNet(nn.Module):
    __factory = {
        "b1": torchvision.models.efficientnet_b1,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, FCN=False, T=1, dim = 256):
        super(EfficientNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.FCN=FCN
        self.T = T
        self.reduce_dim = dim
        if depth not in EfficientNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = EfficientNet.__factory[depth](pretrained=pretrained)

#==========================add dilation=============================#
        if self.FCN:
            for mo in self.base.features[4][0].block[1].modules():
                if isinstance(mo, nn.Conv2d):
                    mo.stride = (1,1)
#================append conv for FCN==============================#
            self.num_features = num_features
            self.num_classes = num_classes
            self.dropout = dropout
            self.out_planes = self.base.classifier[1].in_features
            self.local_conv = nn.Conv2d(self.out_planes, self.num_features, kernel_size=1,padding=0,bias=False)
            init.kaiming_normal(self.local_conv.weight, mode= 'fan_out')
#            init.constant(self.local_conv.bias,0)
            self.feat_bn2d = nn.BatchNorm2d(self.num_features) #may not be used, not working on caffe
            init.constant(self.feat_bn2d.weight,1) #initialize BN, may not be used
            init.constant(self.feat_bn2d.bias,0) # iniitialize BN, may not be used

##---------------------------stripe1----------------------------------------------#
            self.instance0 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance0.weight, std=0.001)
            init.constant(self.instance0.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance1 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance1.weight, std=0.001)
            init.constant(self.instance1.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance2 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance2.weight, std=0.001)
            init.constant(self.instance2.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance3 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance3.weight, std=0.001)
            init.constant(self.instance3.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance4 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance4.weight, std=0.001)
            init.constant(self.instance4.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance5 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance5.weight, std=0.001)
            init.constant(self.instance5.bias, 0)
##---------------------------stripe1----------------------------------------------#
            self.drop = nn.Dropout(self.dropout)
            self.local_mask = nn.Conv2d(self.reduce_dim, 3 , kernel_size=1,padding=0,bias=True)
            init.kaiming_normal(self.local_mask.weight, mode= 'fan_out')
            init.constant(self.local_mask.bias,0)

#===================================================================#

        elif not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features, bias=False)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
#                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
#                self.classifier = nn.Linear(self.num_features, self.num_classes)
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x
#=======================FCN===============================#
        if self.FCN:
            T = self.T         
            y = self.drop(x).unsqueeze(1)
            stride = int(self.out_planes/self.reduce_dim)
            y = F.avg_pool3d(y,kernel_size=(stride,1,1),stride=(stride,1,1)).squeeze(1)
            center = F.avg_pool2d(y,(y.size(2),y.size(3)))
            y = y-center.expand_as(y)
            local_mask = self.local_mask(y)#B,(6,24,8)
            local_mask = F.softmax(T*local_mask)   #using softmax mode

            lw = local_mask.chunk(3,1)
            x = x*3
            f0 = x*(lw[0].expand_as(x))#B,1,24,8 -> B, 256, 24, 8
            f1 = x*(lw[1].expand_as(x))
            f2 = x*(lw[2].expand_as(x))
            f0 = F.avg_pool2d(f0,kernel_size=(f0.size(2),f0.size(3)))  
            f1 = F.avg_pool2d(f1,kernel_size=(f1.size(2),f1.size(3)))  
            f2 = F.avg_pool2d(f2,kernel_size=(f2.size(2),f2.size(3)))  
            x = torch.cat((f0,f1,f2,),2)
            feat = torch.cat((f0,f1,f2,),2)
            
            out0 = feat/feat.norm(2,1).unsqueeze(1).expand_as(feat)
            
            x = self.drop(x)
            x = self.local_conv(x)

            out1 = x.view(x.size(0),-1)
            out1 = x/x.norm(2,1).unsqueeze(1).expand_as(x)
            
            x = self.feat_bn2d(x)
            out1 = x/x.norm(2,1).unsqueeze(1).expand_as(x)
            x = F.relu(x) # relu for local_conv feature
            x = x.chunk(3,2)
            x0 = x[0].contiguous().view(x[0].size(0),-1)
            x1 = x[1].contiguous().view(x[1].size(0),-1)
            x2 = x[2].contiguous().view(x[2].size(0),-1)
            c0 = self.instance0(x0)
            c1 = self.instance1(x1)
            c2 = self.instance2(x2)
            return out0, (c0, c1, c2),local_mask

#==========================================================#


        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        out1 = x
        out1 = x / x.norm(2,1).unsqueeze(1).expand_as(x)
        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
            out2 = x/ x.norm(2,1).unsqueeze(1).expand_as(x)
        if self.norm:
            x = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)


        return out2, x
	

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)




def resnet18_rpp(**kwargs):
    return ResNet(18, **kwargs)
def efficientnet_b1_rpp(**kwargs):
    return EfficientNet('b1',**kwargs)
