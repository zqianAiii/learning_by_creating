import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

class Generator(nn.Module):
    
    def __init__(self, z_dim, label_dim, out_dim, num_channels):
        super(Generator, self).__init__()
        
        self.z_input = nn.Sequential()
        self.y_input = nn.Sequential()
        self.hidden = nn.Sequential()
        self.output = nn.Sequential()
        
        for i in range(len(num_channels)):
              
            if i == 0:
                #DeConv
                z_deconv = nn.ConvTranspose2d(z_dim, num_channels[i]//2, kernel_size=4, stride=1, padding=0)
                nn.init.normal_(z_deconv.weight, mean=0.0, std=0.02)
                nn.init.constant_(z_deconv.bias, 0.0)
                self.z_input.add_module('deconv', z_deconv)
                
                y_deconv = nn.ConvTranspose2d(label_dim, num_channels[i]//2, kernel_size=4, stride=1, padding=0)
                nn.init.normal_(y_deconv.weight, mean=0.0, std=0.02)
                nn.init.constant_(y_deconv.bias, 0.0)
                self.y_input.add_module('deconv', y_deconv)
                
                #Batchnorm
                self.z_input.add_module('batchnorm', nn.BatchNorm2d(num_channels[i]//2))
                self.y_input.add_module('batchnorm', nn.BatchNorm2d(num_channels[i]//2))
                
                #Activation
                self.z_input.add_module('ReLU', nn.ReLU())
                self.y_input.add_module('ReLU', nn.ReLU())
                
            else:
                #DeConv
                deconv = nn.ConvTranspose2d(num_channels[i-1], num_channels[i], kernel_size=4, stride=2, padding=1)
                nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
                nn.init.constant_(deconv.bias, 0.0)
                self.hidden.add_module('deconv'+str(i), deconv)
                
                #Batchnorm
                self.hidden.add_module('batchnorm'+str(i), nn.BatchNorm2d(num_channels[i]))
                
                #Activation
                self.hidden.add_module('ReLU'+str(i), nn.ReLU())

            #DeConv
            deconv = nn.ConvTranspose2d(num_channels[-1], out_dim, kernel_size=4, stride=2, padding=1)
            nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
            nn.init.constant_(deconv.bias, 0.0)
            self.output.add_module('deconv', deconv)
                
            #Activation
            self.output.add_module('Tanh', nn.Tanh())
            
    def forward(self, z, y):
        hz = self.z_input(z)
        hy = self.y_input(y)
        h = self.hidden(torch.cat([hz, hy], 1))
        return self.output(h)
    
    def loss(self, x, target):
        return F.binary_cross_entropy_with_logits(x, target)
    
   
    
    
    
class resnet_transfer(nn.Module):
    
    def __init__(self, freeze_upto = 0):
        super(resnet_transfer, self).__init__()
        self.clf = tv.models.resnet18(pretrained=True)
        self.clf.fc = nn.Linear(in_features=512, out_features=10, bias=True)
        self.freeze(freeze_upto)
        
    def freeze(self, layer):
        if (layer >= 0) and (layer <= 4):
            if layer == 0:
                for param in self.clf.parameters():
                    param.requires_grad = True
            else:
                for idx, l in enumerate(self.clf.children()):
                    if idx < 2:
                        for param in l.parameters():
                            param.requires_grad = False
                    elif (idx >= 4) and (idx <= 7):
                        if layer > 0:
                            for param in l.parameters():
                                param.requires_grad = False
                            layer -= 1
                        else:
                            for param in l.parameters():
                                param.requires_grad = True
        
    def forward(self, x):
        return self.clf(x)
    
    def loss(self, x, target):
        return F.cross_entropy(x, target)
            