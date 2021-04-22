import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

num_classes = 10

#classifier
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

    
#generator operations
class ConvMeanPool(nn.Module):
    
    def __init__(self, input_dim, output_dim, filter_size, he_init=True, biases=True):
        super(ConvMeanPool, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, filter_size, padding = filter_size//2, bias = biases)
        if he_init:
            nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
              
    def forward(self, x):
        output = self.conv(x)
        return (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
    
    
class MeanPoolConv(nn.Module):
    
    def __init__(self, input_dim, output_dim, filter_size, he_init=True, biases=True):
        super(MeanPoolConv, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, filter_size, padding = filter_size//2, bias = biases)
        if he_init:
            nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
              
    def forward(self, x):
        output = (x[:,:,::2,::2] + x[:,:,1::2,::2] + x[:,:,::2,1::2] + x[:,:,1::2,1::2]) / 4
        return self.conv(output)
    
    
class UpsampleConv(nn.Module):
    
    def __init__(self, input_dim, output_dim, filter_size, he_init=True, biases=True):
        super(UpsampleConv, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, filter_size, padding = filter_size//2, bias = biases)
        if he_init:
            nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
              
    def forward(self, x):
        h = torch.cat([x, x, x, x], 1)
        output = F.pixel_shuffle(h, 2)
        return self.conv(output)
    
    
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False, track_running_stats = False)
        self.offset_m = nn.Embedding(num_classes, num_features)
        self.scale_m = nn.Embedding(num_classes, num_features)
        nn.init.zeros_(self.offset_m.weight)
        nn.init.ones_(self.scale_m.weight)

    def forward(self, x, label):
        h = self.bn(x)
        bias = self.offset_m(label)
        weight = self.scale_m(label)
        return h * weight.view(-1, self.num_features, 1, 1) + bias.view(-1, self.num_features, 1, 1)
    
    
class ConditionalLayerNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ConditionalLayerNorm2d, self).__init__()
        self.num_features = num_features
        self.ln = nn.GroupNorm(1, num_features, affine=False)
        self.offset_m = nn.Embedding(num_classes, num_features)
        self.scale_m = nn.Embedding(num_classes, num_features)  
        nn.init.zeros_(self.offset_m.weight)
        nn.init.ones_(self.scale_m.weight)
        
    def forward(self, x, label):
        h = self.ln(x)
        bias = self.offset_m(label)
        weight = self.scale_m(label)
        return h * weight.view(-1, self.num_features, 1, 1) + bias.view(-1, self.num_features, 1, 1)
    
    
class ResidualBlock(nn.Module):
    
    def __init__(self, input_dim, output_dim, filter_size, resample=None, normalize = 'batch'):
        super(ResidualBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        
        if resample == 'down':
            self.conv1 = nn.Conv2d(input_dim, input_dim, filter_size, padding = filter_size//2)
            self.conv2 = ConvMeanPool(input_dim, output_dim, filter_size)
            self.conv_shortcut = ConvMeanPool(input_dim, output_dim, 1, he_init = False)
            if normalize == 'batch':
                self.n2 = ConditionalBatchNorm2d(input_dim, num_classes)
            else:
                self.n2 = ConditionalLayerNorm2d(input_dim, num_classes)
        elif resample == 'up':
            self.conv1 = UpsampleConv(input_dim, output_dim, filter_size)
            self.conv_shortcut = UpsampleConv(input_dim, output_dim, 1, he_init = False)
            self.conv2 = nn.Conv2d(output_dim, output_dim, filter_size, padding = filter_size//2)
            if normalize == 'batch':
                self.n2 = ConditionalBatchNorm2d(output_dim, num_classes)
            else:
                self.n2 = ConditionalLayerNorm2d(output_dim, num_classes)
        else:
            self.conv_shortcut = nn.Conv2d(input_dim, output_dim, 1, padding = filter_size//2)
            self.conv1 =  nn.Conv2d(input_dim, output_dim, filter_size, padding = filter_size//2)
            self.conv2 = nn.Conv2d(output_dim, output_dim, filter_size, padding = filter_size//2)
            if normalize == 'batch':
                self.n2 = ConditionalBatchNorm2d(output_dim, num_classes)
            else:
                self.n2 = ConditionalLayerNorm2d(output_dim, num_classes)
                
        if normalize == 'batch':
            self.n1 = ConditionalBatchNorm2d(input_dim, num_classes)
        else:
            self.n1 = ConditionalLayerNorm2d(input_dim, num_classes)
        
    def forward(self, x, label):
        if self.output_dim == self.input_dim and self.resample == None:
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)
        h = F.relu(self.n1(x, label))
        h = self.conv1(h)
        h = F.relu(self.n2(h, label))
        h = self.conv2(h)
        return h + shortcut


    
class OptimizedResBlockDisc1(nn.Module):
    
    def __init__(self, input_dim, output_dim, filter_size):
        super(OptimizedResBlockDisc1, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, filter_size, padding = 1)
        self.conv2 = ConvMeanPool(output_dim, output_dim, filter_size)
        self.conv_shortcut = MeanPoolConv(input_dim, output_dim, 1, he_init = False)
        
    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return h + shortcut
    
    
#resnet-based generator        
class Generator(nn.Module):
    
    def __init__(self, z_dim, out_dim, num_channels):
        super(Generator, self).__init__()
        self.num_channels = num_channels
        self.input = nn.Linear(z_dim, 4*4*num_channels)
        self.block1 = ResidualBlock(num_channels, num_channels, 3, resample = 'up')
        self.block2 = ResidualBlock(num_channels, num_channels, 3, resample = 'up')
        self.block3 = ResidualBlock(num_channels, num_channels, 3, resample = 'up')
        self.bn = nn.BatchNorm2d(num_channels)
        self.output = nn.Conv2d(num_channels, out_dim, 3, padding = 1, padding_mode = 'replicate')
        
    def forward(self, x, label):
        h = self.input(x).view(-1, self.num_channels, 4, 4)
        h = self.block1(h, label)
        h = self.block2(h, label)
        h = self.block3(h, label)
        h = F.relu(self.bn(h))
        return torch.tanh(self.output(h))
        
        
class Discriminator(nn.Module):
    
    def __init__(self, x_dim, out_dim, num_channels):
        super(Discriminator, self).__init__()
        self.input = OptimizedResBlockDisc1(x_dim, num_channels, 3)
        self.block1 = ResidualBlock(num_channels, num_channels, 3, resample = 'down', normalize = 'layer')
        self.block2 = ResidualBlock(num_channels, num_channels, 3, normalize = 'layer')
        self.block3 = ResidualBlock(num_channels, num_channels, 3, normalize = 'layer')
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(num_channels, out_dim)
        
    def forward(self, x, label):
        h = self.input(x)
        h = self.block1(h, label)
        h = self.block2(h, label)
        h = self.block3(h, label)
        h = self.GAP(F.relu(h)).squeeze()
        return self.output(h)
    
    def gradient_penalty(self, real_samples, fake_samples, labels):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda()
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.forward(interpolates, labels).view(-1, 1)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).cuda()
        fake.requires_grad = False
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradients = gradients[0].view(gradients[0].size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
        
        