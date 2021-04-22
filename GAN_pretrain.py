import os
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import amp

from models import Generator
from DARTS_model import Discriminator

# Parameters
image_size = 32
label_dim = 1
G_in_dim = 128
G_out_dim = 3
D_in_dim = 3
D_out_dim = 1
num_channels = 128

G_lr = 1e-4
D_lr = 3e-4
betas = (0, 0.9)
n_critic = 1
LAMBDA = 10
batch_size = 32
pretrain_epochs = 1000

#load data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

testset = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#pretrain DARTS-based GAN from scratch
def train():

    try:
        checkpoint = torch.load('GAN_checkpoint.pth')
        model, optimizer, scl, epoch= checkpoint['model'], checkpoint['optim'], checkpoint['scaler'], checkpoint['epoch']
        load = True
        print('model loaded')
    except:
        load = False
        epoch = 0
        print('model created')

    G = Generator(G_in_dim, G_out_dim, num_channels)
    D = Discriminator(16, 5)

    if load:
        G.load_state_dict(model[0])
        D.load_state_dict(model[1])
        D.load_alphas(model[2])
        
    if torch.cuda.is_available():
        G, D = G.cuda(), D.cuda()

    optim_G = optim.Adam(G.parameters(), lr = G_lr, betas = betas)
    optim_D = optim.Adam(D.parameters(), lr = D_lr, betas = betas)
    optim_arch = optim.Adam(D.arch_parameters(), lr = 3e-4, betas = (0.5, 0.999), weight_decay = 1e-3)
    scaler = amp.GradScaler()
    
    if load:
        optim_G.load_state_dict(optimizer[0])
        optim_D.load_state_dict(optimizer[1])
        optim_arch.load_state_dict(optimizer[2])
        scaler.load_state_dict(scl)
    
    G.train()
    D.train()

    while epoch < pretrain_epochs:

        G_running_loss = 0
        D_running_real_loss = 0
        D_running_fake_loss = 0
        D_running_loss = 0

        for i, (images, labels) in enumerate(trainloader):

            mini_batch = images.size()[0]
            x_, labels = images.cuda(non_blocking = True), labels.cuda(non_blocking = True)

            y_real_ = torch.ones(mini_batch, device = torch.device('cuda'))
            y_fake_ = torch.zeros(mini_batch, device = torch.device('cuda'))

            # Train discriminator
            optim_D.zero_grad()
            optim_arch.zero_grad()
            
            with amp.autocast():
                D_real_decision = D(x_, labels).squeeze()
                #D_real_loss = G.loss(D_real_decision, y_real_)
                D_real_loss = -torch.mean(D_real_decision)
                D_running_real_loss += D_real_loss.item()

                z_ = torch.randn(mini_batch, G_in_dim, device = torch.device('cuda'))
                gen_image = G(z_, labels)

                D_fake_decision = D(gen_image.detach(), labels).squeeze()
                #D_fake_loss = G.loss(D_fake_decision, y_fake_)
                D_fake_loss = torch.mean(D_fake_decision)
                D_running_fake_loss += D_fake_loss.item()

                gp = D.gradient_penalty(x_, gen_image.detach(), labels)

                D_loss = D_real_loss + D_fake_loss + LAMBDA * gp
                D_running_loss += D_loss.item()
                
            scaler.scale(D_loss).backward()
            
            scaler.step(optim_D)
            scaler.step(optim_arch)

            #Train generator
            if i%n_critic == (n_critic-1):
                optim_G.zero_grad()
                with amp.autocast():
                    D_fake_decision = D(gen_image, labels).squeeze()
                    #G_loss = G.loss(D_fake_decision, y_real_)
                    G_loss = -torch.mean(D_fake_decision)
                    G_running_loss += G_loss.item()
                scaler.scale(G_loss).backward()
                scaler.step(optim_G)
                
            scaler.update()

        epoch += 1
        
        print('({}, {}), G_loss: {}, D_real_loss: {}, D_fake_loss: {}, D_loss: {} {}'.format(epoch, i, G_running_loss/i*n_critic, D_running_real_loss/i, D_running_fake_loss/i, -(D_running_real_loss + D_running_fake_loss)/i, D_running_loss/i))

        model = [G.state_dict(), D.state_dict(), D.arch_parameters()]
        optimizer = [optim_G.state_dict(), optim_D.state_dict(), optim_arch.state_dict()]
        torch.save({'model': model, 'optim': optimizer, 'scaler': scaler.state_dict(), 'epoch': epoch}, 'GAN_checkpoint.pth')
        
if __name__ == '__main__':
    train()