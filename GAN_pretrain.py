import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from DARTS_model import *
from models import *

# Parameters
image_size = 32
label_dim = 10
G_in_dim = 100
G_out_dim = 3
D_in_dim = 3
D_out_dim = 1
num_channels = [512, 256, 128]

GAN_lr = 0.0002
betas = (0.5, 0.999)
batch_size = 32
pretrain_epochs = 100

#load data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
torch.manual_seed(42)
trainset, valset = torch.utils.data.random_split(trainset, [3*len(trainset)//5, 2*len(trainset)//5])
torch.manual_seed(torch.initial_seed())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train():

    onehot = torch.eye(label_dim, device = torch.device('cuda')).view(label_dim, label_dim, 1, 1)
    fill = torch.zeros([label_dim, label_dim, image_size, image_size], device = torch.device('cuda'))
    for i in range(label_dim):
        fill[i, i, :, :] = 1


    G = Generator(G_in_dim, label_dim, G_out_dim, num_channels)
    D = Discriminator(16, 10, 9)

    if torch.cuda.is_available():
        G, D = G.cuda(), D.cuda()

    optim_G = optim.Adam(G.parameters(), lr = GAN_lr, betas = betas)
    optim_D = optim.SGD(D.parameters(), lr = GAN_lr/2, momentum = 0.9, weight_decay = 3e-4)

    G.train()
    D.train()
    epoch = 0

    while epoch < pretrain_epochs:

        G_running_loss = torch.zeros((1, 1), device = torch.device('cuda'))
        D_running_real_loss = torch.zeros((1, 1), device = torch.device('cuda'))
        D_running_fake_loss = torch.zeros((1, 1), device = torch.device('cuda'))

        for i, (images, labels) in enumerate(trainloader):

            mini_batch = images.size()[0]
            x_ = images.cuda(non_blocking = True)

            y_real_ = torch.ones(mini_batch, device = torch.device('cuda'))
            y_fake_ = torch.zeros(mini_batch, device = torch.device('cuda'))
            c_fill_ = fill[labels]

            # Train discriminator
            optim_D.zero_grad()
            D_real_decision = D(x_, c_fill_).squeeze()
            D_real_loss = D.loss(D_real_decision, y_real_)
            D_running_real_loss += D_real_loss.detach()

            z_ = torch.randn(mini_batch, G_in_dim, device = torch.device('cuda')).view(-1, G_in_dim, 1, 1)
            c_ = (torch.rand(mini_batch, 1) * label_dim).type(torch.LongTensor).squeeze()
            c_onehot_ = onehot[c_]
            gen_image = G(z_, c_onehot_)

            c_fill_ = fill[c_]
            D_fake_decision = D(gen_image, c_fill_).squeeze()
            D_fake_loss = D.loss(D_fake_decision, y_fake_)
            D_running_fake_loss += D_fake_loss.detach()

            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            optim_D.step()

            # Train generator
            z_ = torch.randn(mini_batch, G_in_dim, device = torch.device('cuda')).view(-1, G_in_dim, 1, 1)
            c_ = (torch.rand(mini_batch, 1) * label_dim).type(torch.LongTensor).squeeze()
            c_onehot_ = onehot[c_]

            optim_G.zero_grad()
            gen_image = G(z_, c_onehot_)

            c_fill_ = fill[c_]
            D_fake_decision = D(gen_image, c_fill_).squeeze()
            G_loss = G.loss(D_fake_decision, y_real_)
            G_running_loss += G_loss.detach()
            G_loss.backward()
            optim_G.step()

            if i%100 == 99:
                print('({}, {}), G_loss: {}, D_real_loss: {}, D_fake_loss: {}'.format(epoch, i+1, G_running_loss.item()/(i+1), D_running_real_loss.item()/(i+1), D_running_fake_loss.item()/(i+1)))

        model = [G.state_dict(), D.state_dict()]
        optimizer = [optim_G.state_dict(), optim_D.state_dict()]
        torch.save({'model': model, 'optim': optimizer}, 'GAN_checkpoint.pth')
        epoch += 1
        
if __name__ == '__main__':
    train()