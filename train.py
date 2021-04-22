import os
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from models import Generator, resnet_transfer
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
clf_lr = 0.05
betas = (0, 0.9)
n_critic = 1
LAMBDA = 10
gamma = 0.5
batch_size = 8
num_epochs = 50

#load data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
torch.manual_seed(42)
trainset, valset = torch.utils.data.random_split(trainset, [len(trainset)//2, len(trainset)//2])
torch.manual_seed(torch.initial_seed())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

testset = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train():

    try:
        checkpoint = torch.load('checkpoint.pth')
        model, optimizer, scheduler, epoch = checkpoint['model'], checkpoint['optim'], checkpoint['sched'], checkpoint['epoch']
        load = True
        print('model loaded')
    except:
        load = False
        epoch = 0
        try:
            checkpoint = torch.load('GAN_checkpoint.pth')
            model = checkpoint['model']
        except:
            print('GAN missing')
            return
        print('model created')

    G = Generator(G_in_dim, G_out_dim, num_channels)
    D = Discriminator(16, 5)
    clf =  resnet_transfer()

    G.load_state_dict(model[0])
    D.load_state_dict(model[1])
    D.load_alphas(model[2])
    if load:
        clf.load_state_dict(model[3])
        
    if torch.cuda.is_available():
        G, D, clf = G.cuda(), D.cuda(), clf.cuda()

    optim_G = optim.Adam(G.parameters(), lr = G_lr, betas = betas)
    optim_D = optim.Adam(D.parameters(), lr = D_lr, betas = betas)
    optim_arch = optim.Adam(D.arch_parameters(), lr = 3e-5,  betas = (0.5, 0.999), weight_decay = 1e-3)
    optim_clf = optim.SGD(clf.parameters(), lr = clf_lr, momentum = 0.9, weight_decay = 3e-4)
    scheduler_clf = optim.lr_scheduler.CosineAnnealingLR(optim_clf, num_epochs, eta_min = 1e-5)
    
    if load:
        optim_G.load_state_dict(optimizer[0])
        optim_D.load_state_dict(optimizer[1])
        optim_arch.load_state_dict(optimizer[2])
        optim_clf.load_state_dict(optimizer[3])
        scheduler_clf.load_state_dict(scheduler[0])
    
    G.train()
    D.train()
    clf.train()

    while epoch < num_epochs:

        total = 0
        G_running_loss = 0
        D_running_real_loss = 0
        D_running_fake_loss = 0
        clf_running_train_real_loss = 0
        clf_running_train_fake_loss = 0
        clf_running_val_loss = 0
        clf_train_real_correct = 0
        clf_train_fake_correct = 0
        clf_val_correct = 0

        for i, ((images, labels), (val_images, val_labels)) in enumerate(zip(trainloader, valloader)):

            mini_batch = images.size()[0]
            x_, labels = images.cuda(non_blocking = True), labels.cuda(non_blocking = True)
            val_images, val_labels = val_images.cuda(non_blocking = True), val_labels.cuda(non_blocking = True)

            # Train discriminator
            optim_D.zero_grad()
            optim_arch.zero_grad()
            D_real_decision = D(x_, labels).squeeze()
            #D_real_loss = D.loss(D_real_decision, label_smooth*y_real_)
            D_real_loss = -torch.mean(D_real_decision)
            D_running_real_loss += D_real_loss.item()

            z_ = torch.randn(mini_batch, G_in_dim, device = torch.device('cuda'))
            gen_image = G(z_, labels)

            D_fake_decision = D(gen_image, labels).squeeze()
            #D_fake_loss = D.loss(D_fake_decision, y_fake_)
            D_fake_loss = torch.mean(D_fake_decision)
            D_running_fake_loss += D_fake_loss.item()
            
            gp = D.gradient_penalty(x_, gen_image, labels)

            D_loss = D_real_loss + D_fake_loss + LAMBDA * gp
            D_loss.backward(create_graph = True)
            optim_D.step()
            
#             for p in D.parameters():
#                 p.data.clamp_(-0.005, 0.005)

            #Train generator
            if i%n_critic == (n_critic-1):
                z_ = torch.randn(mini_batch, G_in_dim, device = torch.device('cuda'))
                gen_image = G(z_, labels)
                optim_G.zero_grad()
                D_fake_decision = D(gen_image, labels).squeeze()
                #G_loss = G.loss(D_fake_decision, y_real_)
                G_loss = -torch.mean(D_fake_decision)
                G_running_loss += G_loss.item()
                G_loss.backward(create_graph = True)
                optim_G.step()
            
            # Train Resnet
            z_ = torch.randn(mini_batch, G_in_dim, device = torch.device('cuda'))
            gen_image = G(z_, labels)
            
            optim_clf.zero_grad()
            clf_fake_decision = clf(gen_image)
            clf_fake_loss = clf.loss(clf_fake_decision, labels)      
            clf_real_decision = clf(x_)
            clf_real_loss = clf.loss(clf_real_decision, labels)

            clf_running_train_real_loss += clf_real_loss.item()
            clf_running_train_fake_loss += clf_fake_loss.item()
            total += mini_batch
            _, predicted = torch.max(clf_real_decision.data, 1)
            clf_train_real_correct += (predicted == labels).sum().item()
            _, predicted = torch.max(clf_fake_decision.data, 1)
            clf_train_fake_correct += (predicted == labels).sum().item()
            
            clf_loss = gamma * clf_fake_loss + clf_real_loss
            clf_loss.backward(create_graph = True)
            optim_clf.step()

            # Train architecture
            y = clf(val_images)
            loss = clf.loss(y, val_labels)
            clf_running_val_loss += loss.item()
            _, predicted = torch.max(y.data, 1)
            clf_val_correct += (predicted == val_labels).sum().item()
            loss.backward()
            optim_arch.step()

            #clear graphs
            for param in G.parameters():
                param.grad = None
            for param in D.parameters():
                param.grad = None
            for param in clf.parameters():
                param.grad = None
            for param in D.arch_parameters():
                param.grad = None
                
                
            if i%250 == 249:
                print('({}, {}), G_loss: {}, D_real_loss: {}, D_fake_loss: {}, clf_train_real_loss: {}, clf_train_fake_loss: {}, clf_val_loss: {}'.format(epoch, i+1, G_running_loss/(i+1), D_running_real_loss/(i+1), D_running_fake_loss/(i+1), clf_running_train_real_loss/(i+1), clf_running_train_fake_loss/(i+1), clf_running_val_loss/(i+1)))
                print('clf_train_real_acc: {}, clf_train_fake_acc: {}, clf_val_acc: {}'.format(clf_train_real_correct/total, clf_train_fake_correct/total, clf_val_correct/total))

        epoch += 1
        scheduler_clf.step()
        
        model = [G.state_dict(), D.state_dict(), D.arch_parameters(), clf.state_dict()]
        optimizer = [optim_G.state_dict(), optim_D.state_dict(), optim_arch.state_dict(), optim_clf.state_dict()]
        scheduler = [scheduler_clf.state_dict()]
        torch.save({'model': model, 'optim': optimizer, 'sched': scheduler, 'epoch': epoch}, 'checkpoint.pth')
        
if __name__ == '__main__':
    train()