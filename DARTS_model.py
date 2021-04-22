import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from operations import *

#candidate operations
PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

#mixed operation betweeen two nodes
class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.GroupNorm(1, C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))
    
#DARTS cell    
class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)
    
    
#DARTS-based discriminator    
class Discriminator(nn.Module):

    def __init__(self, C, layers, criterion=nn.BCEWithLogitsLoss(), steps=4, multiplier=4, stem_multiplier=3, num_classes = 10):
        super(Discriminator, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier*C
        self.stem0 = nn.Sequential(
          nn.Conv2d(3, C_curr//2, 3, padding=1, bias=False),
          nn.GroupNorm(1, C_curr//2)
        )
 
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.emb = nn.Embedding(num_classes, 256)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)

        self._initialize_alphas()


    def forward(self, x0, x1):
        y0 = self.stem0(x0)
        s0 = s1 = torch.cat([y0, y0], 1)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.fc2(F.relu(self.fc1(torch.cat([out.view(out.size(0),-1), self.emb(x1)], 1))))
        return logits

    def loss(self, x, target):
        return self._criterion(x, target) 

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(PRIMITIVES)

        if torch.cuda.is_available:
            self.alphas_normal = Variable(torch.zeros((k, num_ops), device = 'cuda'), requires_grad=True)
            self.alphas_reduce = Variable(torch.zeros((k, num_ops), device = 'cuda'), requires_grad=True)
        else:
            self.alphas_normal = Variable(torch.zeros(k, num_ops), requires_grad=True)
            self.alphas_reduce = Variable(torch.zeros(k, num_ops), requires_grad=True)

        self._arch_parameters = [
          self.alphas_normal,
          self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters
    
    def load_alphas(self, alphas):
        self.alphas_normal, self.alphas_reduce = alphas
        if torch.cuda.is_available:
            self.alphas_normal, self.alphas_reduce = self.alphas_normal.cuda(), self.alphas_reduce.cuda()
                
        self._arch_parameters = [
          self.alphas_normal,
          self.alphas_reduce,
        ]
        
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

