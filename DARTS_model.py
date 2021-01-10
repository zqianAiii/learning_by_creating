import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from operations import *


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


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))
    
    
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
    
    
    
class Discriminator(nn.Module):

    def __init__(self, C, num_classes, layers, criterion=nn.BCEWithLogitsLoss(), steps=4, multiplier=4, stem_multiplier=3):
        super(Discriminator, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier*C
        self.stem0 = nn.Sequential(
          nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
          nn.BatchNorm2d(C_curr)
        )
        self.stem1 = nn.Sequential(
          nn.Conv2d(num_classes, C_curr, 3, padding=1, bias=False),
          nn.BatchNorm2d(C_curr)
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
        self.classifier = nn.Linear(C_prev, 1)

        self._initialize_alphas()


    def forward(self, x0, x1):
        s0 = self.stem0(x0)
        s1 = self.stem1(x1)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
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
