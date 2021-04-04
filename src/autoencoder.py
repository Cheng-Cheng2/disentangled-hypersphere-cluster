import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def MakePermuteMatrix(ofsize):
    return Variable(torch.eye(ofsize)[torch.LongTensor(np.random.permutation(ofsize)), :])


def SignMatrix(ofsize):
    return Variable(torch.diag((torch.randn(ofsize) > 0.5).long() * 2 - 1).float())


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hd):
        super(Generator, self).__init__()
        # self.map0 = nn.Linear(input_size, input_size)
        self.scalar = nn.Parameter(torch.tensor(1.0))
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, hidden_size)
        self.map4 = nn.Linear(hidden_size, hidden_size)
        # value (input) to freq (output) space, to pool
        #self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        #self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        #self.batchnorm3 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = F.leaky_relu(self.map1(x))
        x = F.leaky_relu(self.map2(x))
        x = F.leaky_relu(self.map3(x))
        return F.leaky_relu(self.map4(x))

    def lastlayer(self, x):
        return F.leaky_relu(self.map4(F.leaky_relu(x)), 1e-1)


class Clusterer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Clusterer, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        # self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        # self.batchnorm2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = F.leaky_relu(self.map1(x))
        x = F.leaky_relu(self.map2(x))
        x = F.softmax(self.map3(x), dim=1)
        return x


class Outputter(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, injectivity_by_positivity=False):
        ''' Can get injectivity by strictly monotonicly increasing activations and positive definite matrix, and
        a positive-element matrix is pd.
        '''
        super(Outputter, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, hidden_size)
        self.map4 = nn.Linear(hidden_size, output_size)
        self.ibp = injectivity_by_positivity
        #self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        #self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        #self.batchnorm3 = nn.BatchNorm1d(hidden_size)

    def enforce_pd(self):
        if self.ibp:
            self.map1.weight.data.clamp_(min=1e-10)
            self.map2.weight.data.clamp_(min=1e-10)
            # self.map3.weight.data.clamp_(min=1e-10)
        else:
            print(
                'Error: only call enforce_pd when injectivity_by_positivity is activated')

    def forward(self, x):
        x = F.leaky_relu(self.map1(x))
        x = F.leaky_relu(self.map2(x))
        x = F.leaky_relu(self.map3(x))
        x = F.leaky_relu(self.map4(x))
        return x


class Enforcer(nn.Module):
    def __init__(self, loss, similarity, similarity2=None):
        super(Enforcer, self).__init__()
        self.loss = loss
        self.similarity = similarity
        self.similarity2 = similarity2  # for t-SNE where similarities are normal- and t-

    def forward(self, x, y):
        if self.similarity2 is None:
            return self.loss(self.similarity(x), self.similarity(y))
        else:
            return self.loss(self.similarity(x), self.similarity2(y))
