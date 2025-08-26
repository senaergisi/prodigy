import functools
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np

import copy
import datasets
import models

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def flatten_params(params):
    return np.concatenate([i.data.cpu().numpy().flatten() for i in params])

def row_into_parameters(row, parameters):
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x,y:x*y, param.shape)
        current_data = row[offset:offset + new_size]

        param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size

class Client:
    def __init__(self, client_id, local_iter, batch_size, is_malicious, momentum, data_source, idx):
        self.batch_size = batch_size
        self.is_malicious = is_malicious
        self.data_source = data_source
        self.client_id = client_id
        self.local_iter = local_iter
        self.learning_rate = None
        self.grads = None
        self.prev_params = None
        if data_source == 'CIFAR10':
            self.data_set = datasets.CIFAR10_local('./data', idx, True, datasets.transform_dataset(True, data_source), None, True) 
            self.net = models.CIFAR10Net() 
        elif data_source == 'FEMNIST':
            self.data_set = datasets.FEMNIST_local( idx, True, datasets.transform_dataset(True, data_source), None, True) 
            self.net = models.FEMNISTNet() 
        self.momentum = momentum
        self.original_params = None
        self.train_loader = torch.utils.data.DataLoader(self.data_set, batch_size, shuffle=True, drop_last=True)
        self.train_iterator = iter(cycle(self.train_loader))


    def train(self, data, target):        
        self.optimizer.zero_grad()
        net_out = self.net(data)
        if self.data_source == 'CIFAR10':
            loss = nn.NLLLoss()(net_out, target) 
        elif self.data_source == 'FEMNIST':
            loss = nn.functional.cross_entropy(net_out, target)
        loss.backward()
        self.optimizer.step() 

    def step(self, current_params, learning_rate):
        self.prev_params = copy.deepcopy(current_params)
        row_into_parameters(current_params, self.net.parameters())
        if self.data_source == 'CIFAR10':
            self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate,  weight_decay=1e-2) 
        elif self.data_source == 'FEMNIST':
            self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate)
        
        for iteration in range(self.local_iter):
            try:
                data, target = next(self.train_iterator)
            except StopIteration:
                dataloader_iterator = iter(self.train_loader)
                data, target = next(self.train_iterator)
            self.train(data, target)

        grads = (current_params-np.concatenate([param.data.numpy().flatten() for param in self.net.parameters()]))/learning_rate
        if self.grads is None:
            self.grads = (1-self.momentum) * grads
        else:
            self.grads = self.momentum * self.grads + (1-self.momentum) * grads
