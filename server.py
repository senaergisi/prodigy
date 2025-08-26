import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
import torch.optim as optim

import datasets
import defenses
import predefenses
import client
import models

class Server:
    def __init__(self, clients, malicious_proportion, batch_size, learning_rate, data_source):
        self.clients = clients
        self.data_source = data_source
        self.mal_prop = malicious_proportion
        self.learning_rate = learning_rate
        if data_source == 'CIFAR10':
            self.data_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=datasets.transform_dataset(False, data_source) )
            self.test_net = models.CIFAR10Net() 
        elif data_source == 'FEMNIST':
            idx = [c.client_id for c in self.clients]
            print(idx)
            self.data_set = datasets.FEMNIST_local( idx, False, datasets.transform_dataset(False, data_source), None, True) 
            self.test_net = models.FEMNISTNet()
        self.test_loader = torch.utils.data.DataLoader(self.data_set, batch_size=batch_size, shuffle=False)
        if data_source == 'CIFAR10':
            self.optimizer = optim.SGD(self.test_net.parameters(), lr=learning_rate, weight_decay=1e-2)
        elif data_source == 'FEMNIST':
            self.optimizer = optim.SGD(self.test_net.parameters(), lr=learning_rate)
        self.current_weights = np.concatenate([i.data.numpy().flatten() for i in self.test_net.parameters()])
        self.clients_grads = np.empty((len(clients), len(self.current_weights)), dtype=self.current_weights.dtype)
        self.velocity = np.zeros(self.current_weights.shape, self.clients_grads.dtype)
       
    def dispatch_weights(self, cur_round): 
        for usr in self.clients:
            usr.step(self.current_weights, self.learning_rate)

    def collect_gradients(self):
        for idx, usr in enumerate(self.clients):
            self.clients_grads[idx, :] = usr.grads

    def defend(self, predefense_method, defense_method, cur_round):
        current_grads = predefenses.predefend[predefense_method](self.clients_grads, len(self.clients), int(len(self.clients)*self.mal_prop))
        current_grads = defenses.defend[defense_method](current_grads, len(current_grads), int(len(current_grads)*self.mal_prop), self.current_weights)
        self.current_weights -= self.learning_rate*current_grads

    def test(self):
        client.row_into_parameters(self.current_weights, self.test_net.parameters())
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:

                net_out = self.test_net(data)
                if self.data_source == 'CIFAR10':
                    loss = nn.NLLLoss()(net_out, target) 
                elif self.data_source == 'FEMNIST':
                    loss = nn.functional.cross_entropy(net_out, target)
                test_loss += loss.data.item()
                pred = net_out.data.max(1)[1]  
                correct += pred.eq(target.data).sum()

        test_loss /= len(self.test_loader.dataset)
        return test_loss, correct

