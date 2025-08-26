import torch
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms

from collections import defaultdict
import json
import os


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data

def read_data(train_data_dir, test_data_dir):

    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def transform_dataset(is_train, dataset):

    transform = None
    if dataset == 'CIFAR10':
        if is_train:
            transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif dataset == 'FEMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ])
    
    return transform

def find_distribution_map(dataset, partition, clients_count, beta=0.1):
    if dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        n_train = len(trainset)
        if partition == 'iid':
            idxs = np.random.permutation(n_train)
            batch_idxs = np.array_split(idxs, clients_count)
            net_dataidx_map = {i: batch_idxs[i] for i in range(clients_count)}

        elif partition == 'noniiddir':
            min_size = 0
            min_require_size = 10
            K = 10
            net_dataidx_map = {}
            beta = 0.1
            trainlabel = np.array(trainset.targets)
            while min_size < min_require_size:
                idx_batch = [[] for _ in range(clients_count)]
                for k in range(K):
                    idx_k = np.where(trainlabel == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(beta, clients_count))
                    proportions = np.array([p * (len(idx_j) < n_train / clients_count) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(clients_count):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]   
    
    elif dataset == 'FEMNIST':

        if partition == 'noniiddir':
            net_dataidx_map = {user: user for user in range(clients_count)}
        print(net_dataidx_map)
    return net_dataidx_map


class CIFAR10_local(torch.utils.data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = torchvision.datasets.CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
        data = cifar_dataobj.data
        target=np.array(cifar_dataobj.targets) 
       
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target
        
    def fliplabels(self):
        cifar_dataobj = torchvision.datasets.CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
        data = cifar_dataobj.data
        target=np.array(cifar_dataobj.targets) 
       
        if self.dataidxs is not None:
            self.data = data[self.dataidxs]
            self.target = 9-target[self.dataidxs]

    def __getitem__(self, index):
    
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class FEMNIST_local(torch.utils.data.Dataset):
  
    def __init__(self, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        train_clients, train_groups, train_data_temp, test_data_temp = read_data("min350/train", "min350/test")   
        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = [] 
            cur_x = train_data_temp[train_clients[self.dataidxs]]['x']
            cur_y = train_data_temp[train_clients[self.dataidxs]]['y']
            for j in range(len(cur_x)):
                train_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                train_data_y.append(cur_y[j])
            data = train_data_x
            label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(self.dataidxs)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    test_data_y.append(cur_y[j])
            data = test_data_x
            label = test_data_y
        print(len(label))
        return data, label
    def __getitem__(self, index):

        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    def fliplabels(self):       
        for i in range(len(self.data )):
            self.target[i] = 61-self.target[i]
    def __len__(self):
        return len(self.data)
        
def print_data_distribution(clients):
    print("Data Distribution:")
    clients_count = len(clients)
    counts = np.zeros((clients_count, 10), dtype=int)
    for client_id in range(clients_count):
        localdataset = clients[client_id].data_set
        for i in range(len(localdataset)):
            l = int(localdataset.target[i])
            counts[client_id,l] += 1 

    print(counts)

