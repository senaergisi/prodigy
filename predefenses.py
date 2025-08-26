import numpy as np
from collections import defaultdict


class PreDefenseTypes:
    NoDefense = 'NoDefense'
    NNM = 'NNM'

    def __str__(self):
        return self.value

def no_defense(clients_grads, clients_count, corrupted_count):
    return clients_grads

def _krum_create_distances(clients_grads):
    distances = defaultdict(dict)
    for i in range(len(clients_grads)):
        for j in range(i):
            distances[i][j] = distances[j][i] = np.linalg.norm(clients_grads[i] - clients_grads[j]) ** 2
    return distances

def nnm(clients_grads, clients_count, corrupted_count):
    neighborhood = clients_count - corrupted_count - 1
    distances = _krum_create_distances(clients_grads)
    
    resulting_grads = np.zeros(clients_grads.shape, dtype=float)
    for client in distances.keys():
        distances[client] = dict(sorted(distances[client].items(),key=lambda item: item[1])) 
        idx_to_be_averaged = list(distances[client].keys())
        idx_to_be_averaged = idx_to_be_averaged[: neighborhood]
        idx_to_be_averaged.append(client) 
        resulting_grads[client,:] = np.mean(clients_grads[idx_to_be_averaged], axis=0)

    return resulting_grads



predefend = {PreDefenseTypes.NoDefense: no_defense, PreDefenseTypes.NNM: nnm}
