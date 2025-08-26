import numpy as np
from collections import defaultdict
import copy

class DefenseTypes:
    NoDefense = 'NoDefense'
    Krum = 'Krum'
    TrimmedMean = 'TrimmedMean'
    Median = 'Median'
    GeoMed = 'GeoMed'
    Prodigy = 'Prodigy'
    CClip = 'CClip'
    def __str__(self):
        return self.value

def no_defense(clients_grads, clients_count, corrupted_count, current_weights):
    return np.mean(clients_grads, axis=0)

def median(clients_grads, clients_count, corrupted_count, current_weights):
    return np.median(clients_grads, axis=0)

def _krum_create_distances(clients_grads):
    distances = defaultdict(dict)
    for i in range(len(clients_grads)):
        for j in range(i):
            distances[i][j] = distances[j][i] = np.linalg.norm(clients_grads[i] - clients_grads[j]) **2
    return distances

def RFA(clients_grads, clients_count, corrupted_count, current_weights):
    alphas = [1 / clients_count for _ in range(clients_count)]
    z = np.zeros(clients_grads[0].shape, dtype=float)
    m = clients_count
    nu = 0.1
    T = 3
    if len(alphas) != m:
        raise ValueError

    if nu < 0:
        raise ValueError

    for t in range(T):
        betas = []
        for k in range(m):
            distance = np.linalg.norm(z - clients_grads[k])
            betas.append(alphas[k] / max(distance, nu))

        z = np.zeros(clients_grads[0].shape, dtype=float)
        for k in range(m):
            z += clients_grads[k] * betas[k]
        z /= sum(betas)
    return z
    
def centeredclipping(clients_grads, clients_count, corrupted_count, current_weights):
    L = 3
    tau = 10
    g = copy.deepcopy(current_weights)
    for l in range(L):
        differences = []
        for i in range(clients_count):
            diff = clients_grads[i] - g       
            differences.append(diff * min(1, tau/np.linalg.norm(diff)) )
        g += np.mean(differences, axis=0)
    return g
    
def krum(clients_grads, clients_count, corrupted_count, current_weights, distances=None,return_index=False, debug=False):
    if not return_index:
        assert clients_count >= 2*corrupted_count + 1,('clients_count>=2*corrupted_count + 3', clients_count, corrupted_count)
    neighborhood = clients_count - corrupted_count - 2
    minimal_error = 1e20
    minimal_error_index = -1

    if distances is None:
        distances = _krum_create_distances(clients_grads)

    for client in distances.keys():
        errors = sorted(distances[client].values())
        current_error = sum(errors[:neighborhood])
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = client

    if return_index:
        return minimal_error_index
    else:
        return clients_grads[minimal_error_index]

def trimmed_mean(clients_grads, clients_count, corrupted_count, current_weights):
    number_to_consider = clients_count-2*corrupted_count
    med = np.median(clients_grads, axis=0)
    resulting_grads = np.zeros(clients_grads[0].shape, dtype=float)
    diff = clients_grads-med
    indices = np.argsort(abs(diff.T), axis=1)
    sorted_matrix = np.take_along_axis(diff.T, indices, axis=1)
    mean_values = np.mean(sorted_matrix[:, :number_to_consider], axis=1)
    current_grads = mean_values.T + med
    return current_grads
     
def prodigy(clients_grads, clients_count, corrupted_count, current_weights):
    neighborhood = corrupted_count - 1
    distances = _krum_create_distances(clients_grads)
    
    score1 = {}
    for client in distances.keys():
        distances[client] = dict(sorted(distances[client].items(),key=lambda item: item[1]))  
        idx_to_be_averaged = list(distances[client].keys())
        idx_to_be_averaged = idx_to_be_averaged[: neighborhood]
        idx_to_be_averaged.append(client) 
        mean_grads = np.mean(clients_grads[idx_to_be_averaged], axis=0)
        std_dev_grads = np.var(clients_grads[idx_to_be_averaged], axis=0) ** 0.5
        score1[client] = np.linalg.norm(std_dev_grads)/np.linalg.norm(mean_grads) 
    score2 = {}
    for client in distances.keys():
        errors = sorted(distances[client].values())
        current_error = sum(errors[neighborhood: clients_count-corrupted_count-1]) 
        score2[client] = current_error


    score = {key: score1[key] / score2[key] for key in score1 if key in score2}
    lowest_values = sorted(score.values())[:corrupted_count]
    score = {k: 0 if v in lowest_values else v for k, v in score.items()}
    sumscore = sum(score.values())
    resulting_grads = np.zeros(clients_grads[0].shape, dtype=float)
    for client in score.keys():
        resulting_grads = resulting_grads + (score[client]/sumscore) * clients_grads[client]
 
    return resulting_grads

defend = {DefenseTypes.NoDefense: no_defense, DefenseTypes.Krum: krum, DefenseTypes.Median: median, DefenseTypes.TrimmedMean: trimmed_mean,  DefenseTypes.Prodigy:prodigy, DefenseTypes.GeoMed:RFA, DefenseTypes.CClip:centeredclipping }