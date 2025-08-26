import numpy as np
import copy
import predefenses
import defenses

class ALIE(object):
    def __init__(self, num_std):
        self.num_std = num_std
        self.grads_mean = None
        self.grads_stdev = None

    def attack(self, mal_users, all_users):
        if len(mal_users) == 0:
            return

        hon_users_grads = []
        for usr in all_users:
            if usr.is_malicious == False:
                hon_users_grads.append(usr.grads)

        self.grads_mean = np.mean(hon_users_grads, axis=0)
        self.grads_stdev = np.var(hon_users_grads, axis=0) ** 0.5

        if self.num_std == 0:
            return

        mal_grads = self.grads_mean - self.num_std * self.grads_stdev
        for usr in mal_users:
            usr.grads = copy.deepcopy(mal_grads)

class ALIE_optim(object):
    def __init__(self, num_std, predfns, dfns):
        self.num_std = num_std
        self.grads_mean = None
        self.grads_stdev = None
        self.predfns = predfns
        self.dfns = dfns
    
    def cal_l2_dist(self, clients_grads, f, prev_weights):
        current_grads = predefenses.predefend[self.predfns](clients_grads, len(clients_grads), f)
        current_grads = defenses.defend[self.dfns](current_grads, len(current_grads), f, prev_weights)
        return np.linalg.norm(current_grads-self.grads_mean)

    def attack(self, mal_users, all_users):
        if len(mal_users) == 0:
            return

        hon_users_grads = np.array([copy.deepcopy(usr.grads) for usr in all_users if not usr.is_malicious])
        self.grads_mean = np.mean(hon_users_grads, axis=0)
        self.grads_stdev = np.var(hon_users_grads, axis=0) ** 0.5

        if self.num_std == 0:
            return

        z_range = list(np.arange(0.25 * self.num_std, 2.01, 0.25 *  self.num_std)) 
        print(z_range)
        max_dist = 0
        best_z = self.num_std
        
        
        for z in z_range:
            mal_grads = self.grads_mean - z * self.grads_stdev
            all_grads = np.vstack((hon_users_grads, np.array([mal_grads] * len(mal_users))))

            dist = self.cal_l2_dist(all_grads, len(mal_users), all_users[0].prev_params)
            if  dist > max_dist:
                max_dist = dist
                best_z = z
        print("chosen z is:", best_z)
        mal_grads = self.grads_mean - best_z * self.grads_stdev
        for usr in mal_users:
            usr.grads = copy.deepcopy(mal_grads)


class FOE(object):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.grads_mean = None

    def attack(self, mal_users, all_users):
        if len(mal_users) == 0:
            return

        hon_users_grads = []
        for usr in all_users:
            if usr.is_malicious == False:
                hon_users_grads.append(usr.grads)

        self.grads_mean = np.mean(hon_users_grads, axis=0)

        mal_grads = - self.epsilon * self.grads_mean
        for usr in mal_users:
            usr.grads = copy.deepcopy(mal_grads)


class FOE_optim(object):
    def __init__(self, epsilon,predfns, dfns):
        self.epsilon = epsilon
        self.grads_mean = None
        self.predfns = predfns
        self.dfns = dfns
    
    def cal_l2_dist(self, clients_grads, f, prev_weights):
        current_grads = predefenses.predefend[self.predfns](clients_grads, len(clients_grads), f)
        current_grads = defenses.defend[self.dfns](current_grads, len(current_grads), f, prev_weights)
        return np.linalg.norm(current_grads-self.grads_mean)


    def attack(self, mal_users, all_users):
        if len(mal_users) == 0:
            return

        hon_users_grads = np.array([copy.deepcopy(usr.grads) for usr in all_users if not usr.is_malicious])

        self.grads_mean = np.mean(hon_users_grads, axis=0)


        eps_range = [0.1*self.epsilon, 0.2*self.epsilon,0.3*self.epsilon,0.4*self.epsilon,0.5*self.epsilon,0.6*self.epsilon,0.7*self.epsilon,0.8*self.epsilon,0.9*self.epsilon, self.epsilon]
        max_dist = 0
        best_eps = self.epsilon
        for eps in eps_range:
            mal_grads = - eps * self.grads_mean
            all_grads = np.vstack((hon_users_grads, np.array([mal_grads] * len(mal_users))))
            dist = self.cal_l2_dist(all_grads, len(mal_users), all_users[0].prev_params)
            if  dist > max_dist:
                max_dist = dist
                best_eps = eps

        print("chosen eps is:", best_eps)
        mal_grads = - best_eps * self.grads_mean
        for usr in mal_users:
            usr.grads = copy.deepcopy(mal_grads)

class SF(object):
    def __init__(self):
        pass
    def attack(self, mal_users, all_users):
        if len(mal_users) == 0:
            return
        for usr in mal_users:
            usr.grads = - usr.grads


class LF(object):
    def __init__(self):
        pass
    def attack(self, mal_users, all_users):
        if len(mal_users) == 0:
            return
        for usr in mal_users:
            usr.data_set.fliplabels()





