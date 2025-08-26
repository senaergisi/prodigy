import argparse
import torch
import numpy as np


import client
import datasets
import server
import malicious

def main(dataname, partition, clients_count, mal_prop, predefense, defense, attack, batch_size=64, rounds=50, learning_rate=0.1, momentum=0, local_iteration=1, test_step = 10, seedd = 0):
    beta = None
    z = 0
    if partition == 'noniiddir' and dataname=='CIFAR10':
        beta = 0.1
        # beta = input("What is beta in Dirichlet distribution: ")
        # beta = float(beta)

    attacker = None
    if attack == 'ALIE':
        z = input("Please enter z for ALIE: ")
        attacker = malicious.ALIE(z)
    if attack == 'ALIE_optim':
        # z=0.12 # N=20, f=2, mal_prop=0.1
        # z=0.25 # N=10, f=2, mal_prop=0.2
        # z=0.38 # N=20, f=4, mal_prop=0.2
        # z=0.41 # N=30, f=6, mal_prop=0.2
        # z=0.50 # N=50, f=10, mal_prop=0.2
        # z=0.67 # N=20, f=6, mal_prop=0.3
        # z=0.8 # N=100, f=30, mal_prop=0.3
        z = input("Please enter z for ALIE: ")
        z = float(z)
        attacker = malicious.ALIE_optim(z,predefense, defense)
    if attack == 'FOE':
        # epsii = 0.1
        epsii = input("Please enter epsilon for FOE: ")
        epsii = float(epsii)
        attacker = malicious.FOE(epsii)
    if attack == 'FOE_optim':
        epsii = 0.1
        # epsii = input("Please enter epsilon for FOE: ")
        # epsii = float(epsii)
        attacker = malicious.FOE_optim(epsii,predefense, defense)
    
    if attack == 'FOE_optim100':
        epsii = 100
        # epsii = input("Please enter epsilon for FOE: ")
        # epsii = float(epsii)
        attacker = malicious.FOE_optim(epsii,predefense, defense)

    if attack == 'SF':
        attacker = malicious.SF()
    if attack == 'LF':
        attacker = malicious.LF()
    
    torch.manual_seed(seedd)
    np.random.seed(seedd)
    corrupted_count = int(mal_prop * clients_count)

    TEST_STEP = test_step
    distribution_map = datasets.find_distribution_map(dataname, partition, clients_count, beta)

    clients = []
    for client_id in range(clients_count):
        if client_id < corrupted_count:
            is_mal = True
        else:
            is_mal = False
        clients.append(client.Client(client_id, local_iteration, batch_size, is_mal, momentum, dataname, distribution_map[client_id])) 

    mal_clients = [u for u in clients if u.is_malicious]

    if dataname=='CIFAR10':
        datasets.print_data_distribution(clients)
    
    if attack == 'LF':
        attacker.attack(mal_clients, clients)
        
    the_server = server.Server(clients, mal_prop, batch_size, learning_rate, dataname)
    test_size = len(the_server.test_loader.dataset)
    
    print("\nStarting Training...")
    accuracies = []
    test_loss = 0
    for round in range(rounds):
        if np.isnan(test_loss):
            print("round: ", round)    
            print("Test loss is NaN, taking action!")
            if round % TEST_STEP == 0 or round == rounds - 1:
                accuracies.append((round, the_server.learning_rate, accuracy))

        else:
            if round == int(2*rounds/3):
                the_server.learning_rate *= 0.1
            
            the_server.dispatch_weights(round)
            print("round: ", round)    
            if attacker is not None and attack != 'LF':
                attacker.attack(mal_clients, clients)            
            the_server.collect_gradients()
    
            the_server.defend(predefense, defense, round)

            if round % TEST_STEP == 0 or round == rounds - 1: 
                print('testing starts...')
                test_loss, correct = the_server.test()
                accuracy = 100. * float(correct) / test_size
                print('Test set: [{:3d}] Learning rate: {:.4f}, Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(round, the_server.learning_rate, test_loss,
                                                                                                    correct,
                                                                                                    test_size,
                                                                                    accuracy))
                accuracies.append((round, the_server.learning_rate, accuracy))

    np.savetxt('{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(dataname,partition, clients_count, mal_prop, predefense,defense,attack,batch_size,rounds,learning_rate,momentum,local_iteration,seedd), accuracies, delimiter=',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning Experiment for CIFAR-10')
    
    parser.add_argument('-e', '--dataname', default='CIFAR10', choices=['CIFAR10', 'FEMNIST'])

    parser.add_argument('-c', '--partition', default='iid', choices=['iid', 'noniiddir'])

    parser.add_argument('-n', '--users_count', default=51, type=int, help='number of participating users')

    parser.add_argument('-m', '--mal_prop', default=0.24, type=float, help='proportion of malicious users')

    parser.add_argument('-p', '--predefense', default='NoDefense', choices=['NoDefense', 'NNM'])
    
    parser.add_argument('-d', '--defense', default='NoDefense', choices=['NoDefense', 'Krum', 'TrimmedMean', 'Median', 'Mine3remove', 'GeoMed', 'CClip'  ])

    parser.add_argument('-a', '--attack', default='NoAttack', choices=['NoAttack', 'ALIE','ALIE_optim', 'FOE', 'FOE_optim','FOE_optim100', 'SF', 'LF'])

    parser.add_argument('-b', '--batch_size', default=64, type=int, help='batch_size')

    parser.add_argument('-r', '--rounds', default=300, type=int)

    parser.add_argument('-l', '--learning_rate', default=0.1, type=float, help='initial learning rate')

    parser.add_argument('-o', '--momentum', default=0, type=float, help='local momentum beta')

    parser.add_argument('-i', '--local_iteration', default=1, type=int, help='local iteration rounds')

    parser.add_argument('-t', '--test_step', default=10, type=int)
    
    parser.add_argument('-s', '--randomseed', default=0, type=int)

    args = parser.parse_args()

    main(args.dataname, args.partition, args.users_count, args.mal_prop, args.predefense, args.defense, args.attack, args.batch_size, args.rounds, args.learning_rate, args.momentum, args.local_iteration, args.test_step, args.randomseed) 
    



