import main

def run_experiments():
    experiment_configs = [
        {'dataname':'CIFAR10', 'partition': 'noniiddir', 'clients_count': 10, 'mal_prop': 0.3, 'predefense': 'NoDefense', 'defense': 'Prodigy', 'attack': 'FOE_optim100', 'batch_size': 64, 'rounds': 2000, 'learning_rate': 0.05, 'momentum': 0.9, 'local_iteration': 1, 'test_step':10 ,'seedd': 0},
        ]
     
    for config in experiment_configs:
        print(f"Running experiment with config: {config}")
        main.main(**config)

if __name__ == "__main__":

    run_experiments()
