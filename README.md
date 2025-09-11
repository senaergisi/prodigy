# ProDiGy: Proximity- and Dissimilarity-Based Byzantine-Robust Federated Learning

The following table can be used for the selection of z for ALIE Attack.
| Malicious Presence | # Clients | # Byzantine clients | Calculation | z |
|--------|------|---------|-------|-------|
| 10% | N=20 | f=2 | s=9, phi(z)<11/20=0.55 | z=0.12|
| 20% | N=10 | f=2 | s=4, phi(z)<6/10=0.6 | z=0.25|
| 20% | N=20 | f=4 | s=7, phi(z)<13/20=0.65 | z=0.38|
| 20% | N=30 | f=6 | s=10, phi(z)<20/30=0.66 | z=0.41|
| 20% | N=50 | f=10 | s=16, phi(z)<34/50=0.68 |z=0.46|
| 30% | N=10 | f=3 | s=3, phi(z)<7/10=0.7 | z=0.52|
| 30% | N=20 | f=6 | s=5, phi(z)<15/20=0.75 | z=0.67|
| 30% | N=100 | f=30 | s=21, phi(z)<79/100=0.79 | z=0.8|


FEMNIST data is originally created by running the following command on leaf framework [TalwalkarLab/leaf](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist):  
./preprocess.sh -s niid --sf 1.0 -k 350 -t sample
Due to storage constraints on Github only a fraction of data is sampled with the command below and added to the folder min350 here.  
./preprocess.sh -s niid --sf .022 -k 350 -t sample

## Acknowledgements 
This repository benefited from the following resources, projects, and references:
- [moranant/attacking_distributed_learning](https://github.com/moranant/attacking_distributed_learning)
- [epfml/byzantine-robust-noniid-optimizer](https://github.com/epfml/byzantine-robust-noniid-optimizer)
  Licensed under MIT License.
- [Xtra-Computing/NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench)
- [LPD-EPFL/robust-collaborative-learning](https://github.com/LPD-EPFL/robust-collaborative-learning)
  Licensed under MIT License.
- [sacs-epfl/decentralized-learning-simulator](https://github.com/sacs-epfl/decentralized-learning-simulator)
  Licensed under MIT License.
- [LPD-EPFL/robust-collaborative-learning](https://github.com/LPD-EPFL/robust-collaborative-learning)
  Licensed under MIT License.
