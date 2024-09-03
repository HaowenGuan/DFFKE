nohup python3 /home/seazer/code/DFFKE/main.py --group_experiment experiment_CIFAR100 --config_file Local.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Local.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --group_experiment experiment_CIFAR100 --config_file LG_FedAvg.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_LG_FedAvg.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --group_experiment experiment_CIFAR100 --config_file FedGen.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_FedGen.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --group_experiment experiment_CIFAR100 --config_file FedGH.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_FedGH.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --group_experiment experiment_CIFAR100 --config_file FML.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_FML.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --group_experiment experiment_CIFAR100 --config_file FedKD.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_FedKD.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --group_experiment experiment_CIFAR100 --config_file FedDistill.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_FedDistill.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --group_experiment experiment_CIFAR100 --config_file FedProto.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_FedProto.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --group_experiment experiment_CIFAR100 --config_file FedKTL.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_FedKTL.log &
