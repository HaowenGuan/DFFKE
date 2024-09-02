nohup python3 /home/seazer/code/DFFKE/main.py --config_file experiment_CIFAR100/Local.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Local.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --config_file experiment_CIFAR100/LG_FedAvg.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_LG_FedAvg.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --config_file experiment_CIFAR100/FedGen.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_FedGen.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --config_file experiment_CIFAR100/FedGH.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_FedGH.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --config_file experiment_CIFAR100/FML.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_FML.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --config_file experiment_CIFAR100/FedKD.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_FedKD.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --config_file experiment_CIFAR100/FedDistill.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_FedDistill.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --config_file experiment_CIFAR100/FedProto.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_FedProto.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --config_file experiment_CIFAR100/FedKTL.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_FedKTL.log &