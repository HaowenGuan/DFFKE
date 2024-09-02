nohup python3 /home/seazer/code/DFFKE/main.py --config_file experiment_CIFAR100/Local.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' &> logs/experiment_CIFAR100_Local.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --config_file experiment_CIFAR100/LG_FedAvg.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' &> experiment_CIFAR100/LG_FedAvg.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --config_file experiment_CIFAR100/LG_FedAvg.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' &> experiment_CIFAR100/LG_FedAvg.log &
