#nohup python3 /home/seazer/code/DFFKE/main.py --config_file gpu0.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE1_DFFKE_98_Y_diff_eval.log &
#nohup python3 /home/seazer/code/DFFKE/main.py --config_file gpu1.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE1_DFFKE_98_X_diff_eval.log &
nohup python3 /home/seazer/code/DFFKE/main.py --group_experiment experiment_CIFAR100_Dir0.1_HtFE1 --config_file DFFKE.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir0.1_HtFE1_DFFKE.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --group_experiment experiment_CIFAR10_Dir0.1_HtFE1 --config_file DFFKE.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR10_Dir0.1_HtFE1_DFFKE.log &
nohup python3 /home/seazer/code/DFFKE/main.py --group_experiment experiment_CIFAR100_Dir1.0_HtFE1 --config_file DFFKE.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE1_DFFKE.log && \
nohup python3 /home/seazer/code/DFFKE/main.py --group_experiment experiment_CIFAR10_Dir1.0_HtFE1 --config_file DFFKE.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR10_Dir1.0_HtFE1_DFFKE.log &
