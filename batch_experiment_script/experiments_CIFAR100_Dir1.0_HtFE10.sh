SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
DFFKE_DIR="${DFFKE_DIR:-$(CDPATH= cd "$SCRIPT_DIR/.." && pwd)}"
cd "$DFFKE_DIR" || exit 1
mkdir -p logs

nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_CIFAR100_Dir1.0_HtFE10 --config_file Local.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE10_Local.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_CIFAR100_Dir1.0_HtFE10 --config_file LG_FedAvg.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE10_LG_FedAvg.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_CIFAR100_Dir1.0_HtFE10 --config_file FedGen.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE10_FedGen.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_CIFAR100_Dir1.0_HtFE10 --config_file FedGH.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE10_FedGH.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_CIFAR100_Dir1.0_HtFE10 --config_file FML.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE10_FML.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_CIFAR100_Dir1.0_HtFE10 --config_file FedKD.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE10_FedKD.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_CIFAR100_Dir1.0_HtFE10 --config_file FedDistill.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE10_FedDistill.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_CIFAR100_Dir1.0_HtFE10 --config_file FedProto.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE10_FedProto.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_CIFAR100_Dir1.0_HtFE10 --config_file FedTGP.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE10_FedTGP.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_CIFAR100_Dir1.0_HtFE10 --config_file FedKTL.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE10_FedKTL.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_CIFAR100_Dir1.0_HtFE10 --config_file DFFKE.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE10_DFFKE.log &
