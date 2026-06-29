SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
DFFKE_DIR="${DFFKE_DIR:-$(CDPATH= cd "$SCRIPT_DIR/.." && pwd)}"
cd "$DFFKE_DIR" || exit 1
mkdir -p logs

nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_TinyImageNet_Dir1.0_HtFE1_PFL --config_file Local.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir1.0_HtFE1_PFL_Local.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_TinyImageNet_Dir1.0_HtFE1_PFL --config_file LG_FedAvg.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir1.0_HtFE1_PFL_LG_FedAvg.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_TinyImageNet_Dir1.0_HtFE1_PFL --config_file FedGen.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir1.0_HtFE1_PFL_FedGen.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_TinyImageNet_Dir1.0_HtFE1_PFL --config_file FedGH.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir1.0_HtFE1_PFL_FedGH.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_TinyImageNet_Dir1.0_HtFE1_PFL --config_file FML.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir1.0_HtFE1_PFL_FML.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_TinyImageNet_Dir1.0_HtFE1_PFL --config_file FedKD.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir1.0_HtFE1_PFL_FedKD.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_TinyImageNet_Dir1.0_HtFE1_PFL --config_file FedDistill.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir1.0_HtFE1_PFL_FedDistill.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_TinyImageNet_Dir1.0_HtFE1_PFL --config_file FedProto.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir1.0_HtFE1_PFL_FedProto.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_TinyImageNet_Dir1.0_HtFE1_PFL --config_file FedTGP.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir1.0_HtFE1_PFL_FedTGP.log && \
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_TinyImageNet_Dir1.0_HtFE1_PFL --config_file FedKTL.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir1.0_HtFE1_PFL_FedKTL.log &
nohup python3 "$DFFKE_DIR/main.py" --group_experiment experiment_TinyImageNet_Dir1.0_HtFE1_PFL --config_file DFFKE.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir1.0_HtFE1_PFL_DFFKE.log &
