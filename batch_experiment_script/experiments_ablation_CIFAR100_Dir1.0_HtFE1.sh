SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
DFFKE_DIR="${DFFKE_DIR:-$(CDPATH= cd "$SCRIPT_DIR/.." && pwd)}"
cd "$DFFKE_DIR" || exit 1
mkdir -p logs

# Original
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file HtFE1_original.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/HtFE1_original.log &
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file class_latent_generator.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/class_latent_generator.log &

# Group A
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file y_clustering.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/y_hat_clustering.log &
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file y_hat_clustering_and_y_hat_loss.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/y_hat_clustering_and_y_hat_loss.log &
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file y_clustering_and_y_loss.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/y_clustering_and_y_loss.log &

# Group B
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file Gaussian_DP_0.05.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/Gaussian_DP_0.05.log &
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file Gaussian_DP_0.01.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/Gaussian_DP_0.01.log &
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file Gaussian_DP_0.10.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/Gaussian_DP_0.10.log &
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file Gaussian_DP_0.20.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/Gaussian_DP_0.20.log &
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file Gaussian_DP_0.40.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/Gaussian_DP_0.40.log &
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file Gaussian_DP_4.00.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/Gaussian_DP_4.00.log &


# Group C
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file data_bank_limit_0.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/without_KE_bank_loss.log &
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file data_bank_limit_1.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/data_bank_limit_1.log &
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file data_bank_limit_5.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/data_bank_limit_5.log &
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file data_bank_limit_10.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/data_bank_limit_10.log &
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file data_bank_limit_20.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/data_bank_limit_20.log &

# Group D
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file local_align_80.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/local_align_80.log &
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file local_align_90.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/local_align_90.log &
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file local_align_95.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/local_align_95.log &

# Visualize synthetic data
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file visualize_synthetic_data.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/visualize_synthetic_data.log &

#Group E
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file participants_limit_2.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/participants_limit_2.log && \
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file participants_limit_5.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/participants_limit_5.log && \
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file participants_limit_6.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/participants_limit_6.log && \
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file participants_limit_9.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/participants_limit_9.log &
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file participants_limit_3.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/participants_limit_3.log && \
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file participants_limit_4.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/participants_limit_4.log && \
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file participants_limit_7.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/participants_limit_7.log && \
#nohup python3 "$DFFKE_DIR/main.py" --group_experiment ablation_study --config_file participants_limit_8.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/participants_limit_8.log &

# Group F: Differential Privacy Sensitivity Analysis
#nohup python3 "$DFFKE_DIR/main.py" --device_id 0 --group_experiment experiment_CIFAR100_Dir0.1_HtFE1_PFL --config_file DFFKE_DP_0.5.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir0.1_HtFE1_PFL_DFFKE_DP_0.5.log && \
#nohup python3 "$DFFKE_DIR/main.py" --device_id 0 --group_experiment experiment_CIFAR10_Dir0.1_HtFE1_PFL --config_file DFFKE_DP_0.5.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR10_Dir0.1_HtFE1_PFL_DFFKE_DP_0.5.log &

#nohup python3 "$DFFKE_DIR/main.py" --device_id 0 --group_experiment experiment_CIFAR100_Dir0.1_HtFE1_PFL --config_file DFFKE_DP_0.75.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir0.1_HtFE1_PFL_DFFKE_DP_0.75.log && \
#nohup python3 "$DFFKE_DIR/main.py" --device_id 0 --group_experiment experiment_CIFAR10_Dir0.1_HtFE1_PFL --config_file DFFKE_DP_0.75.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR10_Dir0.1_HtFE1_PFL_DFFKE_DP_0.75.log &

#nohup python3 "$DFFKE_DIR/main.py" --device_id 1 --group_experiment experiment_CIFAR100_Dir1.0_HtFE1_PFL --config_file DFFKE_DP_0.5.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE1_PFL_DFFKE_DP_0.5.log && \
#nohup python3 "$DFFKE_DIR/main.py" --device_id 1 --group_experiment experiment_CIFAR10_Dir1.0_HtFE1_PFL --config_file DFFKE_DP_0.5.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR10_Dir1.0_HtFE1_PFL_DFFKE_DP_0.5.log &

#nohup python3 "$DFFKE_DIR/main.py" --device_id 1 --group_experiment experiment_CIFAR100_Dir1.0_HtFE1_PFL --config_file DFFKE_DP_0.75.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE1_PFL_DFFKE_DP_0.75.log && \
#nohup python3 "$DFFKE_DIR/main.py" --device_id 1 --group_experiment experiment_CIFAR10_Dir1.0_HtFE1_PFL --config_file DFFKE_DP_0.75.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR10_Dir1.0_HtFE1_PFL_DFFKE_DP_0.75.log &

#nohup python3 "$DFFKE_DIR/main.py" --device_id 2 --group_experiment experiment_TinyImageNet_Dir0.1_HtFE1_PFL --config_file DFFKE_DP_0.5.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir0.1_HtFE1_PFL_DFFKE_DP_0.5.log &
#nohup python3 "$DFFKE_DIR/main.py" --device_id 2 --group_experiment experiment_TinyImageNet_Dir0.1_HtFE1_PFL --config_file DFFKE_DP_0.75.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir0.1_HtFE1_PFL_DFFKE_DP_0.75.log &

#nohup python3 "$DFFKE_DIR/main.py" --device_id 3 --group_experiment experiment_TinyImageNet_Dir1.0_HtFE1_PFL --config_file DFFKE_DP_0.5.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir1.0_HtFE1_PFL_DFFKE_DP_0.5.log &
#nohup python3 "$DFFKE_DIR/main.py" --device_id 3 --group_experiment experiment_TinyImageNet_Dir1.0_HtFE1_PFL --config_file DFFKE_DP_0.75.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir1.0_HtFE1_PFL_DFFKE_DP_0.75.log &

#nohup python3 "$DFFKE_DIR/main.py" --device_id 0 --group_experiment experiment_CIFAR10_Dir0.1_HtFE1_PFL --config_file DFFKE_DP_1.5.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR10_Dir0.1_HtFE1_PFL_DFFKE_DP_1.5.log && \
#nohup python3 "$DFFKE_DIR/main.py" --device_id 1 --group_experiment experiment_CIFAR10_Dir1.0_HtFE1_PFL --config_file DFFKE_DP_1.5.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR10_Dir1.0_HtFE1_PFL_DFFKE_DP_1.5.log && \
#nohup python3 "$DFFKE_DIR/main.py" --device_id 2 --group_experiment experiment_CIFAR100_Dir0.1_HtFE1_PFL --config_file DFFKE_DP_1.5.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir0.1_HtFE1_PFL_DFFKE_DP_1.5.log && \
#nohup python3 "$DFFKE_DIR/main.py" --device_id 3 --group_experiment experiment_CIFAR100_Dir1.0_HtFE1_PFL --config_file DFFKE_DP_1.5.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE1_PFL_DFFKE_DP_1.5.log && \
#nohup python3 "$DFFKE_DIR/main.py" --device_id 2 --group_experiment experiment_TinyImageNet_Dir0.1_HtFE1_PFL --config_file DFFKE_DP_1.5.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir0.1_HtFE1_PFL_DFFKE_DP_1.5.log && \
#nohup python3 "$DFFKE_DIR/main.py" --device_id 3 --group_experiment experiment_TinyImageNet_Dir1.0_HtFE1_PFL --config_file DFFKE_DP_1.5.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir1.0_HtFE1_PFL_DFFKE_DP_1.5.log

#nohup python3 "$DFFKE_DIR/main.py" --device_id 0 --group_experiment experiment_CIFAR100_Dir0.1_HtFE1_PFL --config_file DFFKE_DP_1.0.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir0.1_HtFE1_PFL_DFFKE_DP_1.0.log && \
#nohup python3 "$DFFKE_DIR/main.py" --device_id 0 --group_experiment experiment_CIFAR10_Dir0.1_HtFE1_PFL --config_file DFFKE_DP_1.0.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR10_Dir0.1_HtFE1_PFL_DFFKE_DP_1.0.log &
#
#nohup python3 "$DFFKE_DIR/main.py" --device_id 1 --group_experiment experiment_CIFAR100_Dir1.0_HtFE1_PFL --config_file DFFKE_DP_1.0.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE1_PFL_DFFKE_DP_1.0.log && \
#nohup python3 "$DFFKE_DIR/main.py" --device_id 1 --group_experiment experiment_CIFAR10_Dir1.0_HtFE1_PFL --config_file DFFKE_DP_1.0.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR10_Dir1.0_HtFE1_PFL_DFFKE_DP_1.0.log &
#
#nohup python3 "$DFFKE_DIR/main.py" --device_id 2 --group_experiment experiment_TinyImageNet_Dir0.1_HtFE1_PFL --config_file DFFKE_DP_1.0.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir0.1_HtFE1_PFL_DFFKE_DP_1.0.log &
#
#nohup python3 "$DFFKE_DIR/main.py" --device_id 3 --group_experiment experiment_TinyImageNet_Dir1.0_HtFE1_PFL --config_file DFFKE_DP_1.0.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_TinyImageNet_Dir1.0_HtFE1_PFL_DFFKE_DP_1.0.log &


#nohup python3 "$DFFKE_DIR/main.py" --device_id 0 --group_experiment experiment_CIFAR100_Dir0.1_HtFE1_PFL_Client20 --config_file FedGH.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir0.1_HtFE1_PFL_Client20_FedGH.log && \
#nohup python3 "$DFFKE_DIR/main.py" --device_id 0 --group_experiment experiment_CIFAR10_Dir0.1_HtFE1_PFL_Client20 --config_file FedGH.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR10_Dir0.1_HtFE1_PFL_Client20_FedGH.log &
#
#nohup python3 "$DFFKE_DIR/main.py" --device_id 1 --group_experiment experiment_CIFAR100_Dir0.1_HtFE1_PFL_Client20 --config_file FedTGP.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir0.1_HtFE1_PFL_Client20_FedTGP.log && \
#nohup python3 "$DFFKE_DIR/main.py" --device_id 1 --group_experiment experiment_CIFAR10_Dir0.1_HtFE1_PFL_Client20 --config_file FedTGP.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR10_Dir0.1_HtFE1_PFL_Client20_FedTGP.log &
#
#nohup python3 "$DFFKE_DIR/main.py" --device_id 2 --group_experiment experiment_CIFAR100_Dir0.1_HtFE1_PFL_Client20 --config_file DFFKE.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir0.1_HtFE1_PFL_Client20_DFFKE.log &
#nohup python3 "$DFFKE_DIR/main.py" --device_id 3 --group_experiment experiment_CIFAR10_Dir0.1_HtFE1_PFL_Client20 --config_file DFFKE.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR10_Dir0.1_HtFE1_PFL_Client20_DFFKE.log &


#nohup python3 "$DFFKE_DIR/main.py" --device_id 0 --group_experiment experiment_CIFAR100_Dir0.1_HtFE1_PFL --config_file DFFKE_CNN_pathological.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir0.1_HtFE1_PFL_DFFKE_CNN_pathological.log &
#nohup python3 "$DFFKE_DIR/main.py" --device_id 1 --group_experiment experiment_CIFAR10_Dir0.1_HtFE1_PFL --config_file DFFKE_CNN_pathological.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR10_Dir0.1_HtFE1_PFL_DFFKE_CNN_pathological.log &

#nohup python3 "$DFFKE_DIR/main.py" --device_id 2 --group_experiment experiment_CIFAR100_Dir0.1_HtFE1_PFL_Client20 --config_file DFFKE_FedTGP_HtFE8.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir0.1_HtFE1_PFL_Client20_DFFKE_FedTGP_HtFE8.log &
#nohup python3 "$DFFKE_DIR/main.py" --device_id 3 --group_experiment experiment_CIFAR10_Dir0.1_HtFE1_PFL_Client20 --config_file DFFKE_FedTGP_HtFE8.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR10_Dir0.1_HtFE1_PFL_Client20_DFFKE_FedTGP_HtFE8.log &

#nohup python3 "$DFFKE_DIR/main.py" --device_id 2 --group_experiment experiment_CIFAR100_Dir0.1_HtFE1_PFL_Client20 --config_file DFFKE_FedTGP_HtFE8_pathological.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir0.1_HtFE1_PFL_Client20_DFFKE_FedTGP_HtFE8_pathological.log &
#nohup python3 "$DFFKE_DIR/main.py" --device_id 3 --group_experiment experiment_CIFAR10_Dir0.1_HtFE1_PFL_Client20 --config_file DFFKE_FedTGP_HtFE8_pathological.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR10_Dir0.1_HtFE1_PFL_Client20_DFFKE_FedTGP_HtFE8_pathological.log &



#nohup python3 "$DFFKE_DIR/main.py" --device_id 1 --group_experiment experiment_CIFAR100_Dir1.0_HtFE1_PFL --config_file DFFKE_DP_0.5.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/experiment_CIFAR100_Dir1.0_HtFE1_PFL_DFFKE_DP_0.5.log &










