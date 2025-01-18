#!/bin/bash
#SBATCH --mail-user=s_ssaina@live.concordia.ca
#SBATCH --mail-type=ALL

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

module load StdEnv/2020
module load python/3.8.10
module load cuda/11.7
source $HOME/Adapters/PETL_AST/newenv/bin/activate

cd $HOME/Adapters/PETL_AST/

echo 'Noncaus Pfeiffer'
python num_params.py \
        --epochs 10 \
        --learning-rate 0.004 \
        --weight-decay 1e-2 \
        --batch-size 32 \
        --d_state 8 \
        --d_conv 24 \
        --expand 2 \
        --reduction_rate_adapter 64 \
        --data_path "$SLURM_TMPDIR" \
        --adapter_block "shared_scaled_conv_nocaus_mamba" \
        --adapter_type "Pfeiffer" \
        --dataset_name "FSC" \
	    --seed 10 \

echo 'Noncaus Houlsby'
python num_params.py \
        --epochs 10 \
        --learning-rate 0.004 \
        --weight-decay 1e-2 \
        --batch-size 32 \
        --d_state 8 \
        --d_conv 24 \
        --expand 2 \
        --reduction_rate_adapter 64 \
        --data_path "$SLURM_TMPDIR" \
        --adapter_block "shared_scaled_conv_nocaus_mamba" \
        --adapter_type "Houlsby" \
        --dataset_name "FSC" \
	    --seed 10 \

echo 'Bottleneck Pfeiffer'
python num_params.py \
        --epochs 10 \
        --learning-rate 0.003 \
        --weight-decay 1e-2 \
        --batch-size 32 \
        --d_state 4 \
        --d_conv 2 \
        --expand 2 \
        --reduction_rate_adapter 48 \
        --data_path "$SLURM_TMPDIR" \
        --adapter_block "bottleneck" \
        --adapter_type "Pfeiffer" \
        --dataset_name "FSC" \
	    --seed 10 \

echo 'Bottleneck Houlsby'
python num_params.py \
        --epochs 10 \
        --learning-rate 0.003 \
        --weight-decay 1e-2 \
        --batch-size 32 \
        --d_state 4 \
        --d_conv 2 \
        --expand 2 \
        --reduction_rate_adapter 48 \
        --data_path "$SLURM_TMPDIR" \
        --adapter_block "bottleneck" \
        --adapter_type "Houlsby" \
        --dataset_name "FSC" \
	    --seed 10 \
