#!/bin/bash
#SBATCH --mail-user=s_ssaina@live.concordia.ca
#SBATCH --mail-type=ALL

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

module load StdEnv/2020
module load python/3.8.10
module load cuda/11.7
source $HOME/Adapters/PETL_AST/newenv/bin/activate

scp $HOME/projects/def-ravanelm/datasets/UrbanSound8k.tar.gz $SLURM_TMPDIR/  
cd $SLURM_TMPDIR  
mkdir UrbanSound8k && tar -zxf UrbanSound8k.tar.gz -C UrbanSound8k
cd $HOME/Adapters/PETL_AST/

seeds=(10 3768427010 3721728231 2148699938 3169696615)

for seed in "${seeds[@]}"; do
    python main_wandb.py \
        --epochs 10 \
        --learning-rate 0.005 \
        --weight-decay 1e-2 \
        --batch-size 32 \
        --d_state 8 \
        --d_conv 24 \
        --expand 2 \
        --reduction_rate_adapter 64 \
        --data_path "$SLURM_TMPDIR" \
        --adapter_block "S4A" \
	    --adapter_type "Pfeiffer" \
	    --dataset_name "GSC" \
        --seed "$seed" 

    echo "Run completed for seed $seed"
done

echo "All runs completed."
