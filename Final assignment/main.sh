wandb login

num_processes=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}
num_processes=${num_processes%%(*}
num_processes=${num_processes##*:}

python3 -m accelerate.commands.launch \
    --num_machines 1 \
    --num_processes ${num_processes} \
    train.py \
    --data-dir ./data/cityscapes \
    --batch-size 1 \
    --epochs 50 \
    --lr 0.01 \
    --num-workers 10 \
    --seed 42 \
    --gradient-accumulation-steps 4 \
    --mixed-precision "fp16" \
    --experiment-id "DINOv2_upsample" \
    --decoder "upsample"