NUM_RUNS=3
num_hops=6
DATA_ROOT_DIR=./dataset
output_emb_prefix=/fs/ess/PAS1289/GAMLP/ogbn-papers100M-pmp.node-emb
input_emb_path=/fs/ess/PAS1289/gaint-xrt/proc_data_xrt/ogbn-papers100M/X.all.xrt-emb.npy
gpu=0

if [ ! -f "${output_emb_prefix}_${num_hops}.pt" ]; then
    python -u ./data/preprocess_pmp_papers100m.py \
        --root ${DATA_ROOT_DIR} \
        --num_hops ${num_hops} \
        --pretrained_emb_path ${input_emb_path} \
        --output_emb_prefix ${output_emb_prefix} \
        |& tee ./xrt-gamlp.ogbn-papers100M.prepare-pmp.log
fi
python -u main_pmp.py \
    --gpu ${gpu} \
    --num-runs ${NUM_RUNS} \
    --root ${DATA_ROOT_DIR} \
    --dataset ogbn-papers100M \
    --node_emb_path ${output_emb_prefix} \
    --use-rlu \
    --method R_GAMLP_RLU \
    --stages 100 150 150 150 \
    --train-num-epochs 0 0 0 0 \
    --threshold 0 \
    --input-drop 0 \
    --att-drop 0 \
    --label-drop 0 \
    --dropout 0.5 \
    --pre-process \
    --eval 1 \
    --act sigmoid \
    --batch 5000 \
    --patience 300 \
    --n-layers-2 6 \
    --label-num-hops 9 \
    --num-hops 6 \
    --hidden 1024 \
    --bns \
    --temp 0.001 \
    |& tee ./xrt-gamlp.ogbn-papers100M.pmp.log