
dataset="ogbn-arxiv"
gnn_algo="graph-sage"

if [ ${dataset} == "ogbn-arxiv" ]; then
    RUNS=10
    if [ ${gnn_algo} == "graph-sage" ]; then
        python -u OGB_baselines/${dataset}/gnn_pmp.py \
            --runs ${RUNS} \
            --data_root_dir ./dataset \
            --use_sage \
            --lr 8e-4 \
            --node_emb_path ./proc_data_xrt/${dataset}/X.all.xrt-emb.npy \
            |& tee OGB_baselines/${dataset}/graph-sage.giant-xrt_pmp10.log
    else
        echo "gnn_algo=${gnn_algo} is not yet supported for ogbn-arxiv!"
    fi
else
    echo "dataset=${dataset} is not yet supported!"
fi