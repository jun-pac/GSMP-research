python ./linear_heize_ensemble.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--seed 2 --outputname heize_test2 --save-pred --n-runs 3 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --backbone rev --group 2 --mode teacher\
|& tee log_heize/linear_test2_teacher.log

python ./linear_heize_ensemble.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--seed 2 --outputname heize_test2 --save-pred --n-runs 3 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --backbone rev --group 2 --mode student\
|& tee log_heize/linear_test2_student.log

