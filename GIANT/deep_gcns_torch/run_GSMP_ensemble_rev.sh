python ./GSMP_ensemble_rev.py --data_root_dir ../dataset \
--seed 2 --outputname GSMP_rev_test1 --save-pred --n-runs 3 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --backbone rev --group 2 --mode teacher\
|& tee log_GSMP/noemb_rev_teacher_test1.log
