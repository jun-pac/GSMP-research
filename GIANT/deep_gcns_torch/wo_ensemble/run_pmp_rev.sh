
python ./pmp_modify_rev.py --data_root_dir ../dataset \
--seed 2 --n-runs 3 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --backbone rev --group 2 --mode teacher --seed 0\
|& tee modified_pmp_rev/noemb_res_giant-xrt2.log

