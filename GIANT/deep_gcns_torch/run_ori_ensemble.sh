python ./main_modify_ensemble.py --data_root_dir ../dataset \
--seed 2 --outputname test1_o --save-pred --n-runs 10 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --backbone rev --group 2 --mode teacher\
|& tee modified_ori/noemb_res_giant-xrt3.log
