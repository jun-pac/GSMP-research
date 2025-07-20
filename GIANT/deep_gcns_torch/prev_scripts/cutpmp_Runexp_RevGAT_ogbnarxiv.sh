python ./examples/ogb_eff/ogbn_arxiv_dgl/pmp_cut_main.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --mode teacher --seed 0\
|& tee pmp_cut/teacher_res_giant-xrt1.log

python ./examples/ogb_eff/ogbn_arxiv_dgl/pmp_cut_main.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --alpha 0.95 --temp 0.7 --mode student --seed 0\
|& tee pmp_cut/student_res_giant-xrt1.log

python ./examples/ogb_eff/ogbn_arxiv_dgl/pmp_cut_main.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --mode teacher --seed 1\
|& tee pmp_cut/teacher_res_giant-xrt2.log

python ./examples/ogb_eff/ogbn_arxiv_dgl/pmp_cut_main.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --alpha 0.95 --temp 0.7 --mode student --seed 1\
|& tee pmp_cut/student_res_giant-xrt2.log


python ./examples/ogb_eff/ogbn_arxiv_dgl/pmp_cut_main.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --mode teacher --seed 2\
|& tee pmp_cut/teacher_res_giant-xrt3.log

python ./examples/ogb_eff/ogbn_arxiv_dgl/pmp_cut_main.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --alpha 0.95 --temp 0.7 --mode student --seed 2\
|& tee pmp_cut/student_res_giant-xrt3.log


python ./examples/ogb_eff/ogbn_arxiv_dgl/pmp_cut_main.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --mode teacher --seed 3\
|& tee pmp_cut/teacher_res_giant-xrt4.log

python ./examples/ogb_eff/ogbn_arxiv_dgl/pmp_cut_main.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --alpha 0.95 --temp 0.7 --mode student --seed 3\
|& tee pmp_cut/student_res_giant-xrt4.log


python ./examples/ogb_eff/ogbn_arxiv_dgl/pmp_cut_main.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--seed 4 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --mode teacher\
|& tee pmp_cut/teacher_res_giant-xrt5.log

python ./examples/ogb_eff/ogbn_arxiv_dgl/pmp_cut_main.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--seed 4 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --alpha 0.95 --temp 0.7 --mode student\
|& tee pmp_cut/student_res_giant-xrt5.log

python ./examples/ogb_eff/ogbn_arxiv_dgl/pmp_cut_main.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--seed 5 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --mode teacher\
|& tee pmp_cut/teacher_res_giant-xrt6.log

python ./examples/ogb_eff/ogbn_arxiv_dgl/pmp_cut_main.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--seed 5 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --alpha 0.95 --temp 0.7 --mode student\
|& tee pmp_cut/student_res_giant-xrt6.log

python ./examples/ogb_eff/ogbn_arxiv_dgl/pmp_cut_main.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--seed 6 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --mode teacher\
|& tee pmp_cut/teacher_res_giant-xrt7.log

python ./examples/ogb_eff/ogbn_arxiv_dgl/pmp_cut_main.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--seed 6 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --alpha 0.95 --temp 0.7 --mode student\
|& tee pmp_cut/student_res_giant-xrt7.log

python ./examples/ogb_eff/ogbn_arxiv_dgl/pmp_cut_main.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--seed 7 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --mode teacher\
|& tee pmp_cut/teacher_res_giant-xrt8.log

python ./examples/ogb_eff/ogbn_arxiv_dgl/pmp_cut_main.py --data_root_dir ../dataset \
--pretrain_path ../proc_data_xrt/ogbn-arxiv/X.all.xrt-emb.npy \
--seed 7 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --alpha 0.95 --temp 0.7 --mode student\
|& tee pmp_cut/student_res_giant-xrt8.log