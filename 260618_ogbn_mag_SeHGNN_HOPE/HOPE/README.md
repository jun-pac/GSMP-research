# HOPE: SeHGNN-HOPE ogbn-mag GSMP workspace

This is the local HOPE code used by `../README.md` for the SeHGNN-HOPE baseline and the GSMP paper-stack comparison.

## Requirements
### 1.Neural network libraries for GNNs

Python 3.8.20

dgl-cu117==0.9.1.post1

ogb==1.3.6

torch==1.13.0+cu117

torch_sparse==0.6.18

### 2.Other dependencies
sparse_tools is a dependency provided by [SeHGNN](https://github.com/ICT-GIMLab/SeHGNN/tree/master), please install it following their provided steps, or directly use the intermediate results we have already generated([download link](https://drive.google.com/drive/folders/1s6uAm9mPX4SNenVYx_Qi9bBPzbCdcSv9?usp=sharing)).


# Data preparation
The pre-trained embeddings we use are the same as those in [HGAMLP](https://github.com/GooLiang/HGAMLP_MAG?tab=readme-ov-file#data-preparation), which utilizes the Line method from RpHGNN. To reproduce the results on the OGB Leaderboards (ogbn-mag), Please download the pre-trained embeddings mag.p directly from their Google Drive [link](https://drive.google.com/file/d/1Q7gD1xpmLeFJu5xWWY3nwa46cM8xYClH/view?usp=sharing).

# Run SeHGNN-HOPE for OGB Leaderboards (ogbn-mag)
```
python -u training.py --aggregation SeHGNN-HOPE --label-residual --similarity-threshold 0.6 --lower-bound 0.5 --upper-bound 3 --seeds 1 2 3 4 5 6 7 8 9 10 --use-sparse-tools
```

Note that the parameter use-sparse-tools is used to control whether to use the sparse_tools dependency.

For the complete baseline/proposed commands, Slurm entrypoints, and GSMP scope, see `../README.md`.
