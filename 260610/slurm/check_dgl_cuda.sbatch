#!/bin/bash
#SBATCH --job-name=260610_check_dgl_cuda
#SBATCH --account=PAS1289
#SBATCH --output=/users/PAS1289/jyp531/GSMP-research/260610/runs/ogbn_arxiv_simteg_tape_revgat_gsmp/logs/slurm_check_dgl_cuda_%j.out
#SBATCH --error=/users/PAS1289/jyp531/GSMP-research/260610/runs/ogbn_arxiv_simteg_tape_revgat_gsmp/logs/slurm_check_dgl_cuda_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

set -euo pipefail

module load cuda/12.4.1 2>/dev/null || true
export DGLBACKEND=pytorch

cd /users/PAS1289/jyp531/GSMP-research/260610

../.venv/bin/python - <<'PY'
import torch
import dgl

print("torch", torch.__version__, "torch_cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("dgl", dgl.__version__, dgl.__file__)
g = dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=1)
g = g.to("cuda")
print("dgl graph device", g.device)
print("OK: CUDA-enabled DGL works")
PY
