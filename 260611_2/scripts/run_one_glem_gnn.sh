#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-baseline}"
PROFILE="${PROFILE:-smoke}"

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLEM_DIR="${GLEM_DIR:-$WORKDIR/upstream/GLEM}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-0}"
GPUS="${GPUS:-0}"
GSMP_NORM="${GSMP_NORM:-scale_preserve}"

mkdir -p "$WORKDIR/logs" "$WORKDIR/results/glem_revgat_gsmp" "$WORKDIR/cache/gsmp"
cd "$GLEM_DIR"
export PYTHONPATH="$GLEM_DIR/src:${PYTHONPATH:-}"

if [[ "$PROFILE" == "full" ]]; then
  GNN_EPOCHS="${GNN_EPOCHS:-2000}"
  GNN_EARLY_STOP="${GNN_EARLY_STOP:-300}"
  LM_EPOCHS="${LM_EPOCHS:-3}"
  INF_N_EPOCHS="${INF_N_EPOCHS:-2}"
  LM_EVAL_PATIENCE="${LM_EVAL_PATIENCE:-30460}"
  FULL_EM_RETRAIN="${FULL_EM_RETRAIN:-F}"
  EM_RANGE_DEFAULT=""
else
  GNN_EPOCHS="${GNN_EPOCHS:-3}"
  GNN_EARLY_STOP="${GNN_EARLY_STOP:-0}"
  LM_EPOCHS="${LM_EPOCHS:-1}"
  INF_N_EPOCHS="${INF_N_EPOCHS:-1}"
  LM_EVAL_PATIENCE="${LM_EVAL_PATIENCE:-200}"
  FULL_EM_RETRAIN="${FULL_EM_RETRAIN:-F}"
  EM_RANGE_DEFAULT="0-0"
fi
EM_RANGE="${EM_RANGE:-$EM_RANGE_DEFAULT}"
PROFILE_TAG="$PROFILE"
if [[ "$PROFILE_TAG" == "smoke" ]]; then
  PROFILE_TAG="smk"
fi
GNN_CKPT_HASH="$(printf '%s' "${METHOD}_${PROFILE}_${SEED}_${GNN_EPOCHS}_${GNN_EARLY_STOP}_${GSMP_NORM}" | cksum | awk '{print $1}')"
RAW_GNN_CKPT_TAG="${GNN_CKPT_TAG:-${PROFILE_TAG}_s${SEED}_${GNN_CKPT_HASH}}"
GNN_CKPT_TAG="$(printf '%s' "$RAW_GNN_CKPT_TAG" | tr -cs 'A-Za-z0-9_.-' '_')"
if (( ${#GNN_CKPT_TAG} > 48 )); then
  RAW_GNN_CKPT_HASH="$(printf '%s' "$RAW_GNN_CKPT_TAG" | cksum | awk '{print $1}')"
  GNN_CKPT_TAG="${PROFILE_TAG}_s${SEED}_${RAW_GNN_CKPT_HASH}"
  echo "[WARN] shortened long GNN_CKPT_TAG to $GNN_CKPT_TAG"
fi

COMMON_ARGS=(
  --dataset=arxiv_TA
  --em_order=LM-first
  --gnn_early_stop="$GNN_EARLY_STOP"
  --gnn_epochs="$GNN_EPOCHS"
  --gnn_input_norm=T
  --gnn_label_input=F
  --gnn_pl_ratio=1
  --gnn_pl_weight=0.05
  --gnn_result_root="$WORKDIR/results/glem_revgat_gsmp"
  --gnn_log_every="${LOG_EVERY:-10}"
  --gnn_eval_every="${EVAL_EVERY:-1}"
  --gnn_save_epoch_logs=T
  --inf_n_epochs="$INF_N_EPOCHS"
  --inf_tr_n_nodes="${INF_TR_NODES:-100000}"
  --lm_ce_reduction=mean
  --lm_cla_dropout=0.4
  --lm_epochs="$LM_EPOCHS"
  --lm_eq_batch_size=30
  --lm_eval_patience="$LM_EVAL_PATIENCE"
  --lm_init_ckpt=None
  --lm_label_smoothing_factor=0
  --lm_load_best_model_at_end=T
  --lm_lr=2e-05
  --lm_model=Deberta
  --lm_pl_ratio=1
  --lm_pl_weight=0.8
  --pseudo_temp=0.2
  --gpus="$GPUS"
  --seed="$SEED"
  --freeze_lm_outputs_for_gnn_ablation="${FREEZE_LM_OUTPUTS_FOR_GNN_ABLATION:-T}"
  --full_em_retrain="$FULL_EM_RETRAIN"
  --budget_guard_no_lm_work="${BUDGET_GUARD_NO_LM_WORK:-T}"
)
if [[ -n "$EM_RANGE" ]]; then
  COMMON_ARGS+=(--em_range="$EM_RANGE")
fi

case "$METHOD" in
  baseline)
    METHOD_ARGS=(
      --gnn_model=RevGAT
      --gnn_ckpt=RevGAT_"$GNN_CKPT_TAG"
    )
    ;;
  linear)
    METHOD_ARGS=(
      --gnn_model=LinearRevGAT
      --gnn_ckpt=LinearRevGAT_"$GNN_CKPT_TAG"
      --gnn_use_gsmp=F
      --gnn_gsmp_norm="$GSMP_NORM"
      --gnn_gsmp_cache_dir="$WORKDIR/cache/gsmp"
      --gnn_gsmp_graph_direction=dgl_bidirected_self_loop
      --gnn_linear_aggr=mean
    )
    ;;
  gsmp)
    METHOD_ARGS=(
      --gnn_model=LinearRevGAT
      --gnn_ckpt=LinearRevGATGSMP_"$GNN_CKPT_TAG"
      --gnn_use_gsmp=T
      --gnn_gsmp_norm="$GSMP_NORM"
      --gnn_gsmp_cache_dir="$WORKDIR/cache/gsmp"
      --gnn_gsmp_graph_direction=dgl_bidirected_self_loop
      --gnn_linear_aggr=mean
    )
    ;;
  *)
    echo "Unknown method: $METHOD" >&2
    exit 2
    ;;
esac

echo "[RUN] method=$METHOD profile=$PROFILE seed=$SEED gpus=$GPUS workdir=$WORKDIR"
echo "[RUN] GLEM_DIR=$GLEM_DIR"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[DRY_RUN]'
  for arg in "$PYTHON_BIN" src/models/GLEM/trainGLEM.py "${COMMON_ARGS[@]}" "${METHOD_ARGS[@]}"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  exit 0
fi
"$PYTHON_BIN" src/models/GLEM/trainGLEM.py "${COMMON_ARGS[@]}" "${METHOD_ARGS[@]}"
