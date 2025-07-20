from huggingface_hub import snapshot_download

repo_id = "vermouthdky/X_lminit"
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir="./out",
    local_dir_use_symlinks=False,
    allow_patterns=["ogbn-arxiv/e5-large/main/cached_embs/x_embs.pt"], # for your own use
)
