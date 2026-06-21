import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

try:
    import dgl.function as fn
except ModuleNotFoundError:
    fn = None


IMPACT_CACHE_VERSION = "impact_v4_stageaware_gsmp"


@dataclass
class ImpactConfig:
    method: str = "none"
    apply_to: str = "both"
    gsmp_first_layer_only: bool = True
    paper_year: Optional[torch.Tensor] = None
    t_min: Optional[int] = None
    t_max: Optional[int] = None
    edge_weight_cache: Dict[Tuple[str, str, str, str], torch.Tensor] = field(default_factory=dict)


def normalize_method(method: str) -> str:
    if method == "baseline":
        return "none"
    return method


def build_impact_config(args, paper_year: Optional[torch.Tensor]) -> ImpactConfig:
    method = normalize_method(getattr(args, "impact_method", "none"))
    apply_to = getattr(args, "impact_apply_to", "both")
    first_only = bool(getattr(args, "impact_gsmp_first_layer_only", True))

    if method == "none":
        return ImpactConfig(method="none", apply_to=apply_to, gsmp_first_layer_only=first_only)

    if paper_year is None:
        raise ValueError("SMP/GSMP requires ogbn-mag paper year timestamps, but none were found.")

    paper_year = paper_year.detach().cpu().long().view(-1)
    valid = paper_year >= 0
    if not valid.any():
        raise ValueError("SMP/GSMP requires at least one valid paper year timestamp.")

    return ImpactConfig(
        method=method,
        apply_to=apply_to,
        gsmp_first_layer_only=first_only,
        paper_year=paper_year,
        t_min=int(paper_year[valid].min().item()),
        t_max=int(paper_year[valid].max().item()),
    )


def impact_variant_tag(config: ImpactConfig) -> str:
    if config.method == "gsmp":
        first = "first" if config.gsmp_first_layer_only else "all"
        return f"{config.method}-{first}"
    return config.method


def impact_active_for_scope(config: Optional[ImpactConfig], scope: str) -> bool:
    if config is None or config.method == "none":
        return False
    return config.apply_to in ("both", scope)


def should_apply_impact(
    config: Optional[ImpactConfig],
    scope: str,
    hop: int,
    stype: str,
    dtype: str,
) -> bool:
    if not impact_active_for_scope(config, scope):
        return False
    if stype != "P" or dtype != "P":
        return False
    if config.method == "gsmp" and config.gsmp_first_layer_only and hop != 1:
        return False
    return True


def compute_smp_raw_weights(src_time: torch.Tensor, dst_time: torch.Tensor, t_min: int, t_max: int) -> torch.Tensor:
    delta = torch.abs(src_time - dst_time)
    radius = torch.minimum(dst_time - float(t_min), float(t_max) - dst_time)
    single = (delta == 0) | (delta > radius)
    return torch.where(
        single,
        torch.full_like(src_time, 2.0, dtype=torch.float32),
        torch.ones_like(src_time, dtype=torch.float32),
    )


def compute_gsmp_raw_weights(dst: torch.Tensor, src_time: torch.Tensor) -> torch.Tensor:
    if dst.numel() == 0:
        return torch.empty(0, dtype=torch.float32)

    src_time = src_time.long()
    year_min = int(src_time.min().item())
    year_span = int(src_time.max().item() - year_min + 1)
    keys = dst.long() * year_span + (src_time - year_min)
    _, inverse, counts = torch.unique(keys, sorted=False, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse].to(torch.float32)


def _edge_cache_key(etype, method: str, scope: str, hop: int) -> Tuple[str, str, str, str]:
    if isinstance(etype, tuple):
        etype_name = "::".join(etype)
    else:
        etype_name = str(etype)
    return (etype_name, method, scope, str(hop))


def get_raw_edge_weights(g, etype, config: ImpactConfig, scope: str, hop: int) -> torch.Tensor:
    cache_key = _edge_cache_key(etype, config.method, scope, hop)
    cached = config.edge_weight_cache.get(cache_key)
    if cached is not None:
        return cached

    src, dst = g.edges(etype=etype)
    src = src.detach().cpu().long()
    dst = dst.detach().cpu().long()
    paper_year = config.paper_year
    src_time = paper_year[src].to(torch.float32)
    dst_time = paper_year[dst].to(torch.float32)
    valid = (src_time >= 0) & (dst_time >= 0)

    weights = torch.ones(src.numel(), dtype=torch.float32)
    if valid.any():
        if config.method == "smp":
            weights[valid] = compute_smp_raw_weights(
                src_time[valid], dst_time[valid], config.t_min, config.t_max
            )
        elif config.method == "gsmp":
            weights[valid] = compute_gsmp_raw_weights(dst[valid], src_time[valid].long())
        else:
            raise ValueError(f"Unsupported impact method: {config.method}")

    config.edge_weight_cache[cache_key] = weights
    return weights


def weighted_mean_update_all(g, etype, source_key: str, dst_key: str, config: ImpactConfig, scope: str, hop: int):
    if fn is None:
        raise ModuleNotFoundError("DGL is required for weighted HGAMLP propagation.")
    stype, _, dtype = g.to_canonical_etype(etype)
    weights = get_raw_edge_weights(g, etype, config, scope, hop).view(-1, 1)
    device = g.nodes[stype].data[source_key].device
    weights = weights.to(device)

    # SMP/GSMP use target-normalized weighted aggregation. GSMP is intentionally
    # first-layer-only when requested by should_apply_impact; later paper-paper
    # multiplications fall back to the official unweighted HGAMLP propagation.
    tmp_w = "_impact_w"
    tmp_sum = "_impact_sum"
    tmp_den = "_impact_den"
    g.edges[etype].data[tmp_w] = weights
    g[etype].update_all(fn.u_mul_e(source_key, tmp_w, "m"), fn.sum("m", tmp_sum), etype=etype)
    g[etype].update_all(fn.copy_e(tmp_w, "m"), fn.sum("m", tmp_den), etype=etype)
    numer = g.nodes[dtype].data.pop(tmp_sum)
    denom = g.nodes[dtype].data.pop(tmp_den).clamp_min(1e-12)
    g.nodes[dtype].data[dst_key] = numer / denom
    g.edges[etype].data.pop(tmp_w)


def _effective_path_raw_weight(config: ImpactConfig, target_time: torch.Tensor, source_year: int) -> torch.Tensor:
    if config.method == "gsmp":
        return torch.ones_like(target_time, dtype=torch.float32)
    if config.method == "smp":
        source_time = torch.full_like(target_time, float(source_year), dtype=torch.float32)
        return compute_smp_raw_weights(source_time, target_time, config.t_min, config.t_max)
    raise ValueError(f"Unsupported impact method: {config.method}")


def weighted_effective_pxp(g, middle_type: str, source_key: str, out_key: str, config: ImpactConfig):
    if fn is None:
        raise ModuleNotFoundError("DGL is required for weighted HGAMLP propagation.")
    if config.paper_year is None:
        raise ValueError("weighted effective P-X-P propagation needs paper years.")
    if source_key not in g.nodes["P"].data:
        return

    forward_etype = ("P", f"P-{middle_type}", middle_type)
    backward_etype = (middle_type, f"{middle_type}-P", "P")
    feat = g.nodes["P"].data[source_key]
    device = feat.device
    paper_year = config.paper_year.to(device)
    valid_years = torch.unique(paper_year[paper_year >= 0]).detach().cpu().tolist()
    target_time = paper_year.to(torch.float32)

    numerator = torch.zeros_like(feat)
    denom = torch.zeros((feat.shape[0], 1), dtype=torch.float32, device=device)
    tmp_src = "_impact_pxp_src"
    tmp_mid = "_impact_pxp_mid"
    tmp_out = "_impact_pxp_out"

    for year in sorted(int(y) for y in valid_years):
        year_mask = (paper_year == year).to(feat.dtype).view(-1, 1)
        payload = torch.cat([feat * year_mask, year_mask], dim=1)
        g.nodes["P"].data[tmp_src] = payload
        g[forward_etype].update_all(fn.copy_u(tmp_src, "m"), fn.sum("m", tmp_mid), etype=forward_etype)
        g[backward_etype].update_all(fn.copy_u(tmp_mid, "m"), fn.sum("m", tmp_out), etype=backward_etype)

        path_payload = g.nodes["P"].data.pop(tmp_out)
        path_sum = path_payload[:, :-1]
        path_count = path_payload[:, -1:].to(torch.float32)
        raw_weight = _effective_path_raw_weight(config, target_time, year).view(-1, 1).to(device)
        active = path_count > 0

        if config.method == "gsmp":
            numerator += torch.where(active, path_sum / path_count.clamp_min(1e-12), torch.zeros_like(path_sum))
            denom += active.to(torch.float32)
        else:
            numerator += raw_weight * path_sum
            denom += raw_weight * path_count

        g.nodes[middle_type].data.pop(tmp_mid, None)
        g.nodes["P"].data.pop(tmp_src, None)

    g.nodes["P"].data[out_key] = numerator / denom.clamp_min(1e-12)


def apply_effective_pxp_impact(g, config: Optional[ImpactConfig], scope: str, source_key: str = "P"):
    if not impact_active_for_scope(config, scope):
        return
    # PAP/PFP are effective paper-to-paper meta-path operators. The source
    # paper year and target paper year are both known only after composing the
    # P-X-P path, so we compute sparse path sums by source-year bucket rather
    # than reusing old post-hoc scaling or dense paper-paper matrices.
    for middle_type, out_key in (("A", "PAP"), ("F", "PFP")):
        if out_key in g.nodes["P"].data:
            weighted_effective_pxp(g, middle_type, source_key, out_key, config)


def effective_pxp_diag(g, middle_type: str, config: ImpactConfig) -> torch.Tensor:
    if fn is None:
        raise ModuleNotFoundError("DGL is required for weighted HGAMLP propagation.")
    if config.paper_year is None:
        raise ValueError("effective P-X-P diagonal needs paper years.")

    forward_etype = ("P", f"P-{middle_type}", middle_type)
    backward_etype = (middle_type, f"{middle_type}-P", "P")
    num_papers = g.num_nodes("P")
    src, _ = g.edges(etype=forward_etype)
    src = src.detach().cpu().long()
    self_path_count = torch.zeros(num_papers, dtype=torch.float32)
    if src.numel():
        self_path_count.scatter_add_(0, src, torch.ones(src.numel(), dtype=torch.float32))

    paper_year = config.paper_year.detach().cpu().long()
    valid_years = torch.unique(paper_year[paper_year >= 0]).detach().cpu().tolist()
    target_time = paper_year.to(torch.float32)
    denom = torch.zeros(num_papers, dtype=torch.float32)
    diag_num = torch.zeros(num_papers, dtype=torch.float32)

    tmp_src = "_impact_pxp_diag_src"
    tmp_mid = "_impact_pxp_diag_mid"
    tmp_out = "_impact_pxp_diag_out"

    for year in sorted(int(y) for y in valid_years):
        year_mask = (paper_year == year).to(torch.float32).view(-1, 1)
        g.nodes["P"].data[tmp_src] = year_mask
        g[forward_etype].update_all(fn.copy_u(tmp_src, "m"), fn.sum("m", tmp_mid), etype=forward_etype)
        g[backward_etype].update_all(fn.copy_u(tmp_mid, "m"), fn.sum("m", tmp_out), etype=backward_etype)

        path_count = g.nodes["P"].data.pop(tmp_out).view(-1).detach().cpu().to(torch.float32)
        active = path_count > 0
        same_year_target = paper_year == year

        if config.method == "gsmp":
            denom += active.to(torch.float32)
            same_year_active = same_year_target & active
            diag_num[same_year_active] = (
                self_path_count[same_year_active] / path_count[same_year_active].clamp_min(1e-12)
            )
        elif config.method == "smp":
            raw_weight = _effective_path_raw_weight(config, target_time, year).detach().cpu().view(-1)
            denom += raw_weight * path_count
            same_year_active = same_year_target & active
            diag_num[same_year_active] = raw_weight[same_year_active] * self_path_count[same_year_active]
        else:
            raise ValueError(f"Unsupported impact method: {config.method}")

        g.nodes[middle_type].data.pop(tmp_mid, None)
        g.nodes["P"].data.pop(tmp_src, None)

    return diag_num / denom.clamp_min(1e-12)


def ensure_impact_pxp_diag(g, args, config: ImpactConfig, middle_type: str, out_key: str) -> Path:
    emb_path = Path(getattr(args, "emb_path", "./dataset/ogbn_mag/"))
    emb_path.mkdir(parents=True, exist_ok=True)
    variant = impact_variant_tag(config)
    path = emb_path / f"{args.dataset}_{variant}_{IMPACT_CACHE_VERSION}_{out_key}_diag.pt"
    if path.exists():
        return path

    diag = effective_pxp_diag(g, middle_type, config)
    torch.save(diag, path)
    return path


def row_normalized_pp_edges(g, method: str, config: Optional[ImpactConfig]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    etype = ("P", "P-P", "P")
    src, dst = g.edges(etype=etype)
    src = src.detach().cpu().long()
    dst = dst.detach().cpu().long()
    if method == "none":
        raw = torch.ones(src.numel(), dtype=torch.float32)
    else:
        local = ImpactConfig(
            method=method,
            apply_to="label",
            gsmp_first_layer_only=False,
            paper_year=config.paper_year,
            t_min=config.t_min,
            t_max=config.t_max,
        )
        raw = get_raw_edge_weights(g, etype, local, "label", 1)

    denom = torch.zeros(g.num_nodes("P"), dtype=torch.float32)
    denom.scatter_add_(0, dst, raw)
    norm = raw / denom[dst].clamp_min(1e-12)
    return dst, src, norm


def diag_of_sparse_product(
    left_row: torch.Tensor,
    left_col: torch.Tensor,
    left_val: torch.Tensor,
    right_row: torch.Tensor,
    right_col: torch.Tensor,
    right_val: torch.Tensor,
    n: int,
) -> torch.Tensor:
    left_key = left_col * n + left_row
    right_key = right_row * n + right_col
    order = torch.argsort(right_key)
    sorted_key = right_key[order]
    sorted_val = right_val[order]
    diag = torch.zeros(n, dtype=torch.float32)
    if sorted_key.numel() == 0:
        return diag
    pos = torch.searchsorted(sorted_key, left_key)
    safe_pos = pos.clamp_max(sorted_key.numel() - 1)
    match = (pos < sorted_key.numel()) & (sorted_key[safe_pos] == left_key)
    if match.any():
        diag.scatter_add_(0, left_row[match], left_val[match] * sorted_val[safe_pos[match]])
    return diag


def ensure_impact_ppp_diag(g, args, config: ImpactConfig) -> Path:
    emb_path = Path(getattr(args, "emb_path", "./dataset/ogbn_mag/"))
    emb_path.mkdir(parents=True, exist_ok=True)
    variant = impact_variant_tag(config)
    path = emb_path / f"{args.dataset}_{variant}_{IMPACT_CACHE_VERSION}_PPP_diag.pt"
    if path.exists():
        return path

    if config.method == "gsmp" and config.gsmp_first_layer_only:
        first_method = "gsmp"
        second_method = "none"
    else:
        first_method = config.method
        second_method = config.method

    right_row, right_col, right_val = row_normalized_pp_edges(g, first_method, config)
    left_row, left_col, left_val = row_normalized_pp_edges(g, second_method, config)
    diag = diag_of_sparse_product(
        left_row, left_col, left_val, right_row, right_col, right_val, g.num_nodes("P")
    )
    torch.save(diag, path)
    return path


def propagation_cache_path(args, scope: str, num_hops: int, max_hops: int, extra_metapath) -> Optional[Path]:
    if not getattr(args, "cache_propagation", True):
        return None
    method = normalize_method(getattr(args, "impact_method", "none"))
    apply_to = getattr(args, "impact_apply_to", "both")
    if method != "none" and apply_to not in ("both", scope):
        method = "none"
    cache_apply_to = apply_to
    if scope == "feature":
        # Feature tensors are independent of whether the same run also applies
        # impact to label propagation. Reuse the existing priority1 feature cache
        # for feature-only priority0 instead of rebuilding the 8+ GB tensor.
        cache_apply_to = "both"
    first = "first" if getattr(args, "impact_gsmp_first_layer_only", True) else "all"
    extras = "none" if not extra_metapath else "-".join(sorted(extra_metapath))
    version = "official" if method == "none" else IMPACT_CACHE_VERSION
    cache_dir = Path(getattr(args, "impact_cache_dir", "./impact_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (
        f"{args.dataset}_{scope}_method-{method}_version-{version}_apply-{cache_apply_to}_gsmp-{first}"
        f"_h{num_hops}_max{max_hops}_extra-{extras}.pt"
    )


def _create_progress_header_once(path: Path) -> None:
    header = (
        "timestamp\tjob_id\tmethod\tseed\tstage\tepoch\ttrain_acc\tval_acc\ttest_acc\t"
        "best_epoch\tbest_val\tbest_test_at_best_val\telapsed_sec\n"
    )
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return
    with os.fdopen(fd, "w") as f:
        f.write(header)


def append_live_progress(
    progress_file: str,
    method: str,
    seed: int,
    stage: int,
    epoch: int,
    train_acc: float,
    val_acc: float,
    test_acc: float,
    best_epoch: int,
    best_val: float,
    best_test_at_best_val: float,
    elapsed_sec: float,
) -> None:
    path = Path(progress_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    _create_progress_header_once(path)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    line = (
        f"{timestamp}\t{job_id}\t{method}\t{seed}\t{stage}\t{epoch}\t"
        f"{train_acc:.8f}\t{val_acc:.8f}\t{test_acc:.8f}\t{best_epoch}\t"
        f"{best_val:.8f}\t{best_test_at_best_val:.8f}\t{elapsed_sec:.2f}\n"
    )
    with open(path, "a", buffering=1) as f:
        f.write(line)


def format_result_line(
    method: str,
    seed: int,
    stage: int,
    epoch: int,
    train_acc: float,
    val_acc: float,
    test_acc: float,
    best_epoch: int,
    best_val: float,
    best_test_at_best_val: float,
    elapsed_sec: float,
) -> str:
    return (
        f"RESULT method={method} seed={seed} stage={stage} epoch={epoch} "
        f"train_acc={train_acc:.8f} val_acc={val_acc:.8f} test_acc={test_acc:.8f} "
        f"best_epoch={best_epoch} best_val={best_val:.8f} "
        f"best_test_at_best_val={best_test_at_best_val:.8f} elapsed_sec={elapsed_sec:.2f}"
    )


def run_debug_impact_toy_tests() -> None:
    src_time = torch.tensor([2018, 2017, 2019, 2016], dtype=torch.float32)
    dst_time = torch.tensor([2018, 2018, 2018, 2018], dtype=torch.float32)
    smp = compute_smp_raw_weights(src_time, dst_time, t_min=2017, t_max=2019)
    expected_smp = torch.tensor([2.0, 1.0, 1.0, 2.0])
    assert torch.allclose(smp, expected_smp), (smp, expected_smp)

    dst = torch.zeros(6, dtype=torch.long)
    years = torch.tensor([2017, 2017, 2018, 2019, 2019, 2019])
    gsmp = compute_gsmp_raw_weights(dst, years)
    expected_gsmp = torch.tensor([0.5, 0.5, 1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    assert torch.allclose(gsmp, expected_gsmp, atol=1e-7), (gsmp, expected_gsmp)

    x = torch.tensor([[1.0], [3.0], [10.0], [30.0], [60.0], [90.0]])
    agg = (gsmp.view(-1, 1) * x).sum(dim=0) / gsmp.sum()
    expected_agg = torch.tensor([24.0])
    assert torch.allclose(agg, expected_agg, atol=1e-6), (agg, expected_agg)

    none_config = ImpactConfig(method="none")
    assert not should_apply_impact(none_config, "feature", 1, "P", "P")
    print("impact toy tests passed", flush=True)


if __name__ == "__main__":
    run_debug_impact_toy_tests()
