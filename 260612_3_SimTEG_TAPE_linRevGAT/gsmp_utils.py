from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch


GSMP_NORMS = ("scale_preserve", "strict_observed")


def compute_year_balanced_edge_weight(
    edge_index: torch.Tensor,
    node_year: torch.Tensor,
    num_nodes: int,
    mode: str = "scale_preserve",
    cache_path: str | Path | None = None,
    force_recompute: bool = False,
    log_prefix: str = "[GSMP]",
) -> torch.Tensor:
    """Compute source-year-balanced weights for edges src -> dst.

    `edge_index[0]` is the message source and `edge_index[1]` is the receiver.
    Counts are grouped by receiver node and source node publication year.
    """
    if mode not in GSMP_NORMS:
        raise ValueError(f"Unknown GSMP mode: {mode}")
    cache = Path(cache_path).expanduser() if cache_path is not None else None
    start = time.time()
    if cache is not None:
        print(f"{log_prefix} cache_path={cache}", flush=True)
        if cache.is_file() and not force_recompute:
            weight = torch.load(cache, map_location="cpu", weights_only=False)
            print(f"{log_prefix} loaded_from_cache=True", flush=True)
            _print_weight_stats(log_prefix, edge_index, node_year, int(num_nodes), weight, mode, time.time() - start)
            return weight.detach().cpu().float().contiguous()

    print(f"{log_prefix} loaded_from_cache=False", flush=True)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    src = edge_index[0].detach().cpu().long()
    dst = edge_index[1].detach().cpu().long()
    years = node_year.detach().cpu().long().view(-1)
    if years.numel() != int(num_nodes):
        raise ValueError(f"node_year length {years.numel()} does not match num_nodes {num_nodes}")

    unique_years, year_id_all_nodes = torch.unique(years, sorted=True, return_inverse=True)
    num_years = int(unique_years.numel())
    edge_year_id = year_id_all_nodes[src]
    key = dst * num_years + edge_year_id

    counts = torch.bincount(key, minlength=int(num_nodes) * num_years).float()
    edge_counts = counts[key].clamp_min(1.0)
    base = 1.0 / edge_counts

    if mode == "strict_observed":
        present = counts.view(int(num_nodes), num_years) > 0
        num_present_years = present.sum(dim=1).float().clamp_min(1.0)
        weight = base / num_present_years[dst]
    else:
        sum_base_per_dst = torch.zeros(int(num_nodes), dtype=torch.float32)
        sum_base_per_dst.scatter_add_(0, dst, base)
        deg = torch.bincount(dst, minlength=int(num_nodes)).float().clamp_min(1.0)
        mean_base = sum_base_per_dst / deg
        weight = base / mean_base[dst].clamp_min(1e-12)

    if not torch.isfinite(weight).all():
        raise FloatingPointError("GSMP edge weights contain non-finite values.")
    weight = weight.float().contiguous()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(weight, cache)
    _print_weight_stats(log_prefix, edge_index, years, int(num_nodes), weight, mode, time.time() - start)
    return weight


def apply_pgsmp_preprocess(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_year: torch.Tensor,
    num_nodes: int,
    alpha: float = 0.5,
    depth: int = 1,
    norm: str = "strict_observed",
    self_mode: str = "neighbor_only",
    cache_path: str | Path | None = None,
    chunk_size: int = 1_000_000,
    force_recompute: bool = False,
) -> torch.Tensor:
    """Apply cached preprocessed GSMP feature smoothing on CPU."""
    if norm not in GSMP_NORMS:
        raise ValueError(f"Unknown P-GSMP norm: {norm}")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError("--pgsmp-alpha must be in [0, 1].")
    if int(depth) < 1:
        raise ValueError("--pgsmp-depth must be >= 1.")
    if int(chunk_size) <= 0:
        raise ValueError("--pgsmp chunk_size must be positive.")

    cache = Path(cache_path).expanduser() if cache_path is not None else None
    start = time.time()
    if cache is not None:
        print(f"[PGSMP] cache_path={cache}", flush=True)
        if cache.is_file() and not force_recompute:
            x_pg = torch.load(cache, map_location="cpu", weights_only=False)
            print("[PGSMP] loaded_from_cache=True", flush=True)
            _print_pgsmp_stats(x, x_pg, alpha, depth, norm, self_mode, chunk_size, time.time() - start)
            return x_pg.detach().cpu().float().contiguous()

    print("[PGSMP] loaded_from_cache=False", flush=True)
    x0 = x.detach().cpu().float().contiguous()
    edge_index_cpu = edge_index.detach().cpu().long().contiguous()
    node_year_cpu = node_year.detach().cpu().long().view(-1)

    if self_mode == "include_self":
        self_nodes = torch.arange(int(num_nodes), dtype=torch.long)
        self_edges = torch.stack([self_nodes, self_nodes], dim=0)
        edge_index_cpu = torch.cat([edge_index_cpu, self_edges], dim=1)
    elif self_mode != "neighbor_only":
        raise ValueError(f"Unknown P-GSMP self_mode: {self_mode}")

    weight = compute_year_balanced_edge_weight(
        edge_index_cpu,
        node_year_cpu,
        int(num_nodes),
        mode=norm,
        cache_path=None,
        log_prefix="[PGSMP]",
    )
    src, dst = edge_index_cpu[0], edge_index_cpu[1]
    h = x0
    for depth_idx in range(int(depth)):
        z = torch.zeros_like(x0)
        for start_idx in range(0, edge_index_cpu.size(1), int(chunk_size)):
            end_idx = min(start_idx + int(chunk_size), edge_index_cpu.size(1))
            s = src[start_idx:end_idx]
            d = dst[start_idx:end_idx]
            w = weight[start_idx:end_idx].view(-1, 1).to(dtype=h.dtype)
            z.index_add_(0, d, h[s] * w)
        h = (1.0 - float(alpha)) * x0 + float(alpha) * z
        if not torch.isfinite(h).all():
            raise FloatingPointError(f"P-GSMP produced non-finite features at depth {depth_idx + 1}.")

    x_pg = h.contiguous()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(x_pg, cache)
    _print_pgsmp_stats(x0, x_pg, alpha, depth, norm, self_mode, chunk_size, time.time() - start)
    return x_pg


def make_cache_name(prefix: str, metadata: dict[str, object], suffix: str = ".pt") -> str:
    serial = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(serial.encode("utf-8")).hexdigest()[:16]
    readable_parts = []
    for key in _READABLE_KEYS:
        if key not in metadata:
            continue
        value = _slug(metadata[key])
        if len(value) > _MAX_CACHE_VALUE_CHARS:
            value_digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
            value = f"{value[:_MAX_CACHE_VALUE_CHARS]}-{value_digest}"
        readable_parts.append(f"{_slug(key)}-{value}")
    readable = "_".join(readable_parts)
    max_readable = _MAX_CACHE_FILENAME_CHARS - len(prefix) - len(suffix) - len(digest) - 2
    if max_readable > 0:
        readable = readable[:max_readable].rstrip("_-.")
    else:
        readable = ""
    return f"{prefix}_{readable}_{digest}{suffix}" if readable else f"{prefix}_{digest}{suffix}"


def fingerprint_path(path: str | Path | None) -> str:
    if path is None:
        return "none"
    resolved = Path(path).expanduser()
    stat = resolved.stat() if resolved.exists() else None
    payload = {
        "path": str(resolved),
        "size": stat.st_size if stat else "missing",
        "mtime_ns": stat.st_mtime_ns if stat else "missing",
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def edge_index_from_src_dst(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return torch.stack([src.detach().cpu().long(), dst.detach().cpu().long()], dim=0).contiguous()


_MAX_CACHE_FILENAME_CHARS = 180
_MAX_CACHE_VALUE_CHARS = 32

_READABLE_KEYS = (
    "dataset",
    "direction",
    "undirected",
    "self_loop",
    "norm",
    "alpha",
    "depth",
    "self_mode",
    "num_edges",
    "num_nodes",
    "xshape",
    "feature",
)


def _slug(value: object) -> str:
    text = str(value).replace("/", "-").replace(" ", "")
    return "".join(ch if ch.isalnum() or ch in "._-x" else "-" for ch in text)


def _print_weight_stats(
    log_prefix: str,
    edge_index: torch.Tensor,
    node_year: torch.Tensor,
    num_nodes: int,
    weight: torch.Tensor,
    mode: str,
    elapsed: float,
) -> None:
    years = node_year.detach().cpu().long().view(-1)
    print(f"{log_prefix} num_nodes={num_nodes}", flush=True)
    print(f"{log_prefix} num_edges={int(edge_index.size(1))}", flush=True)
    print(f"{log_prefix} num_years={int(torch.unique(years).numel())}", flush=True)
    print(f"{log_prefix} norm={mode}", flush=True)
    print(f"{log_prefix} weight_min={float(weight.min().item()) if weight.numel() else 0.0}", flush=True)
    print(f"{log_prefix} weight_max={float(weight.max().item()) if weight.numel() else 0.0}", flush=True)
    print(f"{log_prefix} weight_mean={float(weight.mean().item()) if weight.numel() else 0.0}", flush=True)
    print(f"{log_prefix} weight_std={float(weight.std(unbiased=False).item()) if weight.numel() else 0.0}", flush=True)
    print(f"{log_prefix} preprocessing_time={elapsed:.3f}", flush=True)


def _print_pgsmp_stats(
    x_original: torch.Tensor,
    x_pg: torch.Tensor,
    alpha: float,
    depth: int,
    norm: str,
    self_mode: str,
    chunk_size: int,
    elapsed: float,
) -> None:
    print(f"[PGSMP] x_original_shape={tuple(x_original.shape)}", flush=True)
    print(f"[PGSMP] x_pg_shape={tuple(x_pg.shape)}", flush=True)
    print(f"[PGSMP] alpha={float(alpha)}", flush=True)
    print(f"[PGSMP] depth={int(depth)}", flush=True)
    print(f"[PGSMP] norm={norm}", flush=True)
    print(f"[PGSMP] self_mode={self_mode}", flush=True)
    print(f"[PGSMP] chunk_size={int(chunk_size)}", flush=True)
    print(f"[PGSMP] x_pg_mean={float(x_pg.mean().item()) if x_pg.numel() else 0.0}", flush=True)
    print(f"[PGSMP] x_pg_std={float(x_pg.std(unbiased=False).item()) if x_pg.numel() else 0.0}", flush=True)
    print(f"[PGSMP] preprocessing_time={elapsed:.3f}", flush=True)
