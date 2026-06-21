from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ALL_MODELS = ("sgc", "gcn", "graphsage", "linearrevgat")
ALL_METHODS = ("baseline", "gsmp")


def parse_csv(text: str, allowed: Optional[Sequence[str]] = None) -> List[str]:
    values = [part.strip().lower() for part in text.split(",") if part.strip()]
    if len(values) == 1 and values[0] == "all":
        if allowed is None:
            raise ValueError("'all' requires an allowed list.")
        return list(allowed)
    if allowed is not None:
        bad = [value for value in values if value not in allowed]
        if bad:
            raise ValueError(f"Unknown values {bad}; allowed values are {list(allowed)}")
    return values


def parse_seed_list(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: Path) -> Dict[str, object]:
    data = torch.load(path, map_location="cpu")
    required = ["x", "y", "node_time", "edge_index", "train_mask", "val_mask", "test_mask"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Dataset {path} is missing keys: {missing}")
    return data


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    nodes = torch.arange(num_nodes, dtype=torch.long)
    loops = torch.stack([nodes, nodes], dim=0)
    return torch.cat([edge_index.long(), loops], dim=1).contiguous()


def compute_gsmp_edge_weight(
    edge_index: torch.Tensor,
    node_time: torch.Tensor,
    num_nodes: int,
    mode: str = "scale_preserve",
) -> torch.Tensor:
    """Target-side source-timestamp balancing for directed src -> dst edges.

    For every target node v, edges are grouped by source timestamp time(u).
    The scale-preserving multiplier gives each observed source timestamp equal
    total mass while keeping the ordinary mean-aggregation scale.
    """
    if mode not in {"scale_preserve", "strict_observed"}:
        raise ValueError("--gsmp-mode must be scale_preserve or strict_observed.")
    src, dst = edge_index.detach().cpu().long()
    times = node_time.detach().cpu().long().view(-1)
    _, time_id = torch.unique(times, sorted=True, return_inverse=True)
    num_times = int(time_id.max().item()) + 1
    src_time = time_id[src]
    key = dst * num_times + src_time
    counts = torch.bincount(key, minlength=int(num_nodes) * num_times).float()
    per_edge_count = counts[key].clamp_min(1.0)
    base = 1.0 / per_edge_count

    if mode == "strict_observed":
        observed = counts.view(int(num_nodes), num_times) > 0
        observed_count = observed.sum(dim=1).float().clamp_min(1.0)
        weight = base / observed_count[dst]
    else:
        sum_base = torch.zeros(int(num_nodes), dtype=torch.float32)
        sum_base.scatter_add_(0, dst, base)
        deg = torch.bincount(dst, minlength=int(num_nodes)).float().clamp_min(1.0)
        mean_base = sum_base / deg
        weight = base / mean_base[dst].clamp_min(1e-12)

    if not torch.isfinite(weight).all():
        raise FloatingPointError("GSMP weights contain non-finite values.")
    return weight.float().contiguous()


def aggregate_mean(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    src, dst = edge_index
    out = x.new_zeros((num_nodes, x.size(1)))
    if edge_weight is None:
        out.index_add_(0, dst, x[src])
    else:
        out.index_add_(0, dst, x[src] * edge_weight.view(-1, 1).to(dtype=x.dtype, device=x.device))
    deg = torch.bincount(dst, minlength=num_nodes).to(device=x.device, dtype=x.dtype).clamp_min(1.0)
    return out / deg.view(-1, 1)


def edge_weight_for_layer(
    edge_weight: Optional[torch.Tensor],
    method: str,
    gsmp_scope: str,
    layer_idx: int,
) -> Optional[torch.Tensor]:
    if method != "gsmp":
        return None
    if gsmp_scope == "first" and layer_idx > 0:
        return None
    return edge_weight


class SGC(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(x))


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: Optional[torch.Tensor],
        method: str,
        gsmp_scope: str,
    ) -> torch.Tensor:
        h = aggregate_mean(x, edge_index, num_nodes, edge_weight_for_layer(edge_weight, method, gsmp_scope, 0))
        h = F.relu(self.lin1(h))
        h = self.dropout(h)
        h = aggregate_mean(h, edge_index, num_nodes, edge_weight_for_layer(edge_weight, method, gsmp_scope, 1))
        return self.lin2(h)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.self1 = nn.Linear(in_dim, hidden_dim)
        self.neigh1 = nn.Linear(in_dim, hidden_dim)
        self.self2 = nn.Linear(hidden_dim, out_dim)
        self.neigh2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: Optional[torch.Tensor],
        method: str,
        gsmp_scope: str,
    ) -> torch.Tensor:
        neigh = aggregate_mean(x, edge_index, num_nodes, edge_weight_for_layer(edge_weight, method, gsmp_scope, 0))
        h = F.relu(self.self1(x) + self.neigh1(neigh))
        h = self.dropout(h)
        neigh = aggregate_mean(h, edge_index, num_nodes, edge_weight_for_layer(edge_weight, method, gsmp_scope, 1))
        return self.self2(h) + self.neigh2(neigh)


class LinearRevGATBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, residual: bool):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_dim, out_dim, bias=False) if residual or in_dim != out_dim else None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = aggregate_mean(x, edge_index, num_nodes, edge_weight)
        h = self.fc(h)
        if self.residual is None:
            h = h + x
        else:
            h = h + self.residual(x)
        h = self.norm(h)
        return self.dropout(F.relu(h))


class LinearRevGAT(nn.Module):
    """RevGAT-style shell with fixed linear mean aggregation instead of attention."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float, num_layers: int):
        super().__init__()
        if num_layers < 2:
            raise ValueError("linearrevgat requires at least two layers.")
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        self.blocks = nn.ModuleList(
            [LinearRevGATBlock(dims[i], dims[i + 1], dropout, residual=True) for i in range(num_layers - 1)]
        )
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: Optional[torch.Tensor],
        method: str,
        gsmp_scope: str,
    ) -> torch.Tensor:
        h = x
        for layer_idx, block in enumerate(self.blocks):
            h = block(h, edge_index, num_nodes, edge_weight_for_layer(edge_weight, method, gsmp_scope, layer_idx))
        return self.classifier(h)


def propagate_sgc(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor],
    method: str,
    gsmp_scope: str,
    k: int,
) -> torch.Tensor:
    h = x
    for layer_idx in range(int(k)):
        h = aggregate_mean(h, edge_index, num_nodes, edge_weight_for_layer(edge_weight, method, gsmp_scope, layer_idx))
    return h


def build_model(model_name: str, in_dim: int, hidden_dim: int, out_dim: int, dropout: float, num_layers: int) -> nn.Module:
    if model_name == "sgc":
        return SGC(in_dim, out_dim, dropout)
    if model_name == "gcn":
        return GCN(in_dim, hidden_dim, out_dim, dropout)
    if model_name == "graphsage":
        return GraphSAGE(in_dim, hidden_dim, out_dim, dropout)
    if model_name == "linearrevgat":
        return LinearRevGAT(in_dim, hidden_dim, out_dim, dropout, num_layers)
    raise ValueError(f"Unknown model {model_name}")


def forward_model(
    model: nn.Module,
    model_name: str,
    x: torch.Tensor,
    sgc_x: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor],
    method: str,
    gsmp_scope: str,
) -> torch.Tensor:
    if model_name == "sgc":
        if sgc_x is None:
            raise ValueError("SGC requires pre-propagated features.")
        return model(sgc_x)
    return model(x, edge_index, num_nodes, edge_weight, method, gsmp_scope)


def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) == 0:
        return 0.0
    pred = logits[mask].argmax(dim=1)
    return float((pred == labels[mask]).float().mean().item())


def train_one(
    args: argparse.Namespace,
    model_name: str,
    method: str,
    seed: int,
    tensors: Dict[str, torch.Tensor],
    num_classes: int,
) -> Dict[str, object]:
    set_seed(seed)
    device = torch.device(args.device)
    x = tensors["x"].to(device)
    y = tensors["y"].to(device)
    edge_index = tensors["edge_index"].to(device)
    edge_weight = tensors["gsmp_weight"].to(device) if method == "gsmp" else None
    train_mask = tensors["train_mask"].to(device)
    val_mask = tensors["val_mask"].to(device)
    test_mask = tensors["test_mask"].to(device)
    num_nodes = int(x.size(0))

    sgc_x = None
    if model_name == "sgc":
        with torch.no_grad():
            sgc_x = propagate_sgc(
                x,
                edge_index,
                num_nodes,
                edge_weight,
                method,
                args.gsmp_scope,
                args.sgc_k,
            ).detach()

    model = build_model(model_name, x.size(1), args.hidden_dim, num_classes, args.dropout, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best = {
        "best_val": -1.0,
        "test_at_best_val": -1.0,
        "best_epoch": -1,
        "final_val": -1.0,
        "final_test": -1.0,
    }
    stale = 0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = forward_model(
            model,
            model_name,
            x,
            sgc_x,
            edge_index,
            num_nodes,
            edge_weight,
            method,
            args.gsmp_scope,
        )
        loss = loss_fn(logits[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                logits = forward_model(
                    model,
                    model_name,
                    x,
                    sgc_x,
                    edge_index,
                    num_nodes,
                    edge_weight,
                    method,
                    args.gsmp_scope,
                )
                val_acc = accuracy(logits, y, val_mask)
                test_acc = accuracy(logits, y, test_mask)
            best["final_val"] = val_acc
            best["final_test"] = test_acc
            if val_acc > best["best_val"]:
                best["best_val"] = val_acc
                best["test_at_best_val"] = test_acc
                best["best_epoch"] = epoch
                stale = 0
            else:
                stale += args.eval_every
            if args.patience > 0 and stale >= args.patience:
                break

    best.update(
        {
            "model": model_name,
            "method": method,
            "seed": int(seed),
            "runtime_sec": float(time.time() - start_time),
        }
    )
    return best


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary = []
    for model in sorted({str(row["model"]) for row in rows}, key=list(ALL_MODELS).index):
        by_method = {method: [] for method in ALL_METHODS}
        for row in rows:
            if row["model"] == model:
                by_method[str(row["method"])].append(float(row["test_at_best_val"]))
        if not by_method["baseline"] or not by_method["gsmp"]:
            continue
        baseline = torch.tensor(by_method["baseline"], dtype=torch.float32)
        gsmp = torch.tensor(by_method["gsmp"], dtype=torch.float32)
        delta = gsmp - baseline
        summary.append(
            {
                "model": model,
                "baseline_mean": float(baseline.mean().item()),
                "baseline_std": float(baseline.std(unbiased=False).item()) if baseline.numel() > 1 else 0.0,
                "gsmp_mean": float(gsmp.mean().item()),
                "gsmp_std": float(gsmp.std(unbiased=False).item()) if gsmp.numel() > 1 else 0.0,
                "delta_mean": float(delta.mean().item()),
                "delta_min": float(delta.min().item()),
                "relative_delta_pct": float(100.0 * delta.mean().item() / max(float(baseline.mean().item()), 1e-12)),
                "num_seeds": int(min(baseline.numel(), gsmp.numel())),
            }
        )
    return summary


def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model",
        "method",
        "seed",
        "best_val",
        "test_at_best_val",
        "best_epoch",
        "final_val",
        "final_test",
        "runtime_sec",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row[col]) for col in columns) + "\n")


def append_progress(
    progress_file: Path,
    args: argparse.Namespace,
    data: Dict[str, object],
    rows: List[Dict[str, object]],
    summary: List[Dict[str, object]],
    result_json: Path,
    result_tsv: Path,
) -> None:
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    stats = data.get("stats", {})
    config = data.get("config", {})
    with progress_file.open("a", encoding="utf-8") as f:
        f.write(f"\n## Run {timestamp}\n\n")
        f.write(f"- dataset: `{args.data}`\n")
        f.write(f"- result_json: `{result_json}`\n")
        f.write(f"- result_tsv: `{result_tsv}`\n")
        f.write(
            "- graph setting: "
            f"nodes={stats.get('num_nodes')}, edges={stats.get('num_edges')}, "
            f"classes={stats.get('num_classes')}, times={stats.get('num_times')}, "
            f"same_p0={config.get('same_p0')}, cross_p0={config.get('cross_p0')}, "
            f"same_gamma={config.get('same_gamma')}, cross_gamma={config.get('cross_gamma')}, "
            f"feature_scale={config.get('feature_scale')}, "
            f"feature_noise=[{config.get('feature_noise_min')}, {config.get('feature_noise_max')}]\n"
        )
        f.write(
            "- split: "
            f"train_times={config.get('train_times_resolved')}, "
            f"val_times={config.get('val_times_resolved')}, "
            f"test_times={config.get('test_times_resolved')}\n"
        )
        f.write(
            "- train setting: "
            f"models={args.models}, seeds={args.seeds}, epochs={args.epochs}, "
            f"hidden_dim={args.hidden_dim}, lr={args.lr}, dropout={args.dropout}, "
            f"self_loop={args.self_loop}, gsmp_scope={args.gsmp_scope}, gsmp_mode={args.gsmp_mode}\n\n"
        )
        f.write("| model | baseline test | GSMP test | delta | relative delta |\n")
        f.write("| --- | ---: | ---: | ---: | ---: |\n")
        for item in summary:
            f.write(
                f"| {item['model']} | "
                f"{item['baseline_mean']:.4f} +/- {item['baseline_std']:.4f} | "
                f"{item['gsmp_mean']:.4f} +/- {item['gsmp_std']:.4f} | "
                f"{item['delta_mean']:+.4f} | "
                f"{item['relative_delta_pct']:+.2f}% |\n"
            )
        f.write("\nPer-seed rows are in the TSV above.\n")


def print_summary(summary: List[Dict[str, object]]) -> None:
    print("\nSummary: test accuracy at best validation epoch")
    print("| model | baseline | GSMP | delta | relative delta |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for item in summary:
        print(
            f"| {item['model']} | "
            f"{item['baseline_mean']:.4f} +/- {item['baseline_std']:.4f} | "
            f"{item['gsmp_mean']:.4f} +/- {item['gsmp_std']:.4f} | "
            f"{item['delta_mean']:+.4f} | "
            f"{item['relative_delta_pct']:+.2f}% |"
        )


def prepare_tensors(data: Dict[str, object], args: argparse.Namespace) -> Tuple[Dict[str, torch.Tensor], int]:
    x = data["x"].float()
    y = data["y"].long()
    node_time = data["node_time"].long()
    edge_index = data["edge_index"].long()
    num_nodes = int(x.size(0))
    if args.self_loop:
        edge_index = add_self_loops(edge_index, num_nodes)
    gsmp_weight = compute_gsmp_edge_weight(edge_index, node_time, num_nodes, mode=args.gsmp_mode)
    tensors = {
        "x": x,
        "y": y,
        "node_time": node_time,
        "edge_index": edge_index,
        "gsmp_weight": gsmp_weight,
        "train_mask": data["train_mask"].bool(),
        "val_mask": data["val_mask"].bool(),
        "test_mask": data["test_mask"].bool(),
    }
    return tensors, int(y.max().item()) + 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline and GSMP models on a generated TSBM graph.")
    parser.add_argument("--data", type=Path, default=Path("./data_tsbm/tsbm_extreme.pt"))
    parser.add_argument("--models", type=str, default="all")
    parser.add_argument("--methods", type=str, default="baseline,gsmp")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--sgc-k", type=int, default=2)
    parser.add_argument("--self-loop", action="store_true")
    parser.add_argument("--gsmp-scope", choices=["all", "first"], default="all")
    parser.add_argument("--gsmp-mode", choices=["scale_preserve", "strict_observed"], default="scale_preserve")
    parser.add_argument("--results-dir", type=Path, default=Path("./results_tsbm"))
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--progress-file", type=Path, default=Path("./TSBM_progress.md"))
    parser.add_argument("--append-progress", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    models = parse_csv(args.models, ALL_MODELS)
    methods = parse_csv(args.methods, ALL_METHODS)
    seeds = parse_seed_list(args.seeds)
    data = load_dataset(args.data)
    tensors, num_classes = prepare_tensors(data, args)

    print(f"Loaded {args.data}")
    print(f"nodes={tensors['x'].size(0)} edges={tensors['edge_index'].size(1)} classes={num_classes}")
    print(f"models={models} methods={methods} seeds={seeds} device={args.device}")
    print(
        "gsmp_weight "
        f"min={float(tensors['gsmp_weight'].min().item()):.4f} "
        f"max={float(tensors['gsmp_weight'].max().item()):.4f} "
        f"mean={float(tensors['gsmp_weight'].mean().item()):.4f}"
    )

    rows: List[Dict[str, object]] = []
    for model_name in models:
        for seed in seeds:
            for method in methods:
                run_seed = int(seed)
                print(f"[RUN] model={model_name} method={method} seed={run_seed}", flush=True)
                result = train_one(args, model_name, method, run_seed, tensors, num_classes)
                rows.append(result)
                print(
                    f"[DONE] model={model_name} method={method} seed={run_seed} "
                    f"best_val={result['best_val']:.4f} test={result['test_at_best_val']:.4f} "
                    f"epoch={result['best_epoch']} runtime={result['runtime_sec']:.1f}s",
                    flush=True,
                )

    summary = summarize(rows)
    print_summary(summary)

    tag = args.tag.strip() or time.strftime("%Y%m%d_%H%M%S")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_json = args.results_dir / f"{tag}_results.json"
    result_tsv = args.results_dir / f"{tag}_rows.tsv"
    payload = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "data_config": data.get("config", {}),
        "data_stats": data.get("stats", {}),
        "rows": rows,
        "summary": summary,
    }
    with result_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    write_tsv(result_tsv, rows)
    if args.append_progress:
        append_progress(args.progress_file, args, data, rows, summary, result_json, result_tsv)
    print(f"\nWrote {result_json}")
    print(f"Wrote {result_tsv}")


if __name__ == "__main__":
    main()
