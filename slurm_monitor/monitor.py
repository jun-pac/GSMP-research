#!/usr/bin/env python3
"""Poll Slurm accounting data and write a dashboard-friendly state file."""

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STATE = BASE_DIR / "state" / "jobs.json"
DEFAULT_HISTORY = BASE_DIR / "state" / "snapshots.jsonl"
DEFAULT_CONFIG = BASE_DIR / "config.json"

SACCT_FIELDS = [
    "JobIDRaw",
    "JobID",
    "JobName%120",
    "User%40",
    "State%40",
    "Partition%60",
    "Account%60",
    "AllocTRES%200",
    "ReqTRES%200",
    "ReqMem%40",
    "ElapsedRaw",
    "Elapsed",
    "TimelimitRaw",
    "Submit",
    "Start",
    "End",
    "NodeList%120",
    "NNodes",
    "NCPUS",
]

SACCT_NAMES = [field.split("%", 1)[0] for field in SACCT_FIELDS]

ACTIVE_STATES = {
    "BOOT_FAIL",
    "COMPLETING",
    "CONFIGURING",
    "PENDING",
    "PREEMPTED",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "SIGNALING",
    "SUSPENDED",
    "STOPPED",
}

FINISHED_STATES = {
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}

MEMORY_UNITS = {
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
    "T": 1024**4,
    "P": 1024**5,
}

DEFAULT_CONFIG_DATA = {
    "cost_label": "cost units",
    "rates": {
        "gpu_hour": 1.0,
        "cpu_hour": 0.0,
        "memory_gb_hour": 0.0,
        "billing_hour": 0.0,
    },
    "partition_rates": {},
    "gpu_type_rates": {},
}


def now_local() -> datetime:
    return datetime.now().astimezone()


def isoformat(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat(timespec="seconds")


def run_command(command: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return 124, stdout, stderr or "command timed out"
    return completed.returncode, completed.stdout, completed.stderr


def split_state(state: str) -> str:
    state = (state or "").strip().upper()
    if not state:
        return "UNKNOWN"
    state = state.split()[0]
    state = state.split("+", 1)[0]
    return state


def state_group(state: str) -> str:
    normalized = split_state(state)
    if normalized == "RUNNING":
        return "running"
    if normalized == "PENDING":
        return "pending"
    if normalized in FINISHED_STATES:
        return "finished"
    if normalized in ACTIVE_STATES:
        return "active"
    return "unknown"


def parse_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if not text or text in {"Unknown", "N/A", "None", "(null)"}:
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text or text in {"Unknown", "N/A", "None", "(null)"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def parse_slurm_datetime(value: str) -> Optional[datetime]:
    text = (value or "").strip()
    if not text or text in {"Unknown", "N/A", "None", "(null)"}:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=now_local().tzinfo)
        except ValueError:
            pass
    return None


def parse_slurm_duration(value: str) -> Optional[int]:
    text = (value or "").strip()
    if not text or text in {"Unknown", "N/A", "None", "(null)", "UNLIMITED", "NOT_SET"}:
        return None
    if text.isdigit():
        return int(text)

    days = 0
    if "-" in text:
        day_part, text = text.split("-", 1)
        days = parse_int(day_part)

    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = [int(part) for part in parts]
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = [int(part) for part in parts]
        elif len(parts) == 1:
            hours = 0
            minutes = 0
            seconds = int(parts[0])
        else:
            return None
    except ValueError:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def format_duration(seconds: Optional[int]) -> str:
    if seconds is None:
        return "N/A"
    seconds = max(0, int(seconds))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_memory_to_bytes(value: str, cpus: int = 1, nodes: int = 1) -> Optional[int]:
    text = (value or "").strip()
    if not text or text in {"Unknown", "N/A", "None", "(null)", "0"}:
        return None

    multiplier = 1
    if text[-1:].lower() == "n":
        multiplier = max(1, nodes)
        text = text[:-1]
    elif text[-1:].lower() == "c":
        multiplier = max(1, cpus)
        text = text[:-1]

    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)([KMGTPE]?)$", text, re.IGNORECASE)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).upper() or "M"
    factor = MEMORY_UNITS.get(unit)
    if factor is None:
        return None
    return int(amount * factor * multiplier)


def format_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "N/A"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(value) < 1024.0 or unit == "PiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PiB"


def parse_tres(tres: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for part in (tres or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def parse_gpu_from_strings(strings: Iterable[str]) -> Tuple[int, Optional[str]]:
    total = 0
    gpu_type = None
    for text in strings:
        if not text:
            continue
        tres = parse_tres(text)
        for key, value in tres.items():
            if key.startswith("gres/gpu"):
                total += parse_int(value)
                pieces = key.split(":")
                if len(pieces) > 1 and pieces[-1]:
                    gpu_type = gpu_type or pieces[-1]

        for match in re.finditer(r"gpu(?::([A-Za-z0-9_.-]+))?[:=]([0-9]+)", text):
            if parse_tres(text):
                continue
            gpu_type = gpu_type or match.group(1)
            total += parse_int(match.group(2))

    return total, gpu_type


def parse_memory_from_tres(tres_values: Iterable[str], cpus: int, nodes: int) -> Tuple[Optional[int], str]:
    for tres in tres_values:
        fields = parse_tres(tres)
        mem_value = fields.get("mem")
        if mem_value:
            parsed = parse_memory_to_bytes(mem_value, cpus=cpus, nodes=nodes)
            if parsed is not None:
                return parsed, "tres"
    return None, ""


def parse_scontrol_record(output: str) -> Dict[str, str]:
    line = " ".join(part.strip() for part in output.strip().splitlines() if part.strip())
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:/-]*)=", line))
    record: Dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        record[key] = line[start:end].strip()
    return record


def parse_pipe_rows(output: str, field_names: List[str]) -> List[Dict[str, str]]:
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("|")
        if len(parts) < len(field_names):
            parts.extend([""] * (len(field_names) - len(parts)))
        rows.append(dict(zip(field_names, parts[: len(field_names)])))
    return rows


def load_config(path: Path) -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG_DATA))
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            user_config = json.load(handle)
        deep_update(config, user_config)
    return config


def deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value


def rates_for_job(config: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, float]:
    rates = dict(config.get("rates", {}))
    partition = job.get("partition") or ""
    partition_rates = config.get("partition_rates", {}).get(partition)
    if isinstance(partition_rates, dict):
        rates.update(partition_rates)

    gpu_type = job.get("gpu_type") or ""
    gpu_type_rates = config.get("gpu_type_rates", {}).get(gpu_type)
    if isinstance(gpu_type_rates, dict):
        rates.update(gpu_type_rates)

    return {
        "gpu_hour": parse_float(rates.get("gpu_hour")),
        "cpu_hour": parse_float(rates.get("cpu_hour")),
        "memory_gb_hour": parse_float(rates.get("memory_gb_hour")),
        "billing_hour": parse_float(rates.get("billing_hour")),
    }


def add_cost(job: Dict[str, Any], config: Dict[str, Any]) -> None:
    elapsed_seconds = parse_int(job.get("elapsed_seconds"))
    elapsed_hours = elapsed_seconds / 3600.0
    gpus = parse_int(job.get("gpus"))
    cpus = parse_int(job.get("cpus"))
    memory_gb = parse_float(job.get("memory_gb"))
    billing = parse_float(job.get("billing_units"))
    rates = rates_for_job(config, job)

    gpu_hours = gpus * elapsed_hours
    cpu_hours = cpus * elapsed_hours
    memory_gb_hours = memory_gb * elapsed_hours
    billing_hours = billing * elapsed_hours
    cost_units = (
        gpu_hours * rates["gpu_hour"]
        + cpu_hours * rates["cpu_hour"]
        + memory_gb_hours * rates["memory_gb_hour"]
        + billing_hours * rates["billing_hour"]
    )

    job["gpu_hours"] = round(gpu_hours, 4)
    job["cpu_hours"] = round(cpu_hours, 4)
    job["memory_gb_hours"] = round(memory_gb_hours, 4)
    job["billing_hours"] = round(billing_hours, 4)
    job["cost_units"] = round(cost_units, 4)
    job["cost_rates"] = rates


def base_job(job_id: str) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "display_job_id": job_id,
        "name": "",
        "user": "",
        "state": "UNKNOWN",
        "state_group": "unknown",
        "partition": "",
        "account": "",
        "submit_time": None,
        "start_time": None,
        "end_time": None,
        "elapsed_seconds": 0,
        "elapsed": "00:00:00",
        "timelimit_seconds": None,
        "timelimit": "N/A",
        "nodes": 0,
        "cpus": 0,
        "gpus": 0,
        "gpu_type": None,
        "memory_bytes": None,
        "memory_gb": 0.0,
        "memory": "N/A",
        "memory_source": "",
        "billing_units": 0.0,
        "node_list": "",
        "reason": "",
        "alloc_tres": "",
        "req_tres": "",
        "source": "",
    }


def job_from_sacct(row: Dict[str, str], config: Dict[str, Any], collected_at: datetime) -> Dict[str, Any]:
    job_id = row.get("JobIDRaw") or row.get("JobID") or ""
    job = base_job(job_id)
    job.update(
        {
            "display_job_id": row.get("JobID", job_id),
            "name": row.get("JobName", ""),
            "user": row.get("User", ""),
            "state": split_state(row.get("State", "")),
            "partition": row.get("Partition", ""),
            "account": row.get("Account", ""),
            "submit_time": isoformat(parse_slurm_datetime(row.get("Submit", ""))),
            "start_time": isoformat(parse_slurm_datetime(row.get("Start", ""))),
            "end_time": isoformat(parse_slurm_datetime(row.get("End", ""))),
            "nodes": parse_int(row.get("NNodes")),
            "cpus": parse_int(row.get("NCPUS")),
            "node_list": row.get("NodeList", ""),
            "alloc_tres": row.get("AllocTRES", ""),
            "req_tres": row.get("ReqTRES", ""),
            "source": "sacct",
        }
    )

    elapsed = parse_int(row.get("ElapsedRaw"))
    job["elapsed_seconds"] = elapsed
    job["elapsed"] = row.get("Elapsed") or format_duration(elapsed)

    timelimit_raw = row.get("TimelimitRaw", "")
    if timelimit_raw and timelimit_raw not in {"Unknown", "N/A"}:
        job["timelimit_seconds"] = parse_int(timelimit_raw) * 60
        job["timelimit"] = format_duration(job["timelimit_seconds"])

    alloc_tres = row.get("AllocTRES", "")
    req_tres = row.get("ReqTRES", "")
    gpus, gpu_type = parse_gpu_from_strings([alloc_tres, req_tres])
    job["gpus"] = gpus
    job["gpu_type"] = gpu_type

    memory_bytes, memory_source = parse_memory_from_tres([alloc_tres, req_tres], job["cpus"], job["nodes"])
    if memory_bytes is None:
        memory_bytes = parse_memory_to_bytes(row.get("ReqMem", ""), cpus=job["cpus"], nodes=job["nodes"])
        memory_source = "ReqMem" if memory_bytes is not None else ""
    job["memory_bytes"] = memory_bytes
    job["memory_gb"] = round((memory_bytes or 0) / (1024.0**3), 4)
    job["memory"] = format_bytes(memory_bytes)
    job["memory_source"] = memory_source

    tres_fields = parse_tres(alloc_tres) or parse_tres(req_tres)
    job["billing_units"] = parse_float(tres_fields.get("billing"))
    job["state_group"] = state_group(job["state"])
    job["updated_at"] = isoformat(collected_at)
    add_cost(job, config)
    return job


def job_from_scontrol(record: Dict[str, str], config: Dict[str, Any], collected_at: datetime) -> Dict[str, Any]:
    job_id = record.get("JobId") or record.get("JobID") or ""
    job = base_job(job_id)
    job["display_job_id"] = job_id
    job["name"] = record.get("JobName", "")
    job["user"] = record.get("UserId", "").split("(", 1)[0]
    job["state"] = split_state(record.get("JobState", ""))
    job["partition"] = record.get("Partition", "")
    job["account"] = record.get("Account", "")
    job["submit_time"] = isoformat(parse_slurm_datetime(record.get("SubmitTime", "")))
    job["start_time"] = isoformat(parse_slurm_datetime(record.get("StartTime", "")))
    job["end_time"] = isoformat(parse_slurm_datetime(record.get("EndTime", "")))
    job["nodes"] = parse_int(record.get("NumNodes"))
    job["cpus"] = parse_int(record.get("NumCPUs"))
    job["node_list"] = record.get("NodeList", "")
    job["reason"] = record.get("Reason", "")
    job["source"] = "scontrol"

    runtime = parse_slurm_duration(record.get("RunTime", ""))
    if runtime is None and job["start_time"] and job["state"] == "RUNNING":
        start = parse_slurm_datetime(record.get("StartTime", ""))
        runtime = int((collected_at - start).total_seconds()) if start else 0
    job["elapsed_seconds"] = runtime or 0
    job["elapsed"] = format_duration(job["elapsed_seconds"])

    timelimit = parse_slurm_duration(record.get("TimeLimit", ""))
    job["timelimit_seconds"] = timelimit
    job["timelimit"] = format_duration(timelimit)

    tres_candidates = [
        record.get("AllocTRES", ""),
        record.get("ReqTRES", ""),
        record.get("TRES", ""),
        record.get("TresPerNode", ""),
        record.get("Gres", ""),
    ]
    job["alloc_tres"] = record.get("AllocTRES", "") or record.get("TRES", "")
    job["req_tres"] = record.get("ReqTRES", "") or record.get("TresPerNode", "")
    gpus, gpu_type = parse_gpu_from_strings(tres_candidates)
    job["gpus"] = gpus
    job["gpu_type"] = gpu_type

    memory_bytes, memory_source = parse_memory_from_tres(tres_candidates, job["cpus"], job["nodes"])
    if memory_bytes is None:
        memory_bytes = parse_memory_to_bytes(record.get("MinMemoryNode", ""), cpus=job["cpus"], nodes=job["nodes"])
        memory_source = "MinMemoryNode" if memory_bytes is not None else ""
    if memory_bytes is None:
        memory_bytes = parse_memory_to_bytes(record.get("MinMemoryCPU", ""), cpus=job["cpus"], nodes=job["nodes"])
        memory_source = "MinMemoryCPU" if memory_bytes is not None else ""

    job["memory_bytes"] = memory_bytes
    job["memory_gb"] = round((memory_bytes or 0) / (1024.0**3), 4)
    job["memory"] = format_bytes(memory_bytes)
    job["memory_source"] = memory_source

    tres_fields = parse_tres(record.get("AllocTRES", "")) or parse_tres(record.get("TRES", ""))
    job["billing_units"] = parse_float(tres_fields.get("billing"))
    job["state_group"] = state_group(job["state"])
    job["updated_at"] = isoformat(collected_at)
    add_cost(job, config)
    return job


def collect_sacct(args: argparse.Namespace, config: Dict[str, Any], collected_at: datetime) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    start = (collected_at - timedelta(days=args.days)).strftime("%Y-%m-%dT%H:%M:%S")
    command = [
        "sacct",
        "-P",
        "-n",
        "--allocations",
        "-S",
        start,
        "-o",
        ",".join(SACCT_FIELDS),
    ]
    if args.user and not args.all_users:
        command.extend(["-u", args.user])
    if args.account:
        command.extend(["-A", args.account])

    code, stdout, stderr = run_command(command, timeout=args.command_timeout)
    if code != 0:
        errors.append(f"sacct failed ({code}): {stderr.strip()}")
        return [], errors

    rows = parse_pipe_rows(stdout, SACCT_NAMES)
    jobs = [job_from_sacct(row, config, collected_at) for row in rows if row.get("JobIDRaw") or row.get("JobID")]
    return jobs, errors


def collect_active_job_ids(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    command = ["squeue", "-h", "-o", "%i"]
    if args.user and not args.all_users:
        command.extend(["-u", args.user])
    if args.account:
        command.extend(["-A", args.account])

    code, stdout, stderr = run_command(command, timeout=args.command_timeout)
    if code != 0:
        errors.append(f"squeue failed ({code}): {stderr.strip()}")
        return [], errors
    job_ids = [line.strip() for line in stdout.splitlines() if line.strip()]
    return job_ids, errors


def collect_scontrol(job_ids: List[str], args: argparse.Namespace, config: Dict[str, Any], collected_at: datetime) -> Tuple[List[Dict[str, Any]], List[str]]:
    jobs = []
    errors = []
    for job_id in job_ids:
        command = ["scontrol", "show", "job", "-o", job_id]
        code, stdout, stderr = run_command(command, timeout=args.command_timeout)
        if code != 0:
            errors.append(f"scontrol failed for {job_id} ({code}): {stderr.strip()}")
            continue
        record = parse_scontrol_record(stdout)
        if not record:
            continue
        job = job_from_scontrol(record, config, collected_at)
        if job["job_id"]:
            jobs.append(job)
    return jobs, errors


def merge_jobs(history_jobs: List[Dict[str, Any]], active_jobs: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for job in history_jobs:
        merged[job["job_id"]] = job
    for active in active_jobs:
        existing = merged.get(active["job_id"], {})
        combined = dict(existing)
        for key, value in active.items():
            if value not in ("", None, 0, "N/A") or key in {"elapsed_seconds", "state", "state_group", "source"}:
                combined[key] = value
        if existing:
            combined["source"] = "sacct+scontrol"
        add_cost(combined, config)
        merged[active["job_id"]] = combined

    return sorted(
        merged.values(),
        key=lambda job: (
            0 if job.get("state_group") in {"running", "pending", "active"} else 1,
            job.get("start_time") or job.get("submit_time") or "",
            job.get("job_id") or "",
        ),
        reverse=False,
    )


def summarize(jobs: List[Dict[str, Any]], config: Dict[str, Any], collected_at: datetime, errors: List[str]) -> Dict[str, Any]:
    active_jobs = [job for job in jobs if job.get("state_group") in {"running", "pending", "active"}]
    running_jobs = [job for job in jobs if job.get("state_group") == "running"]
    state_counts = Counter(job.get("state", "UNKNOWN") for job in jobs)
    active_gpu_count = sum(parse_int(job.get("gpus")) for job in active_jobs)
    active_memory_bytes = sum(parse_int(job.get("memory_bytes")) for job in active_jobs if job.get("memory_bytes") is not None)
    active_cpus = sum(parse_int(job.get("cpus")) for job in active_jobs)
    running_seconds = sum(parse_int(job.get("elapsed_seconds")) for job in running_jobs)
    active_cost = sum(parse_float(job.get("cost_units")) for job in active_jobs)
    total_cost = sum(parse_float(job.get("cost_units")) for job in jobs)
    total_gpu_hours = sum(parse_float(job.get("gpu_hours")) for job in jobs)
    total_cpu_hours = sum(parse_float(job.get("cpu_hours")) for job in jobs)
    total_memory_gb_hours = sum(parse_float(job.get("memory_gb_hours")) for job in jobs)

    return {
        "collected_at": isoformat(collected_at),
        "cost_label": config.get("cost_label", "cost units"),
        "jobs": len(jobs),
        "active_jobs": len(active_jobs),
        "running_jobs": len(running_jobs),
        "pending_jobs": sum(1 for job in jobs if job.get("state_group") == "pending"),
        "finished_jobs": sum(1 for job in jobs if job.get("state_group") == "finished"),
        "state_counts": dict(sorted(state_counts.items())),
        "active_gpus": active_gpu_count,
        "active_cpus": active_cpus,
        "active_memory_bytes": active_memory_bytes,
        "active_memory": format_bytes(active_memory_bytes),
        "running_elapsed_seconds": running_seconds,
        "running_elapsed": format_duration(running_seconds),
        "active_cost_units": round(active_cost, 4),
        "total_cost_units": round(total_cost, 4),
        "total_gpu_hours": round(total_gpu_hours, 4),
        "total_cpu_hours": round(total_cpu_hours, 4),
        "total_memory_gb_hours": round(total_memory_gb_hours, 4),
        "errors": errors,
    }


def collect_snapshot(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    collected_at = now_local()
    errors: List[str] = []
    history_jobs, sacct_errors = collect_sacct(args, config, collected_at)
    errors.extend(sacct_errors)
    active_ids, squeue_errors = collect_active_job_ids(args)
    errors.extend(squeue_errors)
    active_jobs, scontrol_errors = collect_scontrol(active_ids, args, config, collected_at)
    errors.extend(scontrol_errors)
    jobs = merge_jobs(history_jobs, active_jobs, config)
    summary = summarize(jobs, config, collected_at, errors)

    return {
        "schema_version": 1,
        "generated_by": "slurm_monitor.monitor",
        "query": {
            "user": None if args.all_users else args.user,
            "all_users": bool(args.all_users),
            "account": args.account,
            "days": args.days,
        },
        "summary": summary,
        "jobs": jobs,
    }


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temp_path), str(path))


def append_history(path: Path, snapshot: Dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "collected_at": snapshot["summary"]["collected_at"],
        "summary": snapshot["summary"],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll Slurm jobs and write dashboard state JSON.")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE, help="Path to write current job state JSON.")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY, help="Path to append summary snapshots as JSONL.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Optional cost-rate config JSON.")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds for daemon mode.")
    parser.add_argument("--days", type=int, default=14, help="How many days of sacct history to include.")
    parser.add_argument("--user", default=os.environ.get("SLURM_MONITOR_USER", getpass.getuser()), help="Slurm user to query.")
    parser.add_argument("--account", default=os.environ.get("SLURM_MONITOR_ACCOUNT", ""), help="Optional Slurm account filter.")
    parser.add_argument("--all-users", action="store_true", help="Do not pass a user filter; useful with --account if allowed.")
    parser.add_argument("--once", action="store_true", help="Collect one snapshot and exit.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    parser.add_argument("--command-timeout", type=int, default=30, help="Timeout in seconds for each Slurm command.")
    return parser.parse_args(argv)


def log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    while True:
        try:
            snapshot = collect_snapshot(args, config)
            atomic_write_json(args.state, snapshot)
            append_history(args.history, snapshot)
            summary = snapshot["summary"]
            log(
                (
                    f"{summary['collected_at']} jobs={summary['jobs']} "
                    f"running={summary['running_jobs']} pending={summary['pending_jobs']} "
                    f"gpus={summary['active_gpus']} mem={summary['active_memory']} "
                    f"cost={summary['active_cost_units']} {summary['cost_label']}"
                ),
                args.quiet,
            )
        except KeyboardInterrupt:
            log("stopping", args.quiet)
            return 0
        except Exception as exc:
            log(f"monitor error: {exc}", args.quiet)

        if args.once:
            return 0
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    sys.exit(main())
