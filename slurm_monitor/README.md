# Slurm Monitor

Small dependency-free monitor for Slurm jobs. It records GPU allocation, memory allocation, run time, and configurable cost units for the current user by default.

## Quick Start

From the repository root:

```bash
./slurm_monitor/run_monitor.sh once
./slurm_monitor/run_monitor.sh start
./slurm_monitor/run_dashboard.sh start
```

Open:

```text
http://127.0.0.1:8765/
```

If you are connecting to the cluster over SSH, forward the port:

```bash
ssh -L 8765:127.0.0.1:8765 USER@CLUSTER
```

## Common Commands

```bash
./slurm_monitor/run_monitor.sh status
./slurm_monitor/run_dashboard.sh status
./slurm_monitor/run_monitor.sh stop
./slurm_monitor/run_dashboard.sh stop
./slurm_monitor/run_monitor.sh restart
./slurm_monitor/run_dashboard.sh restart
```

The monitor writes:

```text
slurm_monitor/state/jobs.json
slurm_monitor/state/snapshots.jsonl
slurm_monitor/state/monitor.log
slurm_monitor/state/dashboard.log
```

## Account or Group View

By default the daemon queries jobs for `whoami`. To query an account:

```bash
SLURM_MONITOR_ACCOUNT=PAS1289 ./slurm_monitor/run_monitor.sh restart
```

If your Slurm permissions allow account-wide accounting:

```bash
SLURM_MONITOR_ACCOUNT=PAS1289 SLURM_MONITOR_ALL_USERS=1 ./slurm_monitor/run_monitor.sh restart
```

## Cost Units

The default cost model is:

```text
cost = gpu_hours * 1.0
```

CPU, memory, and Slurm billing TRES rates default to zero. To customize:

```bash
cp slurm_monitor/config.example.json slurm_monitor/config.json
```

Then edit `slurm_monitor/config.json`.

Supported rates:

```json
{
  "rates": {
    "gpu_hour": 1.0,
    "cpu_hour": 0.0,
    "memory_gb_hour": 0.0,
    "billing_hour": 0.0
  },
  "partition_rates": {
    "gpu": {
      "gpu_hour": 1.0
    }
  },
  "gpu_type_rates": {
    "a100": {
      "gpu_hour": 2.0
    }
  }
}
```

## Collection Details

The daemon uses:

- `sacct` for recent accounting history.
- `squeue` for active job IDs.
- `scontrol show job -o` for live job details.

Memory is allocation/request data from Slurm (`AllocTRES.mem`, `ReqTRES.mem`, `ReqMem`, or `MinMemory*`). It does not use RSS or process memory usage.

GPU count comes from Slurm TRES/GRES fields such as `gres/gpu=1`, `gres/gpu:a100=1`, or `gpu:1`.

Run time comes from `ElapsedRaw` for accounting rows and `RunTime` for active jobs.
