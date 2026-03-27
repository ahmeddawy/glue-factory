"""
Monitor glue-factory LightGlue training experiments.
Reads TensorBoard events from a local directory (sync from GCS first).

TensorBoard key structure (glue-factory):
  Train (step = tot_n_samples):
    training//loss/total      -- main training loss
    training//loss/last       -- last-layer loss only
    training//loss/confidence -- token confidence loss
    training//loss/row_norm   -- assignment row norm diagnostic
    training//lr              -- learning rate
    training//epoch           -- current epoch number

  Val (step = tot_n_samples, written every eval_every_iter steps):
    val/loss/total            -- main val loss  ← best_key
    val/loss/last
    val/match_recall          -- fraction of GT matches found
    val/match_precision       -- fraction of predicted matches correct
    val/accuracy
    val/average_precision

Usage:
  # 1. Sync TB events from GCS
  gsutil -m rsync -r gs://rembrand-data/dawy/glue_factory_exp/lightglue_exp01/ /tmp/tb_exp01/

  # 2. Run this script
  python monitor_exp.py
"""

import glob
import math
import os
import shutil
import struct
import tempfile

import numpy as np


EXPERIMENTS = [
    # (exp_id, local_tb_dir)
    ("EXP-01", "/tmp/tb_exp01"),
]

# Keys to display
TRAIN_KEYS = ["loss/total", "loss/last", "loss/confidence"]
VAL_KEYS   = ["loss/total", "loss/last", "match_recall", "match_precision", "average_precision"]


def tensor_val(e):
    tp = e.tensor_proto
    if tp.float_val:
        return tp.float_val[0]
    if tp.double_val:
        return tp.double_val[0]
    if tp.tensor_content:
        return struct.unpack("f", tp.tensor_content[:4])[0]
    return float("nan")


def scalar_val(e):
    """Handle both scalar and tensor events."""
    try:
        return e.simple_value
    except AttributeError:
        return tensor_val(e)


def parse(exp_id, tb_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print("tensorboard not installed — pip install tensorboard")
        return

    files = [
        f for f in glob.glob(f"{tb_dir}/events.out.tfevents.*")
        if os.path.getsize(f) > 1000
    ]
    if not files:
        print(f"{exp_id}: no TB data at {tb_dir}")
        return

    tmp = tempfile.mkdtemp()
    shutil.copy(max(files, key=os.path.getmtime), tmp)
    ea = EventAccumulator(tmp, size_guidance={"scalars": 0, "tensors": 0})
    ea.Reload()
    shutil.rmtree(tmp)

    tags = ea.Tags()
    scalars  = {t: ea.Scalars(t)  for t in tags.get("scalars",  [])}
    tensors  = {t: ea.Tensors(t)  for t in tags.get("tensors",  [])}

    def get_latest(key):
        """Try scalars then tensors, return (value, step) or (nan, -1)."""
        if key in scalars and scalars[key]:
            e = scalars[key][-1]
            return e.value, e.step
        if key in tensors and tensors[key]:
            e = tensors[key][-1]
            return tensor_val(e), e.step
        return float("nan"), -1

    # Current epoch
    epoch_val, _ = get_latest("training/epoch")
    epoch = int(epoch_val) if not math.isnan(epoch_val) else "?"

    # LR
    lr, _ = get_latest("training/lr")

    # Latest train values
    train_results = {}
    for k in TRAIN_KEYS:
        v, _ = get_latest(f"training//{k}")
        if math.isnan(v):
            v, _ = get_latest(f"training/{k}")  # fallback without double slash
        train_results[k] = v

    # Latest val values
    val_results = {}
    for k in VAL_KEYS:
        v, step = get_latest(f"val/{k}")
        val_results[k] = v

    # Best val/loss/total
    best_val, best_step = float("inf"), -1
    key = "val/loss/total"
    events = scalars.get(key, []) or tensors.get(key, [])
    for e in events:
        v = e.value if hasattr(e, "value") else tensor_val(e)
        if not math.isnan(v) and v < best_val:
            best_val = v
            best_step = e.step

    # Issue detection
    issues = []
    for k, v in {**train_results, **val_results}.items():
        if math.isnan(v) or math.isinf(v):
            issues.append(f"NaN/Inf in {k}")
        if "loss" in k and abs(v) > 100:
            issues.append(f"Explosion {k}={v:.1f}")
    vl = val_results.get("loss/total", float("nan"))
    if not math.isnan(vl) and vl > 1.0:
        issues.append(f"val/loss/total HIGH={vl:.3f}")
    vr = val_results.get("match_recall", float("nan"))
    if not math.isnan(vr) and vr < 0.3 and epoch != "?":
        issues.append(f"match_recall LOW={vr:.3f}")

    status = "ISSUE" if issues else "ok"
    print(f"\n[{status}] {exp_id}  ep{epoch}  lr={lr:.2e}")

    print("  Train:")
    for k, v in train_results.items():
        print(f"    {k}: {v:.4f}")

    print("  Val:")
    for k, v in val_results.items():
        print(f"    {k}: {v:.4f}")

    if best_step >= 0:
        print(f"  best val/loss/total: {best_val:.4f}  (step {best_step})")

    for issue in issues:
        print(f"  !! {issue}")


def sync_and_parse():
    """Sync from GCS then parse. Edit GCS paths to match your setup."""
    gsutil = "/home/oem/google-cloud-sdk/bin/gsutil"
    gcs_base = "gs://rembrand-data/dawy/glue_factory_exp"

    for exp_id, local_dir in EXPERIMENTS:
        exp_name = exp_id.lower().replace("-", "").replace("exp", "lightglue_exp")
        gcs_path = f"{gcs_base}/{exp_name}/"
        os.makedirs(local_dir, exist_ok=True)

        # Remove incomplete transfer files
        for f in glob.glob(f"{local_dir}/*.gstmp"):
            os.remove(f)

        print(f"Syncing {exp_id} from GCS...")
        os.system(f"{gsutil} -m rsync -r '{gcs_path}' '{local_dir}/' 2>&1 | tail -1")

    print("\n=== Experiment Status ===")
    for exp_id, local_dir in EXPERIMENTS:
        parse(exp_id, local_dir)


if __name__ == "__main__":
    import sys
    if "--no-sync" in sys.argv:
        print("=== Experiment Status (local only) ===")
        for exp_id, local_dir in EXPERIMENTS:
            parse(exp_id, local_dir)
    else:
        sync_and_parse()
