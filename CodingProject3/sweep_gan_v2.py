#!/usr/bin/env python3
"""
GAN Hyperparameter Sweep V2 — Focused search based on Round 1 findings.

Key changes from v1:
  - Fixed betas=(0.0, 0.9) — clearly the best from round 1
  - Higher lr_d range (5e-4, 7e-4, 1e-3) — strongest single factor
  - Finer lr_g grid around 2e-4 sweet spot
  - Added n_critic=3
  - Dropped wasserstein loss (worst performer)
  - Increased to 80 epochs

Grid: 3 batch × 3 lr_g × 3 lr_d × 2 loss × 3 n_critic = 162 configs
Estimated runtime: ~22-25 hours

Usage:
    uv run python sweep_gan_v2.py                  # run full sweep
    uv run python sweep_gan_v2.py --max-runs 10    # limit to 10 runs
    uv run python sweep_gan_v2.py --dry-run        # list configs without running
    uv run python sweep_gan_v2.py --report-only    # regenerate report
"""

import argparse
import gc
import itertools
import json
import logging
import math
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import v2

from modules.gan import CustomGANGenerator, CustomGANDiscriminator
from utils import DEVICE, ensure_reproducibility

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("sweep_results_v2")
RESULTS_FILE = RESULTS_DIR / "results.json"
REPORT_FILE = RESULTS_DIR / "sweep_report.md"
BEST_CKPT = Path("checkpoints/gan_sweep_v2_best.pth")

# ---------------------------------------------------------------------------
# Dataset transform
# ---------------------------------------------------------------------------
TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# ---------------------------------------------------------------------------
# Hyperparameter grid — Round 2
# ---------------------------------------------------------------------------
HPARAM_GRID = {
    "batch_size":  [64, 128, 256],
    "lr_g":        [1.5e-4, 2e-4, 3e-4],
    "lr_d":        [5e-4, 7e-4, 1e-3],
    "loss_type":   ["hinge", "bce"],
    "n_critic":    [1, 2, 3],
}
# Fixed from round 1 findings
FIXED_BETAS = (0.0, 0.9)
N_EPOCHS = 80

# Diversity thresholds from README
MIN_STD = [0.17, 0.08, 0.17, 0.15, 0.14, 0.16, 0.15, 0.13, 0.15, 0.13]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(RESULTS_DIR / "sweep.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

logger = logging.getLogger(__name__)

# ===================================================================
# Config helpers
# ===================================================================

def generate_configs():
    keys = sorted(HPARAM_GRID.keys())
    values = [HPARAM_GRID[k] for k in keys]
    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        config["betas"] = list(FIXED_BETAS)
        config["n_epochs"] = N_EPOCHS
        configs.append(config)
    return configs


def config_id(config):
    return (
        f"bs{config['batch_size']}"
        f"_lrg{config['lr_g']}"
        f"_lrd{config['lr_d']}"
        f"_{config['loss_type']}"
        f"_nc{config['n_critic']}"
    )

# ===================================================================
# Loss functions
# ===================================================================

def get_loss_fns(loss_type):
    if loss_type == "hinge":
        def d_loss_fn(d_real, d_fake):
            return torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()
        def g_loss_fn(d_fake):
            return -d_fake.mean()

    elif loss_type == "bce":
        bce = nn.BCEWithLogitsLoss()
        def d_loss_fn(d_real, d_fake):
            return (bce(d_real, torch.ones_like(d_real))
                    + bce(d_fake, torch.zeros_like(d_fake)))
        def g_loss_fn(d_fake):
            return bce(d_fake, torch.ones_like(d_fake))

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return d_loss_fn, g_loss_fn

# ===================================================================
# Training
# ===================================================================

def train_one_config(config, dataset):
    """Train a GAN with *config*. Returns (generator, final_d_loss, final_g_loss, seconds)."""
    ensure_reproducibility()

    generator = CustomGANGenerator().to(DEVICE)
    discriminator = CustomGANDiscriminator().to(DEVICE)
    generator.train()
    discriminator.train()

    bs = config["batch_size"]
    betas = tuple(config["betas"])
    n_critic = config["n_critic"]
    latent_dim = generator.latent_dim

    dataloader = DataLoader(
        dataset, batch_size=bs, shuffle=True, num_workers=4,
        pin_memory=torch.cuda.is_available(), drop_last=True,
    )

    opt_g = torch.optim.Adam(generator.parameters(), lr=config["lr_g"], betas=betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config["lr_d"], betas=betas)
    d_loss_fn, g_loss_fn = get_loss_fns(config["loss_type"])

    start = time.time()
    final_d = final_g = 0.0
    diverged = False

    for epoch in range(config["n_epochs"]):
        total_d = 0.0
        total_g = 0.0
        n_d = n_g = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            curr_bs = images.size(0)
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            real = images * 2 - 1

            # --- Discriminator ---
            opt_d.zero_grad()
            d_real = discriminator(real, labels)
            z = torch.randn(curr_bs, latent_dim, device=DEVICE)
            fake = generator(z, labels).detach()
            d_fake = discriminator(fake, labels)
            d_loss = d_loss_fn(d_real, d_fake)
            d_loss.backward()
            opt_d.step()
            total_d += d_loss.item()
            n_d += 1

            # --- Generator (every n_critic steps) ---
            if (batch_idx + 1) % n_critic == 0:
                opt_g.zero_grad()
                z = torch.randn(curr_bs, latent_dim, device=DEVICE)
                fake = generator(z, labels)
                d_fake = discriminator(fake, labels)
                g_loss = g_loss_fn(d_fake)
                g_loss.backward()
                opt_g.step()
                total_g += g_loss.item()
                n_g += 1

            # NaN detection
            if math.isnan(d_loss.item()) or (n_g > 0 and math.isnan(total_g)):
                diverged = True
                break

        if diverged:
            logger.warning(f"  NaN detected at epoch {epoch+1}, aborting run.")
            break

        final_d = total_d / n_d if n_d else 0.0
        final_g = total_g / n_g if n_g else 0.0

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}/{config['n_epochs']}, D: {final_d:.4f}, G: {final_g:.4f}")

    elapsed = time.time() - start

    del discriminator, opt_d, opt_g, dataloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if diverged:
        return None, float("nan"), float("nan"), elapsed

    return generator, final_d, final_g, elapsed

# ===================================================================
# Evaluation
# ===================================================================

def evaluate_model(generator, feature_extractor, test_dataset):
    from evaluate_fid import evaluate as fid_evaluate
    ensure_reproducibility()
    generator.eval()
    generator.to(DEVICE)
    feature_extractor.cpu()
    mean_fid, per_digit_fid, per_digit_std = fid_evaluate(
        generator, feature_extractor, test_dataset, "gan", False
    )
    return mean_fid, per_digit_fid, per_digit_std


def check_diversity(per_digit_std):
    return all(s >= t for s, t in zip(per_digit_std, MIN_STD))

# ===================================================================
# Results I/O
# ===================================================================

def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

# ===================================================================
# Report
# ===================================================================

def generate_report(results):
    valid = [r for r in results if r.get("status") == "success" and r.get("mean_fid") is not None]
    failed = [r for r in results if r.get("status") != "success"]
    valid.sort(key=lambda x: x["mean_fid"])

    L = []
    L.append("# GAN Hyperparameter Sweep V2 Report\n")
    L.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    L.append(f"Fixed: betas={list(FIXED_BETAS)}, n_epochs={N_EPOCHS}\n")

    L.append("## Overview\n")
    L.append("| Item | Value |")
    L.append("|------|-------|")
    L.append(f"| Total configurations | {len(results)} |")
    L.append(f"| Successful runs | {len(valid)} |")
    L.append(f"| Failed / diverged | {len(failed)} |")
    if valid:
        total_sec = sum(r.get("training_time", 0) for r in results)
        L.append(f"| Total wall time | {timedelta(seconds=int(total_sec))} |")
        L.append(f"| Best mean FID | {valid[0]['mean_fid']:.4f} |")
        L.append(f"| Median mean FID | {valid[len(valid)//2]['mean_fid']:.4f} |")
        L.append(f"| Worst mean FID | {valid[-1]['mean_fid']:.4f} |")
    L.append("")

    if not valid:
        L.append("No successful runs to report.\n")
        _write_report(L)
        return

    # Top 15
    L.append("## Top 15 Configurations (by Mean FID)\n")
    L.append("| # | batch_size | lr_g | lr_d | loss | n_critic | FID | Div OK | Time |")
    L.append("|---|-----------|------|------|------|----------|-----|--------|------|")
    for i, r in enumerate(valid[:15]):
        c = r["config"]
        div = "Y" if r.get("diversity_passed") else "N"
        t = f"{r.get('training_time',0):.0f}s"
        L.append(
            f"| {i+1} | {c['batch_size']} | {c['lr_g']} | {c['lr_d']} "
            f"| {c['loss_type']} | {c['n_critic']} "
            f"| {r['mean_fid']:.2f} | {div} | {t} |"
        )
    L.append("")

    # Best config detail
    best = valid[0]
    L.append("## Best Configuration\n")
    L.append(f"**Config:** `{best['config_id']}`\n")
    L.append("| Param | Value |")
    L.append("|-------|-------|")
    for k, v in sorted(best["config"].items()):
        L.append(f"| {k} | {v} |")
    L.append(f"| mean FID | {best['mean_fid']:.4f} |")
    L.append(f"| diversity passed | {'Yes' if best.get('diversity_passed') else 'No'} |")
    L.append(f"| final D loss | {best.get('final_d_loss', 'N/A')} |")
    L.append(f"| final G loss | {best.get('final_g_loss', 'N/A')} |")
    L.append("")

    L.append("**Per-digit FID:**\n")
    L.append("| Digit | " + " | ".join(str(d) for d in range(10)) + " |")
    L.append("|-------" + "|------" * 10 + "|")
    L.append("| FID   | " + " | ".join(f"{f:.2f}" for f in best["per_digit_fid"]) + " |")
    L.append("| Std   | " + " | ".join(f"{s:.4f}" for s in best["per_digit_std"]) + " |")
    L.append("| Min Std | " + " | ".join(f"{t:.2f}" for t in MIN_STD) + " |")
    L.append("")

    # Best with diversity
    div_pass = [r for r in valid if r.get("diversity_passed")]
    if div_pass:
        div_pass.sort(key=lambda x: x["mean_fid"])
        bp = div_pass[0]
        L.append("## Best Config Passing Diversity Threshold\n")
        L.append(f"- Config: `{bp['config_id']}`")
        L.append(f"- Mean FID: {bp['mean_fid']:.4f}")
        L.append(f"- Checkpoint saved to: `{BEST_CKPT}`\n")

    # Hyperparameter impact
    L.append("## Hyperparameter Impact Analysis\n")
    L.append("Average FID per hyperparameter value (lower = better):\n")

    for param in ["batch_size", "lr_g", "lr_d", "loss_type", "n_critic"]:
        L.append(f"### {param}\n")
        L.append("| Value | Avg FID | Best FID | Count | Div Pass % |")
        L.append("|-------|---------|----------|-------|------------|")
        groups = {}
        for r in valid:
            val = str(r["config"][param])
            groups.setdefault(val, []).append(r)
        for val in sorted(groups):
            g = groups[val]
            avg = sum(r["mean_fid"] for r in g) / len(g)
            best_g = min(r["mean_fid"] for r in g)
            dp = sum(1 for r in g if r.get("diversity_passed")) / len(g) * 100
            L.append(f"| {val} | {avg:.2f} | {best_g:.2f} | {len(g)} | {dp:.0f}% |")
        L.append("")

    # Comparison with round 1
    L.append("## Comparison with Round 1\n")
    L.append("| Metric | Round 1 (50ep) | Round 2 (80ep) |")
    L.append("|--------|---------------|----------------|")
    L.append(f"| Best FID | 3.7554 | {valid[0]['mean_fid']:.4f} |")
    div_best = div_pass[0]['mean_fid'] if div_pass else float('inf')
    L.append(f"| Best FID (div OK) | 4.2603 | {div_best:.4f} |")
    dp_rate = len(div_pass) / len(valid) * 100 if valid else 0
    L.append(f"| Diversity pass rate | 12% | {dp_rate:.0f}% |")
    L.append("")

    if failed:
        L.append("## Failed / Diverged Runs\n")
        L.append(f"Total: {len(failed)}\n")
        for r in failed[:30]:
            L.append(f"- `{r['config_id']}` — {r.get('error', 'NaN divergence')}")
        if len(failed) > 30:
            L.append(f"- ... and {len(failed)-30} more")
        L.append("")

    _write_report(L)


def _write_report(lines):
    text = "\n".join(lines)
    with open(REPORT_FILE, "w") as f:
        f.write(text)
    logger.info(f"Report saved to {REPORT_FILE}")

# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="GAN hyperparameter sweep V2")
    parser.add_argument("--max-runs", type=int, default=0, help="Max runs (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="List configs and exit")
    parser.add_argument("--report-only", action="store_true", help="Generate report from existing results")
    args = parser.parse_args()

    setup_logging()

    configs = generate_configs()
    logger.info(f"Total configurations in grid: {len(configs)}")
    logger.info(f"Fixed: betas={list(FIXED_BETAS)}, n_epochs={N_EPOCHS}")

    if args.dry_run:
        for i, c in enumerate(configs):
            print(f"  [{i+1:3d}] {config_id(c)}")
        print(f"\nTotal: {len(configs)} configs × {N_EPOCHS} epochs each")
        return

    if args.report_only:
        results = load_results()
        if results:
            generate_report(results)
            logger.info(f"Report generated from {len(results)} existing results.")
        else:
            logger.info("No results found.")
        return

    logger.info(f"Device: {DEVICE}")
    logger.info("Loading training dataset...")
    train_dataset = MNIST(root="data", download=True, train=True, transform=TRANSFORM)
    logger.info("Loading test dataset...")
    test_dataset = MNIST(root="data", download=True, train=False)

    logger.info("Loading feature extractor for FID...")
    from evaluate_fid import _load_feature_extractor
    feature_extractor = _load_feature_extractor()

    # Resume support
    results = load_results()
    completed_ids = {r["config_id"] for r in results}
    logger.info(f"Already completed: {len(completed_ids)}")

    best_fid = min(
        (r["mean_fid"] for r in results if r.get("status") == "success"),
        default=float("inf"),
    )
    best_div_fid = min(
        (r["mean_fid"] for r in results if r.get("status") == "success" and r.get("diversity_passed")),
        default=float("inf"),
    )

    remaining = [(i, c) for i, c in enumerate(configs) if config_id(c) not in completed_ids]
    remaining.sort(key=lambda x: -x[1]["batch_size"])
    if args.max_runs > 0:
        remaining = remaining[:args.max_runs]
    logger.info(f"Runs to execute: {len(remaining)}")

    sweep_start = time.time()

    for run_idx, (global_idx, config) in enumerate(remaining):
        cid = config_id(config)
        logger.info(f"\n{'='*70}")
        logger.info(f"[{run_idx+1}/{len(remaining)}] (global {global_idx+1}/{len(configs)})  {cid}")

        result = {
            "config_id": cid,
            "config": config,
            "start_time": datetime.now().isoformat(),
        }

        try:
            generator, final_d, final_g, train_time = train_one_config(config, train_dataset)
            result["final_d_loss"] = final_d
            result["final_g_loss"] = final_g
            result["training_time"] = train_time

            if generator is None:
                raise RuntimeError("Training diverged (NaN)")

            logger.info("  Evaluating FID...")
            mean_fid, per_digit_fid, per_digit_std = evaluate_model(
                generator, feature_extractor, test_dataset
            )
            result["mean_fid"] = mean_fid
            result["per_digit_fid"] = per_digit_fid
            result["per_digit_std"] = per_digit_std
            result["diversity_passed"] = check_diversity(per_digit_std)
            result["status"] = "success"

            logger.info(
                f"  FID={mean_fid:.4f}  Diversity={'PASS' if result['diversity_passed'] else 'FAIL'}"
            )

            # Save best checkpoint (prioritize diversity-passing configs)
            if result["diversity_passed"] and mean_fid < best_div_fid:
                best_div_fid = mean_fid
                BEST_CKPT.parent.mkdir(parents=True, exist_ok=True)
                generator.cpu()
                torch.save(generator.state_dict(), BEST_CKPT)
                result["is_best_div"] = True
                logger.info(f"  *** New best diversity-passing FID={mean_fid:.4f} -> {BEST_CKPT}")
            elif mean_fid < best_fid:
                best_fid = mean_fid
                # Save to a separate path for best-FID (may not pass diversity)
                fid_ckpt = Path("checkpoints/gan_sweep_v2_best_fid.pth")
                fid_ckpt.parent.mkdir(parents=True, exist_ok=True)
                generator.cpu()
                torch.save(generator.state_dict(), fid_ckpt)
                result["is_best_fid"] = True
                logger.info(f"  *** New best FID={mean_fid:.4f} -> {fid_ckpt}")

            del generator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"  FAILED: {e}\n{traceback.format_exc()}")
            result["status"] = "failed"
            result["error"] = str(e)

        result["end_time"] = datetime.now().isoformat()
        results.append(result)
        save_results(results)

        # ETA
        elapsed = time.time() - sweep_start
        avg_per_run = elapsed / (run_idx + 1)
        eta = avg_per_run * (len(remaining) - run_idx - 1)
        logger.info(f"  ETA: {timedelta(seconds=int(eta))}  (avg {avg_per_run:.1f}s/run)")

        # Interim report every 20 runs
        if (run_idx + 1) % 20 == 0:
            generate_report(results)
            logger.info(f"  Interim report updated ({len([r for r in results if r.get('status')=='success'])} successful runs)")

    # Final report
    logger.info("\n" + "=" * 70)
    logger.info("Sweep complete! Generating report...")
    generate_report(results)

    # Copy best to gan_best.pth if it passes diversity
    if BEST_CKPT.exists():
        shutil.copy2(BEST_CKPT, "checkpoints/gan_best.pth")
        logger.info(f"Best diversity-passing checkpoint copied to checkpoints/gan_best.pth")

    logger.info("Done.")


if __name__ == "__main__":
    main()
