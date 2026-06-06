
import argparse
import csv
import gc
import json
import sys
import yaml
import torch
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
from train import main as train_main
from evaluate import main as eval_main

CONFIG_PATH = Path("configs/training.yaml")
RESULTS_DIR = Path("results")
SUMMARY_CSV = RESULTS_DIR / "summary.csv"

def run_one(model, max_length, lr, train_size, config, dataset="uk", save_checkpoint=True):
    tag = f"{dataset}__{model.split('/')[-1]}__ml{max_length}__lr{lr:.0e}__ts{train_size or 'full'}"
    out_dir = RESULTS_DIR / tag

    print(f"\n{'='*60}")
    print(f"RUN: {tag}")
    print(f"{'='*60}")

    best_f1 = train_main(model, str(out_dir), config,
                         max_length=max_length, lr=lr, train_size=train_size,
                         dataset=dataset, save_checkpoint=True)
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    test_metrics_path = out_dir / "test_metrics.json"
    eval_main(str(out_dir / "best"), str(test_metrics_path),
              max_length=max_length, batch_size=config["training"]["batch_size"],
              dataset=dataset)
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not save_checkpoint:
        for fname in ["model.safetensors", "pytorch_model.bin"]:
            f = out_dir / "best" / fname
            if f.exists():
                f.unlink()

    with open(test_metrics_path) as f:
        m = json.load(f)

    return {
        "dataset":      dataset,
        "model":        model,
        "max_length":   max_length,
        "lr":           lr,
        "train_size":   train_size or "full",
        "val_f1":       round(best_f1, 4),
        "test_acc":     round(m["accuracy"], 4),
        "test_precision": round(m["precision"], 4),
        "test_recall":  round(m["recall"], 4),
        "test_f1":      round(m["f1"], 4),
        "run_dir":      str(out_dir),
    }

def main(sweep: str | None = None, dataset: str = "uk", model_filter: str | None = None):
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    exp = config["experiments"]
    models      = config["models"]
    default_ml  = config["data"]["max_length"]
    default_lr  = config["training"]["learning_rate"]

    runs = []

    if sweep == "main":

        for model in models:
            runs.append((model, default_ml, default_lr, None))

    if sweep in (None, "size"):

        mps_active = hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not torch.cuda.is_available()
        for model, size in product(models, exp["training_sizes"]):
            if mps_active and "large" in model.lower():
                print(f"  SKIP on MPS (too large): {model}")
                continue
            runs.append((model, default_ml, default_lr, size))

    if sweep in (None, "hp"):

        for model, ml, lr in product(models, exp["max_lengths"], exp["learning_rates"]):
            if (model, ml, default_lr, None) not in runs:
                runs.append((model, ml, lr, None))

    seen, unique_runs = set(), []
    for r in runs:
        if r not in seen:
            seen.add(r)
            if model_filter is None or model_filter.lower() in r[0].lower():
                unique_runs.append(r)

    print(f"Total runs scheduled: {len(unique_runs)}")

    fieldnames = ["dataset", "model", "max_length", "lr", "train_size",
                  "val_f1", "test_acc", "test_precision", "test_recall", "test_f1", "run_dir"]

    existing = set()
    if SUMMARY_CSV.exists():
        with open(SUMMARY_CSV) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "dataset" in reader.fieldnames:
                existing = {tuple(r[k] for k in ["dataset", "model", "max_length", "lr", "train_size"])
                            for r in reader}

    with open(SUMMARY_CSV, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if SUMMARY_CSV.stat().st_size == 0:
            writer.writeheader()

        is_main = sweep == "main"

        for model, ml, lr, size in unique_runs:
            key = (dataset, model, str(ml), str(lr), str(size or "full"))
            if key in existing:
                print(f"  SKIP (already done): {key}")
                continue
            row = run_one(model, ml, lr, size, config, dataset=dataset,
                          save_checkpoint=is_main)
            writer.writerow(row)
            csvfile.flush()
            print(f"  -> saved to {SUMMARY_CSV}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=["main", "size", "hp"], default="main",
                        help="main=one run per model (default); size=training size curves; hp=hyperparameter search")
    parser.add_argument("--dataset", choices=["uk", "en", "both"], default="uk",
                        help="Which dataset to train on")
    parser.add_argument("--model", default=None,
                        help="Run only this model (substring match, e.g. mdeberta)")
    args = parser.parse_args()
    main(args.sweep, args.dataset, model_filter=args.model)
