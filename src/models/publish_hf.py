"""
Publish a trained query-intent model to the Hugging Face Hub.

Prerequisites (do once):
    pip install huggingface_hub
    huggingface-cli login          # paste your WRITE token

Usage:
    python src/models/publish_hf.py --repo YOUR_USERNAME/query-intent-uk --dataset uk
"""
import argparse
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def best_checkpoint(dataset: str) -> Path | None:
    df = pd.read_csv("results/summary.csv")
    df = df[(df["dataset"] == dataset) & (df["train_size"].astype(str) == "full")]
    df = df.sort_values("test_f1", ascending=False)
    for _, row in df.iterrows():
        ckpt = Path(row["run_dir"]) / "best"
        if (ckpt / "model.safetensors").exists() or (ckpt / "pytorch_model.bin").exists():
            return ckpt
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. username/query-intent-uk")
    ap.add_argument("--dataset", choices=["uk", "en", "both"], default="uk")
    ap.add_argument("--checkpoint", default=None, help="explicit checkpoint path (optional)")
    ap.add_argument("--private", action="store_true", help="create a private repo")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else best_checkpoint(args.dataset)
    if ckpt is None or not ckpt.exists():
        raise SystemExit(f"No trained checkpoint with weights found for dataset={args.dataset}. "
                         "Train it first with src/models/train.py")

    print(f"Publishing {ckpt} -> {args.repo}")
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    tok   = AutoTokenizer.from_pretrained(ckpt)

    model.config.id2label = {0: "solution_oriented", 1: "learning_oriented"}
    model.config.label2id = {"solution_oriented": 0, "learning_oriented": 1}

    model.push_to_hub(args.repo, private=args.private)
    tok.push_to_hub(args.repo, private=args.private)
    print(f"Done → https://huggingface.co/{args.repo}")

if __name__ == "__main__":
    main()
