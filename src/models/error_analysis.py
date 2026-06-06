"""
Error analysis for the trained query intent classifier.

Finds misclassified examples and prints patterns useful for the thesis.

Usage:
    python src/models/error_analysis.py --dataset uk
    python src/models/error_analysis.py --dataset uk --checkpoint results/.../best
"""
import argparse
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent))
from train import load_split

LABELS = {0: "solution_oriented", 1: "learning_oriented"}

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def find_best_checkpoint(dataset: str) -> Path | None:
    summary = Path("results/summary.csv")
    if not summary.exists():
        return None
    df = pd.read_csv(summary)
    df = df[df["dataset"] == dataset].sort_values("test_f1", ascending=False)
    for _, row in df.iterrows():
        ckpt = Path(row["run_dir"]) / "best"
        if (ckpt / "model.safetensors").exists() or (ckpt / "pytorch_model.bin").exists():
            return ckpt
    return None

def predict_all(checkpoint: Path, df: pd.DataFrame, max_length: int = 256) -> pd.DataFrame:
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()

    texts = df["text"].tolist()
    enc = tokenizer(texts, truncation=True, padding=True,
                    max_length=max_length, return_tensors="pt")

    all_preds, all_confs = [], []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = {k: v[i:i+batch_size].to(device) for k, v in enc.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**batch).logits, dim=-1).cpu()
        all_preds.extend(probs.argmax(dim=-1).tolist())
        all_confs.extend(probs.max(dim=-1).values.tolist())

    result = df.copy()
    result["pred"]       = all_preds
    result["confidence"] = [round(c, 3) for c in all_confs]
    result["correct"]    = result["label"] == result["pred"]
    return result

def analyze(df: pd.DataFrame):
    errors = df[~df["correct"]].copy()
    total  = len(df)
    n_err  = len(errors)

    print(f"\n{'='*60}")
    print(f"TEST SET:  {total} examples  |  Errors: {n_err} ({n_err/total:.1%})")
    print(f"{'='*60}")

    fn = errors[(errors["label"] == 1) & (errors["pred"] == 0)]
    fp = errors[(errors["label"] == 0) & (errors["pred"] == 1)]

    print(f"\nFalse negatives (learning → predicted solution): {len(fn)}")
    print(f"False positives (solution → predicted learning):  {len(fp)}")

    for name, subset in [("FALSE NEGATIVES — should be learning_oriented", fn),
                         ("FALSE POSITIVES — should be solution_oriented", fp)]:
        print(f"\n{'─'*60}")
        print(f"{name}  ({len(subset)} examples)")
        print(f"{'─'*60}")
        for _, row in subset.sort_values("confidence").iterrows():
            print(f"  conf={row['confidence']:.2f}  {row['text'][:100]}")

    print(f"\n{'─'*60}")
    print("FIRST WORD PATTERNS in errors")
    print(f"{'─'*60}")
    errors["first_word"] = errors["text"].str.split().str[0].str.lower()
    fw_counts = errors["first_word"].value_counts().head(15)
    for word, count in fw_counts.items():
        correct_with_word = df[df["text"].str.lower().str.startswith(word)]["correct"].mean()
        print(f"  {word:<20} {count:>3} errors   acc={correct_with_word:.0%} on all with this word")

    if "source" in df.columns:
        print(f"\n{'─'*60}")
        print("ERROR RATE BY SOURCE")
        print(f"{'─'*60}")
        for src, grp in df.groupby("source"):
            err_rate = 1 - grp["correct"].mean()
            if err_rate > 0:
                print(f"  {src:<35} {err_rate:.0%} ({grp['correct'].eq(False).sum()}/{len(grp)})")

    print(f"\n{'─'*60}")
    print("CONFIDENCE DISTRIBUTION")
    print(f"{'─'*60}")
    for label_name, is_error in [("correct", True), ("errors", False)]:
        subset = df[df["correct"] == is_error]["confidence"]
        if len(subset):
            print(f"  {label_name:<10}  mean={subset.mean():.2f}  min={subset.min():.2f}  max={subset.max():.2f}")

    return errors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["uk", "en", "both"], default="uk")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--save", default=None,
                        help="Save misclassified examples to CSV (e.g. results/errors_uk.csv)")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint(args.dataset)
    if checkpoint is None:
        print(f"No checkpoint with weights found for dataset={args.dataset}")
        sys.exit(1)

    print(f"Checkpoint: {checkpoint}")
    data_dir = "both" if args.dataset == "both" else f"data/splits/{args.dataset}"
    test_df = load_split(data_dir, "test")

    results = predict_all(checkpoint, test_df, args.max_length)
    errors  = analyze(results)

    if args.save:
        errors.to_csv(args.save, index=False)
        print(f"\nSaved errors → {args.save}")

if __name__ == "__main__":
    main()
