"""
Evaluate a saved model checkpoint on the test split.

Usage:
  python src/models/evaluate.py --checkpoint results/bert-multilingual/best --output results/bert-multilingual/test_metrics.json
"""
import argparse
import json
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import QueryDataset

LABELS = {0: "solution_oriented", 1: "learning_oriented"}

def main(checkpoint: str, output: str, max_length: int = 128, batch_size: int = 32,
         dataset: str = "uk"):
    from train import load_split
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}  |  Checkpoint: {checkpoint}  |  Dataset: {dataset}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()

    data_dir = "both" if dataset == "both" else f"data/splits/{dataset}"
    test_ds = QueryDataset(None, tokenizer, max_length, df=load_split(data_dir, "test"))
    loader = DataLoader(test_ds, batch_size=batch_size)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch).logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")

    report = classification_report(all_labels, all_preds, target_names=list(LABELS.values()))
    cm = confusion_matrix(all_labels, all_preds).tolist()

    metrics = {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "confusion_matrix": cm}

    print(f"\n{report}")
    print(f"Confusion matrix:\n{cm}")

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved → {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", choices=["uk", "en", "both"], default="uk")
    args = parser.parse_args()
    main(args.checkpoint, args.output, args.max_length, args.batch_size, args.dataset)
