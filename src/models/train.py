import argparse
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

LABELS = {0: "solution_oriented", 1: "learning_oriented"}
CONFIG_PATH = Path("configs/training.yaml")

def load_split(data_dir: Path | str, split: str) -> pd.DataFrame:
    """Load train/val/test split; for 'both' concatenate uk + en."""
    if str(data_dir) == "both":
        uk = pd.read_csv(Path("data/splits/uk") / f"{split}.csv")
        en = pd.read_csv(Path("data/splits/en") / f"{split}.csv")
        return pd.concat([uk, en], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return pd.read_csv(Path(data_dir) / f"{split}.csv")

class QueryDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int, n_samples: int | None = None,
                 df: pd.DataFrame | None = None):
        df = pd.read_csv(path) if df is None else df
        if n_samples is not None:
            per_class = n_samples // 2
            df = pd.concat([
                df[df["label"] == lbl].sample(min(per_class, (df["label"] == lbl).sum()), random_state=42)
                for lbl in [0, 1]
            ]).sample(frac=1, random_state=42).reset_index(drop=True)
        self.labels = df["label"].tolist()
        self.encodings = tokenizer(
            df["text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

def evaluate(model, loader, device, loss_fn=None):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())
            if loss_fn is not None:
                total_loss += loss_fn(outputs.logits, batch["labels"]).item()
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    result = {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
    if loss_fn is not None:
        result["loss"] = total_loss / len(loader)
    return result

def main(model_name: str, output_dir: str, config: dict,
         max_length: int | None = None,
         lr: float | None = None,
         train_size: int | None = None,
         dataset: str = "uk",
         save_checkpoint: bool = True):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tc = config["training"]
    dc = config["data"]

    max_length  = max_length  or dc["max_length"]
    lr          = lr          or tc["learning_rate"]

    data_dir = "both" if dataset == "both" else f"data/splits/{dataset}"
    print(f"Device: {device}  |  Model: {model_name}  |  Dataset: {dataset}")
    print(f"max_length={max_length}  lr={lr}  train_size={train_size or 'full'}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device).float()

    train_ds = QueryDataset(None, tokenizer, max_length, n_samples=train_size,
                            df=load_split(data_dir, "train"))
    val_ds   = QueryDataset(None, tokenizer, max_length,
                            df=load_split(data_dir, "val"))

    if tc.get("class_weights", False):
        train_labels = np.array([train_ds.labels[i] for i in range(len(train_ds))])
        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_labels)
        class_weights = torch.tensor(cw, dtype=torch.float).to(device)
        print(f"Class weights: {cw.tolist()}")
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    batch_size = tc["batch_size"]
    if "large" in model_name.lower():
        divisor = 4 if device.type == "mps" else 2
        batch_size = max(1, tc["batch_size"] // divisor)
        lr = lr * (batch_size / tc["batch_size"])
        print(f"Large model — batch_size={batch_size}, scaled lr={lr:.2e}")
    elif max_length >= 512:
        batch_size = max(1, tc["batch_size"] // 2)
        lr = lr * (batch_size / tc["batch_size"])
        print(f"max_length≥512 — batch_size={batch_size}, scaled lr={lr:.2e}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tc["weight_decay"])
    total_steps = len(train_loader) * tc["num_epochs"]
    warmup_steps = int(total_steps * tc["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1, patience_count = 0.0, 0
    history = []

    for epoch in range(1, tc["num_epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"],
                          attention_mask=batch["attention_mask"]).logits
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        print(f"Epoch {epoch}/{tc['num_epochs']}  "
              f"train_loss={train_loss:.4f}  val_loss={val_metrics.get('loss', 0):.4f}  "
              f"val_acc={val_metrics['accuracy']:.4f}  val_f1={val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_count = 0
            if save_checkpoint:
                model.save_pretrained(out / "best")
                tokenizer.save_pretrained(out / "best")
            print(f"  -> saved best (f1={best_f1:.4f})")
        else:
            patience_count += 1
            if patience_count >= tc["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val F1: {best_f1:.4f}  -> {out / 'best'}")
    return best_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-multilingual-cased")
    parser.add_argument("--output", default="results/bert-multilingual")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Override data.max_length from config")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override training.learning_rate from config")
    parser.add_argument("--train_size", type=int, default=None,
                        help="Subsample N training examples (stratified)")
    parser.add_argument("--dataset", choices=["uk", "en", "both"], default="uk",
                        help="Which dataset splits to use")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    main(args.model, args.output, config,
         max_length=args.max_length, lr=args.lr, train_size=args.train_size,
         dataset=args.dataset)
