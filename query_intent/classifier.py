from __future__ import annotations

import torch
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

INTENT_NAMES = {0: "solution_oriented", 1: "learning_oriented"}
INTENT_UA    = {0: "орієнтований на рішення", 1: "орієнтований на навчання"}

@dataclass
class IntentResult:
    label:      int
    intent:     str
    intent_ua:  str
    confidence: float

    def __repr__(self):
        bar = "█" * int(self.confidence * 20)
        return f"IntentResult(intent={self.intent!r}, confidence={self.confidence:.0%}) {bar}"

class QueryClassifier:

    def __init__(self, checkpoint: str | Path, device: str | None = None,
                 max_length: int = 256):
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.max_length = max_length
        self.tokenizer  = AutoTokenizer.from_pretrained(checkpoint)
        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(checkpoint)
            .to(self.device)
            .eval()
        )

    @classmethod
    def from_best(cls, results_dir: str | Path = "results",
                  dataset: str = "uk", **kwargs) -> "QueryClassifier":
        checkpoint = find_best_checkpoint(results_dir, dataset)
        if checkpoint is None:
            raise FileNotFoundError(
                f"No trained model found for dataset={dataset!r} in {results_dir}. "
                "Run training first: bash run_all.sh"
            )
        return cls(checkpoint, **kwargs)

    def __call__(self, text: str) -> IntentResult:
        return self.predict(text)

    def predict(self, text: str) -> IntentResult:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[IntentResult]:
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs  = torch.softmax(logits, dim=-1).cpu()
            labels = probs.argmax(dim=-1).tolist()
            confs  = probs.max(dim=-1).values.tolist()

        return [
            IntentResult(
                label=lbl,
                intent=INTENT_NAMES[lbl],
                intent_ua=INTENT_UA[lbl],
                confidence=round(conf, 4),
            )
            for lbl, conf in zip(labels, confs)
        ]

def find_best_checkpoint(results_dir: str | Path = "results",
                         dataset: str = "uk") -> Path | None:
    """Return path to best available checkpoint (by test F1) from results/summary.csv.

    Prefers full-dataset runs; falls back to any run with actual model weights.
    """
    summary = Path(results_dir) / "summary.csv"
    if not summary.exists():
        return None
    df = pd.read_csv(summary)
    df = df[df["dataset"] == dataset].sort_values("test_f1", ascending=False)
    if df.empty:
        return None
    for _, row in df.iterrows():
        ckpt = Path(row["run_dir"]) / "best"
        if _has_weights(ckpt):
            return ckpt
    return None

def _has_weights(path: Path) -> bool:
    return any(
        (path / f).exists()
        for f in ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"]
    )
