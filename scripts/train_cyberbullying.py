"""Real-data training script for the Cyberbullying Head (Section 5).

Trains Stage 1 (single-message training) ONLY -- Stage 2 (conversation
context) needs labeled multi-turn conversations, which none of the data
sources below provide, so it is deliberately skipped here.

Data sources you provide locally (see --help for the exact flags):

  - Cyberbullying Classification (Kaggle, andrewmvd) -- required. Real
    labeled tweets across 6 categories: not_cyberbullying, gender,
    religion, age, ethnicity, other_cyberbullying. Download the CSV
    yourself (Kaggle login required) and point --cyberbullying-csv at it.
    https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

  - DailyDialog (HuggingFace `datasets`) -- optional, via --use-dailydialog.
    Adds benign everyday-conversation negatives, since the Kaggle data is
    Twitter-only and may not generalize to ordinary chat text. NOTE: this
    is a script-based dataset on the HF Hub; if newer `datasets` releases
    have dropped script support by the time you run this, the call below
    will fail with a clear error -- paste it back if so, and we'll either
    pin an older `datasets` version or swap in a parquet-format mirror.

This is a REAL training run, not an illustrative demo: it loads a real
pretrained encoder and fine-tunes it together with the Cyberbullying
Head. Defaults are tuned to keep GPU/compute load LOW while still
producing a usable checkpoint:

  - --model-name defaults to distilbert-base-uncased (Section 4.1
    explicitly allows "BERT or DistilBERT") -- 6 layers instead of BERT's
    12, ~40% fewer parameters, ~97% of BERT's downstream accuracy in the
    original DistilBERT paper's benchmarks.
  - --max-length defaults to 64 (tweets rarely need 128 tokens).
  - --batch-size defaults to 16.
  - --freeze-encoder-layers defaults to 0 (no freezing). Raising it
    freezes the bottom N transformer layers so backprop/optimizer state
    skip them, cutting compute further -- but unlike the three knobs
    above, it CAN cost you accuracy if pushed too far, so it's opt-in:
    run once at 0 to get a baseline val_accuracy, then try e.g. 3 and
    compare before trusting the frozen run's checkpoint.

Do NOT reduce --epochs or the amount of data to save compute -- that's
the one knob that directly risks an undertrained, unusable checkpoint
rather than just a slower one. --max-examples exists only for a quick
pre-flight smoke test, not as a real-run setting.

GPU utilization is not something this script can pin to an exact
percentage -- that's an emergent result of model/batch size vs. GPU
capacity, not a settable knob. Watch `nvidia-smi -l 1` in another
terminal while a few steps run to see where you land.
"""

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from risk_detection.model import CyberbullyingHead, MessageEncoder, cyberbullying_loss

NOT_BULLYING_LABEL_DEFAULT = "not_cyberbullying"


@dataclass
class LabeledText:
    text: str
    label: str


def load_kaggle_csv(path: Path, text_column: str, label_column: str) -> list[LabeledText]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_column)
            label = row.get(label_column)
            if text and label:
                rows.append(LabeledText(text=text, label=label))
    return rows


def load_dailydialog_negatives(not_bullying_label: str, max_count: int | None) -> list[LabeledText]:
    from datasets import load_dataset

    dataset = load_dataset("daily_dialog", split="train", trust_remote_code=True)
    rows = []
    for item in dataset:
        for utterance in item["dialog"]:
            utterance = utterance.strip()
            if utterance:
                rows.append(LabeledText(text=utterance, label=not_bullying_label))
            if max_count is not None and len(rows) >= max_count:
                return rows
    return rows


def freeze_encoder_layers(message_encoder: MessageEncoder, num_layers: int) -> None:
    """Freezes the embeddings and the bottom `num_layers` transformer
    layers of the underlying BERT/DistilBERT encoder, leaving only the
    top layers (and the downstream Cyberbullying Head) trainable. This is
    standard partial fine-tuning: it cuts backward-pass compute and
    optimizer memory for the frozen layers without retraining the whole
    network, at some (untested here) risk to accuracy if num_layers is
    pushed too high. No-op when num_layers <= 0.
    """
    if num_layers <= 0:
        return

    encoder = message_encoder.encoder
    for param in encoder.embeddings.parameters():
        param.requires_grad_(False)

    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        layers = encoder.encoder.layer  # BERT-style: model.encoder.layer
    elif hasattr(encoder, "transformer") and hasattr(encoder.transformer, "layer"):
        layers = encoder.transformer.layer  # DistilBERT-style: model.transformer.layer
    else:
        raise ValueError(
            "Don't know how to locate transformer layers on this encoder type "
            f"({type(encoder).__name__}); pass --freeze-encoder-layers 0 to skip freezing."
        )

    for layer in layers[:num_layers]:
        for param in layer.parameters():
            param.requires_grad_(False)


def save_artifact(obj, output_dir: Path, name: str, timestamp: str, as_json: bool = False) -> None:
    """Saves `obj` two ways:

      - output_dir/<name>.<ext>           -- overwritable "latest" copy,
        always the most recent best checkpoint from any run. Point
        inference code here.
      - output_dir/<name>/<name>_<timestamp>.<ext> -- a copy that is never
        overwritten, one per run (same timestamp reused across epochs
        within one run, so only the run's final best survives there too).

    timestamp should be computed once per script invocation (not per
    epoch) so repeated improving epochs within one run overwrite the same
    history file, while separate runs get separate history files.
    """
    suffix = "json" if as_json else "pt"
    history_dir = output_dir / name
    history_dir.mkdir(parents=True, exist_ok=True)

    latest_path = output_dir / f"{name}.{suffix}"
    history_path = history_dir / f"{name}_{timestamp}.{suffix}"

    # Each write goes to a temp file and is atomically swapped into place, so
    # a crash mid-save can't leave `latest_path` -- "point inference code
    # here", per the docstring above -- truncated or corrupted.
    for path in (latest_path, history_path):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        if as_json:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
        else:
            torch.save(obj, tmp_path)
        os.replace(tmp_path, path)


def build_label_mapping(rows: list[LabeledText], not_bullying_label: str) -> dict[str, int]:
    """not_bullying_label is pinned to index 0 (= CyberbullyingHead's
    non_bullying_index); every other label found in the data gets the
    next index, alphabetically, so the mapping is reproducible."""
    other_labels = sorted({r.label for r in rows} - {not_bullying_label})
    mapping = {not_bullying_label: 0}
    for i, label in enumerate(other_labels, start=1):
        mapping[label] = i
    return mapping


class CyberbullyingDataset(Dataset):
    def __init__(self, rows: list[LabeledText], label_to_index: dict[str, int]):
        self.rows = rows
        self.label_to_index = label_to_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        row = self.rows[idx]
        return row.text, self.label_to_index[row.label]


def collate(batch: list[tuple[str, int]]) -> tuple[list[str], torch.Tensor]:
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.long)


def evaluate(
    message_encoder: MessageEncoder,
    head: CyberbullyingHead,
    loader: DataLoader,
    device: torch.device,
    non_bullying_index: int,
) -> dict[str, float]:
    message_encoder.eval()
    head.eval()
    correct = 0
    binary_correct = 0
    true_bullying = 0
    recalled_bullying = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for texts, labels in loader:
            labels = labels.to(device)
            speakers = ["user"] * len(texts)
            h = message_encoder(speakers, texts)
            p = head.forward_stage1(h)
            loss = cyberbullying_loss(p, labels).mean()
            total_loss += loss.item() * len(texts)

            preds = p.argmax(dim=-1)
            correct += (preds == labels).sum().item()

            true_is_bullying = labels != non_bullying_index
            pred_is_bullying = preds != non_bullying_index
            binary_correct += (pred_is_bullying == true_is_bullying).sum().item()
            true_bullying += true_is_bullying.sum().item()
            recalled_bullying += (pred_is_bullying & true_is_bullying).sum().item()

            total += len(texts)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "binary_accuracy": binary_correct / total,
        "bullying_recall": recalled_bullying / true_bullying if true_bullying else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cyberbullying-csv",
        type=Path,
        required=True,
        help="Path to the downloaded Kaggle cyberbullying_tweets.csv",
    )
    parser.add_argument("--text-column", default="tweet_text")
    parser.add_argument("--label-column", default="cyberbullying_type")
    parser.add_argument("--not-bullying-label", default=NOT_BULLYING_LABEL_DEFAULT)
    parser.add_argument(
        "--use-dailydialog",
        action="store_true",
        help="Also pull benign negative examples from DailyDialog (HF datasets)",
    )
    parser.add_argument("--dailydialog-negative-count", type=int, default=5000)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Cap total examples, for a quick smoke test before committing to a full run",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="Shared encoder checkpoint (Section 4.1 allows BERT or DistilBERT)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Tokenizer truncation length; tweets rarely need more than this",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--freeze-encoder-layers",
        type=int,
        default=0,
        help="Freeze the bottom N transformer layers (0 = fine-tune everything). "
        "Opt-in: get a baseline at 0 before trusting a frozen run's accuracy.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("trained_weights"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.cyberbullying_csv} ...")
    rows = load_kaggle_csv(args.cyberbullying_csv, args.text_column, args.label_column)
    print(f"  {len(rows)} labeled tweets loaded")

    if args.use_dailydialog:
        print("Loading DailyDialog negatives ...")
        rows += load_dailydialog_negatives(args.not_bullying_label, args.dailydialog_negative_count)
        print(f"  total rows after adding DailyDialog: {len(rows)}")

    random.shuffle(rows)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
        print(f"--max-examples set: using only {len(rows)} rows for a smoke test")

    label_to_index = build_label_mapping(rows, args.not_bullying_label)
    non_bullying_index = label_to_index[args.not_bullying_label]
    print(f"Label mapping ({len(label_to_index)} classes): {label_to_index}")

    split = int(len(rows) * (1 - args.val_fraction))
    train_rows, val_rows = rows[:split], rows[split:]
    train_loader = DataLoader(
        CyberbullyingDataset(train_rows, label_to_index),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        CyberbullyingDataset(val_rows, label_to_index),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    print(f"Train: {len(train_rows)}  Val: {len(val_rows)}")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    message_encoder = MessageEncoder(model_name=args.model_name, max_length=args.max_length)
    head = CyberbullyingHead(
        d=message_encoder.d,
        d_z=message_encoder.d,
        num_classes=len(label_to_index),
        non_bullying_index=non_bullying_index,
    )
    message_encoder.to(device)
    head.to(device)

    if args.freeze_encoder_layers > 0:
        freeze_encoder_layers(message_encoder, args.freeze_encoder_layers)
        print(f"Froze embeddings + bottom {args.freeze_encoder_layers} encoder layers")

    params = [p for p in list(message_encoder.parameters()) + list(head.parameters()) if p.requires_grad]
    num_trainable = sum(p.numel() for p in params)
    num_total = sum(p.numel() for p in message_encoder.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"Trainable parameters: {num_trainable:,} / {num_total:,}")
    optimizer = optim.AdamW(params, lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        message_encoder.train()
        head.train()
        running_loss = 0.0
        seen = 0
        for step, (texts, labels) in enumerate(train_loader):
            labels = labels.to(device)
            speakers = ["user"] * len(texts)

            optimizer.zero_grad()
            h = message_encoder(speakers, texts)
            p = head.forward_stage1(h)
            loss = cyberbullying_loss(p, labels).mean()
            loss.backward()
            clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(texts)
            seen += len(texts)
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)}: loss={loss.item():.4f}")

        val_metrics = evaluate(message_encoder, head, val_loader, device, non_bullying_index)
        print(
            f"epoch {epoch}: train_loss={running_loss / seen:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc(exact)={val_metrics['accuracy']:.3f} "
            f"val_acc(bullying-vs-not)={val_metrics['binary_accuracy']:.3f} "
            f"bullying_recall={val_metrics['bullying_recall']:.3f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_artifact(message_encoder.state_dict(), args.output_dir, "message_encoder", run_timestamp)
            save_artifact(head.state_dict(), args.output_dir, "cyberbullying_head", run_timestamp)
            save_artifact(label_to_index, args.output_dir, "label_mapping", run_timestamp, as_json=True)
            print(f"  -> new best val_loss, saved checkpoint (run {run_timestamp}) to {args.output_dir}")

    print(f"\nDone. Best val_loss={best_val_loss:.4f}. Run {run_timestamp}. Checkpoints in {args.output_dir}:")
    print(f"  {args.output_dir}/message_encoder.pt              (latest, overwritable)")
    print(f"  {args.output_dir}/message_encoder/message_encoder_{run_timestamp}.pt   (this run's copy)")
    print(f"  {args.output_dir}/cyberbullying_head.pt            (latest, overwritable)")
    print(f"  {args.output_dir}/cyberbullying_head/cyberbullying_head_{run_timestamp}.pt   (this run's copy)")
    print(f"  {args.output_dir}/label_mapping.json               (latest, overwritable)")
    print(f"  {args.output_dir}/label_mapping/label_mapping_{run_timestamp}.json   (this run's copy)")


if __name__ == "__main__":
    main()
