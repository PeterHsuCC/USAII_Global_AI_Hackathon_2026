"""Training script for the Online Grooming Head ablation (Section 6).

Trains and compares three variants on the same PAN12 training/validation
samples, with the official PAN12 test corpus reserved for final
evaluation (Section 6/19.1):

  A. Text-only:    S_g(t) = sigmoid(W_z z_t + b)
  B. Text + Rules:  S_g(t) = sigmoid(W[z_t; Q_t] + b)
  C. Linear Hybrid: S_g(t) = sigmoid(W_g[z_t; L_t; Q_t] + b_g)

All three reuse the same GroomingHead architecture (src/risk_detection/
model/heads/grooming_head.py), instantiated with safety_dim=0/5/11
respectively, fed the empty/rule-only/rule+LLM slice of the safety
feature vector -- not three separate model classes, so a real ablation
comparison rather than three independently-designed heads.

The conversation_label fed to all three variants is the predator-presence
-derived weak label from risk_detection.data.pan12 (conversation_label = 1
iff PAN12 lists any author in the conversation as a predator;
label_source = "pan12_predator_identity") -- not a directly-annotated
grooming judgment (Section 6).

Variant C calls the LLM safety signal extractor (Section 3.1), which
costs real API calls. Results are cached to disk per conversation
(--llm-cache) so re-running, or running A/B/C back to back, doesn't
re-pay for the same conversation twice. Variant C requires an
ANTHROPIC_API_KEY (see LLMSafetySignalExtractor) unless you pass a stub
extractor in your own code.

Data prerequisites: a generated PAN12 datapack under --pan12-dir (see
data/README.md and data/PAN12/create_datapack.py -- this script does not
generate that itself, it only consumes it via risk_detection.data.pan12).
Optionally also a VTPAN datapack under --vtpan-dir for the Section 19.1
supplementary evaluation, and/or --identity-disjoint for the Section 19.2
supplementary evaluation (excludes PAN12's officially-overlapping predator
ids from training; the official test corpus is left unchanged either way).

Like scripts/train_cyberbullying.py: do NOT reduce --epochs to save
compute. --max-conversations exists only for a quick pre-flight smoke
test (e.g. with a handful of conversations) before committing to a real
run over the full PAN12 corpus, which is large (66,927 train / 155,128
test conversations) and will take a while.
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from risk_detection.conversation import ConversationWindow
from risk_detection.data.pan12 import (
    ConversationSample,
    default_overlapping_predator_ids,
    default_pan12_dir,
    filter_identity_disjoint,
    full_conversation_samples,
    load_split,
)
from risk_detection.model import ConversationEncoder, GroomingHead, MessageEncoder, binary_review_loss
from risk_detection.signals.llm_safety import LLMSafetySignalExtractor
from risk_detection.signals.rules import RuleSignalExtractor

VARIANTS = ("A", "B", "C")
SAFETY_DIM_BY_VARIANT = {"A": 0, "B": 5, "C": 11}


class CachedLLMSafetyExtractor:
    """Wraps LLMSafetySignalExtractor with a JSON disk cache keyed by
    "<conversation_id>:<window_start>-<window_end>", so the same L_t
    values are reused across repeated runs and across the A/B/C
    comparison (Section 6: "LLM features are extracted once and cached
    offline")."""

    def __init__(self, cache_path: Path, extractor: LLMSafetySignalExtractor | None = None):
        self.cache_path = cache_path
        self.extractor = extractor or LLMSafetySignalExtractor()
        self._cache: dict[str, list[float]] = {}
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        self._dirty = False

    def get(self, key: str, window: ConversationWindow) -> list[float]:
        if key not in self._cache:
            self._cache[key] = self.extractor.extract(window).to_vector()
            self._dirty = True
        return self._cache[key]

    def save(self) -> None:
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)
        self._dirty = False


def build_safety_features(
    variant: str,
    sample: ConversationSample,
    rule_extractor: RuleSignalExtractor,
    llm_cache: CachedLLMSafetyExtractor | None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if variant == "A":
        return torch.zeros(0, device=device)

    window = sample.to_window()
    rule_vec = rule_extractor.extract(window).to_vector()  # 5-dim, [Q_t]
    if variant == "B":
        return torch.tensor(rule_vec, dtype=torch.float32, device=device)

    # Variant C: [L_t ; Q_t], 11-dim
    assert llm_cache is not None
    cache_key = f"{sample.conversation_id}:{sample.window_start}-{sample.window_end}"
    llm_vec = llm_cache.get(cache_key, window)
    return torch.tensor(llm_vec + rule_vec, dtype=torch.float32, device=device)


def encode_sample(
    sample: ConversationSample,
    message_encoder: MessageEncoder,
    conversation_encoder: ConversationEncoder,
) -> torch.Tensor:
    """Returns z_t in R^d_z for one conversation sample."""
    window = sample.to_window()
    h = message_encoder.encode_window(window)
    z, _alpha = conversation_encoder.encode(h)
    return z


def save_artifact(obj, output_dir: Path, name: str, timestamp: str, as_json: bool = False) -> None:
    """Same two-tier save convention as scripts/train_cyberbullying.py:
    an overwritable "latest" copy plus a per-run history copy."""
    suffix = "json" if as_json else "pt"
    history_dir = output_dir / name
    history_dir.mkdir(parents=True, exist_ok=True)

    latest_path = output_dir / f"{name}.{suffix}"
    history_path = history_dir / f"{name}_{timestamp}.{suffix}"

    if as_json:
        for path in (latest_path, history_path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
    else:
        torch.save(obj, latest_path)
        torch.save(obj, history_path)


@dataclass
class EvalMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    n: int
    n_positive: int


def evaluate(
    samples: list[ConversationSample],
    variant: str,
    message_encoder: MessageEncoder,
    conversation_encoder: ConversationEncoder,
    head: GroomingHead,
    rule_extractor: RuleSignalExtractor,
    llm_cache: CachedLLMSafetyExtractor | None,
    device: torch.device,
) -> EvalMetrics:
    message_encoder.eval()
    conversation_encoder.eval()
    head.eval()

    total_loss = 0.0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    correct = 0
    n_positive = 0

    with torch.no_grad():
        for sample in samples:
            z = encode_sample(sample, message_encoder, conversation_encoder)
            safety = build_safety_features(variant, sample, rule_extractor, llm_cache, device=device)
            s_g, _b_t = head(z, safety)

            label = torch.tensor(float(sample.label), device=device)
            total_loss += binary_review_loss(s_g, label).item()

            pred = s_g.item() >= 0.5
            true = sample.label == 1
            correct += int(pred == true)
            n_positive += int(true)
            true_positive += int(pred and true)
            false_positive += int(pred and not true)
            false_negative += int((not pred) and true)

    n = len(samples)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else float("nan")
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else float("nan")
    return EvalMetrics(
        loss=total_loss / n if n else float("nan"),
        accuracy=correct / n if n else float("nan"),
        precision=precision,
        recall=recall,
        n=n,
        n_positive=n_positive,
    )


def train_one_variant(
    variant: str,
    train_samples: list[ConversationSample],
    val_samples: list[ConversationSample],
    test_samples: list[ConversationSample],
    vtpan_samples: list[ConversationSample] | None,
    args: argparse.Namespace,
    run_timestamp: str,
) -> None:
    print(f"\n=== Variant {variant} (safety_dim={SAFETY_DIM_BY_VARIANT[variant]}) ===")

    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    message_encoder = MessageEncoder(model_name=args.model_name, max_length=args.max_length)
    conversation_encoder = ConversationEncoder(d=message_encoder.d)
    head = GroomingHead(d_z=message_encoder.d, safety_dim=SAFETY_DIM_BY_VARIANT[variant])
    message_encoder.to(device)
    conversation_encoder.to(device)
    head.to(device)

    rule_extractor = RuleSignalExtractor()
    llm_cache = None
    if variant == "C":
        llm_cache = CachedLLMSafetyExtractor(args.llm_cache)

    params = [
        p
        for p in list(message_encoder.parameters())
        + list(conversation_encoder.parameters())
        + list(head.parameters())
        if p.requires_grad
    ]
    print(f"Trainable parameters: {sum(p.numel() for p in params):,}")
    optimizer = optim.AdamW(params, lr=args.lr)

    best_val_loss = float("inf")
    checkpoint_saved = False
    rng = random.Random(args.seed)

    for epoch in range(args.epochs):
        message_encoder.train()
        conversation_encoder.train()
        head.train()

        order = list(range(len(train_samples)))
        rng.shuffle(order)

        running_loss = 0.0
        seen = 0
        optimizer.zero_grad()
        for step, idx in enumerate(order):
            sample = train_samples[idx]
            z = encode_sample(sample, message_encoder, conversation_encoder)
            safety = build_safety_features(variant, sample, rule_extractor, llm_cache, device=device)
            s_g, _b_t = head(z, safety)

            label = torch.tensor(float(sample.label), device=device)
            loss = binary_review_loss(s_g, label) / args.batch_size
            loss.backward()

            running_loss += loss.item() * args.batch_size
            seen += 1

            if (step + 1) % args.batch_size == 0 or step == len(order) - 1:
                clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(order)}: running_loss={running_loss / max(seen, 1):.4f}")

        val_metrics = evaluate(
            val_samples, variant, message_encoder, conversation_encoder, head, rule_extractor, llm_cache, device
        )
        print(
            f"epoch {epoch} [{variant}]: train_loss={running_loss / max(seen, 1):.4f} "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.3f} "
            f"val_precision={val_metrics.precision:.3f} val_recall={val_metrics.recall:.3f} "
            f"(n={val_metrics.n}, n_positive={val_metrics.n_positive})"
        )

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            checkpoint_saved = True
            save_artifact(message_encoder.state_dict(), args.output_dir, f"grooming_message_encoder_{variant}", run_timestamp)
            save_artifact(conversation_encoder.state_dict(), args.output_dir, f"grooming_conversation_encoder_{variant}", run_timestamp)
            save_artifact(head.state_dict(), args.output_dir, f"grooming_head_{variant}", run_timestamp)
            print(f"  -> new best val_loss, saved checkpoint (run {run_timestamp}) to {args.output_dir}")

        if llm_cache is not None:
            llm_cache.save()

    if checkpoint_saved:
        message_encoder.load_state_dict(
            torch.load(args.output_dir / f"grooming_message_encoder_{variant}.pt", map_location=device)
        )
        conversation_encoder.load_state_dict(
            torch.load(args.output_dir / f"grooming_conversation_encoder_{variant}.pt", map_location=device)
        )
        head.load_state_dict(
            torch.load(args.output_dir / f"grooming_head_{variant}.pt", map_location=device)
        )
        print(f"  -> reloaded best val_loss checkpoint (val_loss={best_val_loss:.4f}) before final test evaluation")

    test_metrics = evaluate(
        test_samples, variant, message_encoder, conversation_encoder, head, rule_extractor, llm_cache, device
    )
    print(
        f"[{variant}] official PAN12 test: loss={test_metrics.loss:.4f} acc={test_metrics.accuracy:.3f} "
        f"precision={test_metrics.precision:.3f} recall={test_metrics.recall:.3f} "
        f"(n={test_metrics.n}, n_positive={test_metrics.n_positive})"
    )
    if vtpan_samples is not None:
        vtpan_metrics = evaluate(
            vtpan_samples, variant, message_encoder, conversation_encoder, head, rule_extractor, llm_cache, device
        )
        print(
            f"[{variant}] VTPAN supplementary eval: loss={vtpan_metrics.loss:.4f} acc={vtpan_metrics.accuracy:.3f} "
            f"precision={vtpan_metrics.precision:.3f} recall={vtpan_metrics.recall:.3f} "
            f"(n={vtpan_metrics.n}, n_positive={vtpan_metrics.n_positive})"
        )
    if llm_cache is not None:
        llm_cache.save()


def load_samples(
    dataset_dir: Path,
    split: str,
    max_conversations: int | None,
    datapack_id: str = "PAN12",
    excluded_author_ids: set[str] | None = None,
) -> list[ConversationSample]:
    conversations = load_split(dataset_dir, split, datapack_id=datapack_id)
    if excluded_author_ids:
        conversations = filter_identity_disjoint(conversations, excluded_author_ids)
    if max_conversations is not None:
        conversations = (c for i, c in enumerate(conversations) if i < max_conversations)
    return list(full_conversation_samples(conversations))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--variant",
        choices=(*VARIANTS, "all"),
        default="all",
        help="Which ablation variant to train; 'all' runs A, B, C in one invocation "
        "on identical data/splits/seed for a fair comparison (default).",
    )
    parser.add_argument("--pan12-dir", type=Path, default=default_pan12_dir())
    parser.add_argument(
        "--vtpan-dir",
        type=Path,
        default=None,
        help="Optional VTPAN datapack dir for the Section 19.1 supplementary evaluation",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Cap conversations per split, for a quick smoke test before a full run "
        "(PAN12 train/test have 66,927/155,128 conversations)",
    )
    parser.add_argument(
        "--identity-disjoint",
        action="store_true",
        help="Additionally exclude PAN12's officially-overlapping predator identities (Section 19.2; "
        "2 ids as of the 2012-05-01 release) from the training pool. The official test corpus is left "
        "unchanged, so this run's test metrics can be compared against the official-split run's to check "
        "whether the overlap was inflating apparent performance.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8, help="Gradient accumulation steps per optimizer update")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("trained_weights"))
    parser.add_argument("--llm-cache", type=Path, default=Path("trained_weights/llm_signal_cache.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    excluded_ids: set[str] = set()
    if args.identity_disjoint:
        excluded_ids = default_overlapping_predator_ids(args.pan12_dir)
        print(f"--identity-disjoint: excluding {len(excluded_ids)} overlapping predator id(s) from training: {sorted(excluded_ids)}")

    print(f"Loading PAN12 train split from {args.pan12_dir} ...")
    train_all = load_samples(args.pan12_dir, "train", args.max_conversations, excluded_author_ids=excluded_ids)
    print(f"  {len(train_all)} conversations" + (" (after identity-disjoint filtering)" if excluded_ids else ""))
    print(f"Loading PAN12 test split from {args.pan12_dir} ...")
    test_samples = load_samples(args.pan12_dir, "test", args.max_conversations)
    print(f"  {len(test_samples)} conversations")

    vtpan_samples = None
    if args.vtpan_dir is not None:
        print(f"Loading VTPAN test split from {args.vtpan_dir} ...")
        vtpan_samples = load_samples(args.vtpan_dir, "test", args.max_conversations, datapack_id="VTPAN")
        print(f"  {len(vtpan_samples)} conversations")

    rng = random.Random(args.seed)
    order = list(range(len(train_all)))
    rng.shuffle(order)
    split_idx = int(len(order) * (1 - args.val_fraction))
    train_samples = [train_all[i] for i in order[:split_idx]]
    val_samples = [train_all[i] for i in order[split_idx:]]
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}  (same split/seed reused for every variant)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    variants = VARIANTS if args.variant == "all" else (args.variant,)
    for variant in variants:
        train_one_variant(variant, train_samples, val_samples, test_samples, vtpan_samples, args, run_timestamp)

    print(f"\nDone. Run {run_timestamp}. Checkpoints in {args.output_dir}, named grooming_*_<variant>.pt")
    print("Compare variants: B-A isolates rule contribution, C-B isolates LLM-signal contribution (Section 6).")


if __name__ == "__main__":
    main()
