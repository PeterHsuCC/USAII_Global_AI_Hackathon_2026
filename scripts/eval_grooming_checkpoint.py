"""Re-evaluate a saved grooming checkpoint without retraining.

Runs of scripts/train_grooming.py from before the best-checkpoint-reload fix
printed test/VTPAN metrics computed from the last training epoch's in-memory
model, not the best-val-loss checkpoint actually written to --output-dir.
This script reloads that checkpoint from disk and re-runs the official PAN12
test (and optional VTPAN) evaluation against it, so those runs don't need to
be retrained to get metrics that match the saved checkpoint.

Example:
    python scripts/eval_grooming_checkpoint.py --variant B --vtpan-dir data/VTPAN
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from risk_detection.model import ConversationEncoder, GroomingHead, MessageEncoder
from risk_detection.signals.rules import RuleSignalExtractor
from train_grooming import (
    VARIANTS,
    CachedLLMSafetyExtractor,
    SAFETY_DIM_BY_VARIANT,
    default_pan12_dir,
    evaluate,
    load_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--pan12-dir", type=Path, default=default_pan12_dir())
    parser.add_argument("--vtpan-dir", type=Path, default=None)
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("trained_weights"))
    parser.add_argument("--llm-cache", type=Path, default=Path("trained_weights/llm_signal_cache.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    variant = args.variant

    message_encoder = MessageEncoder(model_name=args.model_name, max_length=args.max_length)
    conversation_encoder = ConversationEncoder(d=message_encoder.d)
    head = GroomingHead(d_z=message_encoder.d, safety_dim=SAFETY_DIM_BY_VARIANT[variant])

    message_encoder.load_state_dict(
        torch.load(args.output_dir / f"grooming_message_encoder_{variant}.pt", map_location=device)
    )
    conversation_encoder.load_state_dict(
        torch.load(args.output_dir / f"grooming_conversation_encoder_{variant}.pt", map_location=device)
    )
    head.load_state_dict(
        torch.load(args.output_dir / f"grooming_head_{variant}.pt", map_location=device)
    )
    message_encoder.to(device)
    conversation_encoder.to(device)
    head.to(device)

    rule_extractor = RuleSignalExtractor()
    llm_cache = CachedLLMSafetyExtractor(args.llm_cache) if variant == "C" else None

    print(f"Loading PAN12 test split from {args.pan12_dir} ...")
    test_samples = load_samples(args.pan12_dir, "test", args.max_conversations)
    print(f"  {len(test_samples)} conversations")

    test_metrics = evaluate(
        test_samples, variant, message_encoder, conversation_encoder, head, rule_extractor, llm_cache, device
    )
    print(
        f"[{variant}] official PAN12 test (best-val checkpoint): loss={test_metrics.loss:.4f} "
        f"acc={test_metrics.accuracy:.3f} precision={test_metrics.precision:.3f} recall={test_metrics.recall:.3f} "
        f"(n={test_metrics.n}, n_positive={test_metrics.n_positive})"
    )

    if args.vtpan_dir is not None:
        print(f"Loading VTPAN test split from {args.vtpan_dir} ...")
        vtpan_samples = load_samples(args.vtpan_dir, "test", args.max_conversations, datapack_id="VTPAN")
        print(f"  {len(vtpan_samples)} conversations")
        vtpan_metrics = evaluate(
            vtpan_samples, variant, message_encoder, conversation_encoder, head, rule_extractor, llm_cache, device
        )
        print(
            f"[{variant}] VTPAN supplementary eval (best-val checkpoint): loss={vtpan_metrics.loss:.4f} "
            f"acc={vtpan_metrics.accuracy:.3f} precision={vtpan_metrics.precision:.3f} recall={vtpan_metrics.recall:.3f} "
            f"(n={vtpan_metrics.n}, n_positive={vtpan_metrics.n_positive})"
        )

    if llm_cache is not None:
        llm_cache.save()


if __name__ == "__main__":
    main()
