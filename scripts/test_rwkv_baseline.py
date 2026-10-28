"""Phase 0 deliverable: RWKV-4-169M baseline sanity.

Downloads `RWKV/rwkv-4-169m-pile` from HuggingFace and generates ~100 tokens
from a simple prompt to confirm the teacher model runs in pure FP inference
mode on the 5090. We use the HuggingFace `transformers` `RwkvForCausalLM`
class rather than BlinkDL's native runtime because it integrates cleanly with
the rest of our pipeline (tokenizer, generate, dtype handling).

Note: the tokenizer used here (`RWKV/rwkv-4-169m-pile`) is the original
RWKV-4-pile tokenizer. We will reuse the same tokenizer in Stage A so the
student inherits the teacher's vocabulary mapping with zero drift.
"""
from __future__ import annotations

import argparse
import sys
import time

from spiking_ternary_lm.utils.logging import get_logger

LOG = get_logger("rwkv_baseline")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="RWKV/rwkv-4-169m-pile",
        help="HF repo id for the teacher model.",
    )
    parser.add_argument(
        "--prompt", default="Once upon a time, in a small village by the sea,",
        help="Prompt to seed generation.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"FAIL: import error ({e}). Run `uv sync`.", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        args.dtype
    ]
    device = torch.device(args.device)

    LOG.info("loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    LOG.info("loading model: %s (dtype=%s)", args.model, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    LOG.info("model param count: %.2f M", n_params / 1e6)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    LOG.info("prompt tokens: %d", inputs.input_ids.shape[-1])

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    n_new = out.shape[-1] - inputs.input_ids.shape[-1]
    LOG.info("generated %d tokens in %.2fs (%.1f tok/s)", n_new, dt, n_new / max(dt, 1e-6))

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("=" * 60)
    print(text)
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
