"""Driver for Stage A fine-tune. Thin wrapper around the training module."""
from __future__ import annotations

import sys

from spiking_ternary_lm.training.stage_a_finetune import main

if __name__ == "__main__":
    sys.exit(main())
