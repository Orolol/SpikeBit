"""Driver for Stage C from-scratch training. Thin wrapper."""
from __future__ import annotations

import sys

from spiking_ternary_lm.training.stage_c_scratch import main

if __name__ == "__main__":
    sys.exit(main())
