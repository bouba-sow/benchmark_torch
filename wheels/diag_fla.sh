#!/bin/bash
set -euo pipefail
source /lustre/fswork/projects/rech/ldh/una68ug/benchmark/.venv/bin/activate
python -c "
import fla
print('fla imported OK from', fla.__file__)
from fla.layers.mamba2 import Mamba2 as FlaMamba2
print('fla.layers.mamba2.Mamba2 OK')
"
