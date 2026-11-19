"""Import the previously validated W1 physics utilities without modifying the original example."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE.parent / "w1_pharmacy_carton"
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location(
    "pharmacy_reference_physics", HERE.parent / "demo/demo_w1_pharmacy_carton.py"
)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
