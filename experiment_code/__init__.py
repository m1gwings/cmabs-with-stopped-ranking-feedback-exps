"""Experiment components; stable EWS identifiers preserve seeded streams."""

import sys
from . import algorithms, data, figures, metrics

# EWS includes these identifiers in its random-stream hashes.
for name, module in {
    "uif": algorithms, "baselines": algorithms, "stopped_data": data,
}.items():
    sys.modules[f"{__name__}.{name}"] = module
