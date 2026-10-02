"""Where the results, the datasets and the detector weights live.

No script in this repository hardcodes a machine path. Results and figures are
found relative to the repository. Datasets and weights come from environment
variables, so the same code runs unchanged on any machine:

    DATA_ROOT      holds coco/, nightowls/, llvip/LLVIP/ and kaist/
    ULTRA_WEIGHTS  holds the Ultralytics checkpoints (*.pt)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(FIGURES, exist_ok=True)


def data(*parts):
    root = os.environ.get("DATA_ROOT")
    if not root:
        raise SystemExit("Set DATA_ROOT to the directory that holds coco/, "
                         "nightowls/, llvip/ and kaist/.")
    return os.path.join(root, *parts)
