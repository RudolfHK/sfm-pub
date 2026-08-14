"""
Make the repository root importable so `import sfm` works regardless of the
directory pytest is invoked from and of whether the package is installed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
