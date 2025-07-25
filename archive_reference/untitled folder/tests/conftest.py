# tests/conftest.py
import os
import sys

# tests/
# ├── conftest.py
# ├── src/
# │   ├── data_preparation.py
# │   └── ...
# └── test_*.py

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH     = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
