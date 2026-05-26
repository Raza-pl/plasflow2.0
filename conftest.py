"""Pytest configuration — add src/ to sys.path for editable installs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
