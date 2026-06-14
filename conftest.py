"""Make the repo root importable so `import recsys` works in tests without installing."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
