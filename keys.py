"""The DeepInfra API key: $DEEPINFRA_API_KEY if set, else the git-ignored file .deepinfra_key next to this file.
Never write the key into a tracked file: the repository is public."""
import os
from pathlib import Path


def deepinfra_key():
    if os.environ.get("DEEPINFRA_API_KEY"):
        return os.environ["DEEPINFRA_API_KEY"]
    path = Path(__file__).with_name(".deepinfra_key")
    if path.exists():
        return path.read_text().strip()
    raise SystemExit("Set DEEPINFRA_API_KEY, or put the key in .deepinfra_key (git-ignored)")
