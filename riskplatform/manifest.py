"""Run manifest: what code, data, config and library versions produced a set of artifacts."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import platform
import subprocess
from dataclasses import asdict

import pandas as pd

from . import __version__
from .config import COST, GEN, ROOT, SPLIT

PACKAGES = ["pandas", "numpy", "scikit-learn", "sqlalchemy", "pyarrow", "joblib", "fastapi", "streamlit"]


def data_fingerprint(tx: pd.DataFrame) -> str:
    """Order-independent hash of the transaction table content (not of the database file)."""
    t = tx.sort_values("txn_id").reset_index(drop=True)
    h = pd.util.hash_pandas_object(t, index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10, check=True).stdout.strip() or None
    except Exception:
        return None


def build_manifest(tx: pd.DataFrame, seed: int, headline: dict) -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    cfg = {"gen": asdict(GEN), "split": asdict(SPLIT), "cost": asdict(COST), "seed": seed}
    return {"project_version": __version__, "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
            "python": platform.python_version(), "platform": platform.platform(), "packages": versions,
            "n_transactions": int(len(tx)), "data_sha256": data_fingerprint(tx),
            "config_sha256": hashlib.sha256(repr(sorted(cfg.items())).encode()).hexdigest(), "config": cfg,
            "headline_synthetic_metrics": headline}
