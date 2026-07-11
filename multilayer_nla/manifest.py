"""Run manifest (plan §12.6) — record provenance before any reportable run.

A flat dict of everything needed to reproduce a number and to detect silent
configuration drift later: code commit, model/tokenizer identity, layer triplet,
seeds, prompt/marker (Stage 2+), and package versions. Both the extractor and
the headroom probe stamp one of these into their output sidecar / results JSON.

Deliberately dependency-light (stdlib only) so it imports anywhere — including
the numpy-only test environment — without pulling torch/pyarrow.
"""

from __future__ import annotations

import datetime
import hashlib
import subprocess
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Packages whose versions pin numerical behavior. Recorded best-effort —
# absence is noted, never fatal (the probe runs with numpy alone).
_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "numpy",
    "pyarrow",
    "datasets",
    "peft",
    "safetensors",
)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT,
            capture_output=True, text=True, check=False, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version
    out: dict[str, str | None] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def tokenizer_hash(tokenizer: Any) -> str | None:
    """Stable hash of a tokenizer's vocabulary + special tokens.

    Catches a silent tokenizer swap/version drift (the failure mode CLAUDE.md
    warns about: marker token ID drifts -> injection lands wrong -> CJK output).
    Hashes the sorted (token, id) vocab pairs plus the special-token map, so two
    tokenizers that encode identically hash identically regardless of file
    layout.
    """
    if tokenizer is None:
        return None
    try:
        vocab = tokenizer.get_vocab()  # {token_str: id}
    except Exception:
        return None
    h = hashlib.sha256()
    for tok, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
        h.update(f"{tid}\t{tok}\n".encode("utf-8", "surrogatepass"))
    special = getattr(tokenizer, "all_special_tokens", None)
    if special:
        h.update(("\x00".join(map(str, special))).encode("utf-8", "surrogatepass"))
    return h.hexdigest()[:16]


def build_manifest(stage: str, extra: dict[str, Any] | None = None,
                   tokenizer: Any = None) -> dict[str, Any]:
    """Assemble a run manifest. `extra` overrides/augments the common fields.

    Common fields (plan §12.6): commit, dirty flag, timestamp, package versions,
    and (when a tokenizer is passed) its content hash. Stage-specific fields —
    layer triplet, corpus revision, seeds, prompt template, marker IDs, LoRA
    config — are passed in `extra` by the caller.
    """
    manifest: dict[str, Any] = {
        "stage": stage,
        "created_at": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # `git status --porcelain` empty => clean tree.
        "git_dirty": bool(_git("status", "--porcelain")),
        "packages": _package_versions(),
        # Fields the caller should fill per plan §12.6; default None so a missing
        # one is visible in the manifest rather than silently absent.
        "base_model": None,
        "tokenizer_hash": tokenizer_hash(tokenizer),
        "layer_triplet": None,
        "corpus": None,
        "corpus_slice": None,
        "doc_split_seed": None,
        "position_seed": None,
        "prompt_template": None,   # Stage 2+ (AV) — None at extraction/probe
        "marker_ids": None,        # Stage 2+ (AV) — None at extraction/probe
        "lora_config": None,       # Stage 2+ — None at extraction/probe
    }
    if extra:
        manifest.update(extra)
    return manifest
