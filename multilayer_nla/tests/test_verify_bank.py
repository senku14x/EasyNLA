"""Offline tests for verify_bank (fabricated parquets; no GPU/model).

Validates the correctness gate itself: a good unified bank passes (window
internal parity + stored-vector parity + external cross-parity), and an
off-position reference, an internal last-slot mismatch, non-finite values, and
a wrong stored vector each FAIL. Guards against the dangerous case — a verifier
that false-passes a real gather bug.

Run: python -m pytest multilayer_nla/tests/test_verify_bank.py -q
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from multilayer_nla.verify_bank import verify_bank

D, W, N = 8, 4, 12
SAVE = [23, 24, 25]
WINL = [24, 25]


def _fsl(mat, inner, dtype):
    flat = np.ascontiguousarray(mat).reshape(-1).astype(dtype)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat), inner)


def _rng():
    return np.random.default_rng(0)


def _write_bank(path, finals, wins, *, act_override=None, finite=True,
                stored=None, stored_layer=24, window_size=W):
    """finals: {k: [N, D] fp32}; wins: {k: [N, W, D] fp16}."""
    cols = {
        "doc_id": pa.array([f"d{i}" for i in range(N)]),
        "n_raw_tokens": pa.array([100 + i for i in range(N)], pa.int64()),
    }
    if stored is not None:
        cols["activation_vector"] = _fsl(stored, D, np.float32)
        cols["activation_layer"] = pa.array([stored_layer] * N, pa.int64())
    for k in SAVE:
        f = finals[k] if act_override is None or k not in act_override else act_override[k]
        cols[f"activation_L{k}"] = _fsl(f, D, np.float32)
    for k in WINL:
        w = wins[k].copy()
        if not finite:
            w[0, 0, 0] = np.inf
        cols[f"window_L{k}"] = _fsl(w.reshape(N, W * D), W * D, np.float16)
    cols["window_size"] = pa.array([window_size] * N, pa.int32())
    pq.write_table(pa.table(cols), path)
    Path(path + ".mlnla_meta.yaml").write_text(yaml.safe_dump({
        "kind": "mlnla_bank", "schema_version": 2, "save_layers": SAVE,
        "window_layers": WINL, "window_size": W, "activation_dtype": "float32",
        "window_dtype": "float16", "norm": "none", "d_model": D, "rows_out": N,
    }))


def _mk():
    rng = _rng()
    wins = {k: (rng.standard_normal((N, W, D)) * 5).astype(np.float16) for k in WINL}
    finals = {}
    for k in SAVE:
        if k in wins:
            finals[k] = wins[k][:, -1, :].astype(np.float32)  # last slot == p
        else:
            finals[k] = (rng.standard_normal((N, D)) * 5).astype(np.float32)
    return finals, wins


def _write_ref(path, finals):
    cols = {
        "doc_id": pa.array([f"d{i}" for i in range(N)]),
        "n_raw_tokens": pa.array([100 + i for i in range(N)], pa.int64()),
    }
    for k in SAVE:
        cols[f"activation_L{k}"] = _fsl(finals[k], D, np.float32)
    pq.write_table(pa.table(cols), path)


def test_good_bank_passes():
    with tempfile.TemporaryDirectory() as td:
        b = f"{td}/bank.parquet"
        finals, wins = _mk()
        _write_bank(b, finals, wins)
        assert verify_bank(b, log=lambda *a: None) == 0


def test_good_bank_with_stored_vector_passes():
    with tempfile.TemporaryDirectory() as td:
        b = f"{td}/bank.parquet"
        finals, wins = _mk()
        _write_bank(b, finals, wins, stored=finals[24], stored_layer=24)
        assert verify_bank(b, log=lambda *a: None) == 0


def test_wrong_stored_vector_fails():
    with tempfile.TemporaryDirectory() as td:
        b = f"{td}/bank.parquet"
        finals, wins = _mk()
        wrong = _rng().standard_normal((N, D)).astype(np.float32) + 7.0
        _write_bank(b, finals, wins, stored=wrong, stored_layer=24)
        assert verify_bank(b, log=lambda *a: None) > 0


def test_cross_parity_passes_when_ref_matches():
    with tempfile.TemporaryDirectory() as td:
        b, r = f"{td}/bank.parquet", f"{td}/ref.parquet"
        finals, wins = _mk()
        _write_bank(b, finals, wins)
        _write_ref(r, finals)
        assert verify_bank(b, r, strict=True, log=lambda *a: None) == 0
        assert verify_bank(b, r, strict=False, log=lambda *a: None) == 0


def test_off_position_reference_fails():
    with tempfile.TemporaryDirectory() as td:
        b, r = f"{td}/bank.parquet", f"{td}/ref.parquet"
        finals, wins = _mk()
        _write_bank(b, finals, wins)
        # reference vectors = the OLDEST window slot (off by W-1 positions)
        off = {k: (wins[k][:, 0, :].astype(np.float32) if k in wins else finals[k])
               for k in SAVE}
        _write_ref(r, off)
        assert verify_bank(b, r, strict=False, log=lambda *a: None) > 0


def test_internal_last_slot_mismatch_fails():
    with tempfile.TemporaryDirectory() as td:
        b = f"{td}/bank.parquet"
        finals, wins = _mk()
        bad = {24: (finals[24] + 3.0).astype(np.float32)}
        _write_bank(b, finals, wins, act_override=bad)
        assert verify_bank(b, log=lambda *a: None) > 0


def test_non_finite_fails():
    with tempfile.TemporaryDirectory() as td:
        b = f"{td}/bank.parquet"
        finals, wins = _mk()
        _write_bank(b, finals, wins, finite=False)
        assert verify_bank(b, log=lambda *a: None) > 0


def test_window_size_mismatch_fails():
    with tempfile.TemporaryDirectory() as td:
        b = f"{td}/bank.parquet"
        finals, wins = _mk()
        _write_bank(b, finals, wins, window_size=W + 1)  # column != sidecar
        assert verify_bank(b, log=lambda *a: None) > 0


def test_missing_sidecar_fails():
    with tempfile.TemporaryDirectory() as td:
        b = f"{td}/bank.parquet"
        finals, wins = _mk()
        _write_bank(b, finals, wins)
        Path(b + ".mlnla_meta.yaml").unlink()
        assert verify_bank(b, log=lambda *a: None) > 0


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS {name}")
    print("\nALL PASSED")


if __name__ == "__main__":
    _run_all()
