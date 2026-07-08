"""End-to-end CPU smoke of the unified extraction path on a TINY in-memory
Llama (random weights, byte-level tokenizer) — no GPU, no network, no real
checkpoint. This is the pre-GPU gate: it exercises the REAL forward path —
multi-hook registration, the _CaptureComplete short-circuit at max(layer), the
on-GPU-style final-token + window gathers, right-padding assumptions, the
round-trip guard, stored-vector cross-parity, and the bank writer — so a broken
setup fails HERE, not 2 hours into an H200 run.

Checks (the decisive ones):
  1. extract_multi center tap == single-layer extract() BITWISE (parity).
  2. extract_multi(final_token_only=True) == full capture's last row BITWISE.
  3. extract_bank final == extract_multi final BITWISE; window last slot == the
     final-token vector (fp16-cast); window slots equal the full capture's last
     W rows; short rows flagged invalid.
  4. layers above max(requested) never run (the short-circuit actually fires).
  5. regenerate-bank main() over a synthetic published parquet: output columns,
     window_size, sidecar, stored-parity PASS, and verify_bank PASS on the
     result; a corrupted stored vector makes the parity guard raise.

Run: python -m pytest multilayer_nla/tests/test_extractor_e2e_cpu.py -q
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

transformers = pytest.importorskip("transformers")

D_MODEL = 32
N_LAYERS = 8
VOCAB = 300


def _tiny_model_dir(tmp: Path) -> str:
    """Write a tiny random-init Llama + byte-level tokenizer to disk so the
    extractor's from_pretrained/load_tokenizer path is the REAL one."""
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders

    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=D_MODEL, intermediate_size=64,
        num_hidden_layers=N_LAYERS, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    d = tmp / "tiny_llama"
    model.save_pretrained(str(d))

    tok = Tokenizer(models.BPE(vocab={chr(32 + i): i for i in range(VOCAB - 2)},
                               merges=[], unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token=chr(32 + VOCAB - 3), pad_token=chr(32 + VOCAB - 4))
    fast.save_pretrained(str(d))
    return str(d)


@pytest.fixture(scope="module")
def extractor():
    with tempfile.TemporaryDirectory() as td:
        model_dir = _tiny_model_dir(Path(td))
        from multilayer_nla.extract_multilayer import MultiLayerHFExtractor
        yield MultiLayerHFExtractor(
            model_name=model_dir, device_map=None, torch_dtype=torch.float32,
            max_length=64, batch_size=3, attn_implementation="eager",
        )


TEXTS = ["hello world this is a doc", "b", "another longer document goes here ok",
         "mid size text", "x y z"]


def test_center_parity_multi_vs_single(extractor):
    center = 4
    legacy = extractor.extract(TEXTS, center)
    multi = extractor.extract_multi(TEXTS, [center - 1, center, center + 1])
    assert len(legacy) == len(multi)
    for lr, mr in zip(legacy, multi):
        assert lr.token_ids == mr["token_ids"]
        assert torch.equal(mr["hidden"][center], lr.hidden_states)  # bitwise


def test_final_token_only_matches_full_capture(extractor):
    layers = [2, 5]
    full = extractor.extract_multi(TEXTS, layers)
    fin = extractor.extract_multi(TEXTS, layers, final_token_only=True)
    for f, g in zip(full, fin):
        for li in layers:
            assert torch.equal(g["hidden"][li], f["hidden"][li][-1])


def test_extract_bank_finals_windows_and_validity(extractor):
    save_layers, window_layers, W = [2, 4, 6], [4, 6], 4
    full = extractor.extract_multi(TEXTS, save_layers)
    bank = extractor.extract_bank(TEXTS, save_layers, window_layers, W)
    for f, b in zip(full, bank):
        n_tok = len(f["token_ids"])
        assert b["token_ids"] == f["token_ids"]
        for li in save_layers:
            assert torch.equal(b["final"][li], f["hidden"][li][-1])  # bitwise
        if n_tok >= W:
            assert b["valid"]
            for li in window_layers:
                want = f["hidden"][li][n_tok - W: n_tok].to(torch.float16)
                assert torch.equal(b["window"][li], want)
                assert torch.equal(b["window"][li][-1],
                                   b["final"][li].to(torch.float16))
        else:
            assert not b["valid"]
    # at least one of TEXTS is shorter than W tokens -> exercises the invalid path
    assert any(not b["valid"] for b in bank)
    assert any(b["valid"] for b in bank)


def test_short_circuit_skips_upper_layers(extractor):
    from nla.utils.arch_adapters import resolve_decoder_layers
    fired = []
    top = resolve_decoder_layers(extractor.model)[N_LAYERS - 1]
    h = top.register_forward_hook(lambda *a: fired.append(1))
    try:
        extractor.extract_multi(TEXTS[:2], [1, 3])
        assert not fired, "layer above max(requested) ran — short-circuit broken"
        extractor.extract_multi(TEXTS[:2], [N_LAYERS - 1])
        assert fired, "requesting the top layer must still reach it"
    finally:
        h.remove()


def _published_parquet(path: str, extractor, texts, stored_layer=4, corrupt=False):
    """Synthesize a 'published warmstart' parquet: prefix text + n_raw_tokens +
    the stored single-layer activation_vector (computed by the extractor itself,
    so parity must hold bitwise unless we corrupt it)."""
    res = extractor.extract_multi(texts, [stored_layer], final_token_only=True)
    n_raw = [len(r["token_ids"]) for r in res]
    stored = np.stack([r["hidden"][stored_layer].numpy() for r in res])
    if corrupt:
        stored = stored + 1.0
    flat = np.ascontiguousarray(stored).reshape(-1).astype(np.float32)
    pq.write_table(pa.table({
        "detokenized_text_truncated": pa.array(texts),
        "n_raw_tokens": pa.array(n_raw, pa.int64()),
        "activation_vector": pa.FixedSizeListArray.from_arrays(pa.array(flat), D_MODEL),
        "activation_layer": pa.array([stored_layer] * len(texts), pa.int64()),
        "doc_id": pa.array([f"doc:{i}" for i in range(len(texts))], pa.string()),
        "response": pa.array([f"<explanation>\nf{i}\n</explanation>" for i in range(len(texts))]),
    }), path)


def _run_regen_main(argv):
    from multilayer_nla import regenerate_bank
    old = sys.argv
    sys.argv = ["regenerate_bank"] + argv
    try:
        regenerate_bank.main()
    finally:
        sys.argv = old


def test_regenerate_bank_end_to_end(extractor, monkeypatch):
    # monkeypatch the extractor loader so main() reuses the tiny fixture model
    import multilayer_nla.regenerate_bank as rb
    monkeypatch.setattr(rb, "load_class", lambda _: (lambda **kw: extractor))
    texts = [t for t in TEXTS if len(t) > 8]  # keep rows that fill a W=4 window
    with tempfile.TemporaryDirectory() as td:
        inp, out = f"{td}/av_sft.parquet", f"{td}/bank.parquet"
        _published_parquet(inp, extractor, texts)
        _run_regen_main([
            "--in", inp, "--out", out, "--base-model", "tiny", "--center-layer", "4",
            "--save-layers", "3-5", "--window", "4", "--window-layers", "4,5",
            "--max-length", "64", "--batch-size", "3", "--chunk-size", "2",
        ])
        t = pq.read_table(out)
        for c in ("activation_L3", "activation_L4", "activation_L5",
                  "window_L4", "window_L5", "window_size", "center_layer",
                  "response", "doc_id", "activation_vector"):
            assert c in t.schema.names, f"missing {c}"
        assert set(t.column("window_size").to_pylist()) == {4}
        # stored-vector parity held (main would have raised otherwise); the
        # regenerated L4 must equal the stored vector BITWISE here (same
        # extractor, same batching within each chunk? batch differs — so allow
        # cosine ~1 rather than bitwise).
        a = np.asarray(t.column("activation_L4").to_pylist(), dtype=np.float32)
        s = np.asarray(t.column("activation_vector").to_pylist(), dtype=np.float32)
        cos = (a * s).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(s, axis=1) + 1e-8)
        assert float(np.median(cos)) > 0.999
        meta = Path(out + ".mlnla_meta.yaml")
        assert meta.exists()
        # the produced bank passes the offline verifier (incl. stored parity)
        from multilayer_nla.verify_bank import verify_bank
        assert verify_bank(out, log=lambda *a: None) == 0

        # corrupted stored vectors -> the parity guard must hard-fail
        inp2, out2 = f"{td}/bad.parquet", f"{td}/bad_bank.parquet"
        _published_parquet(inp2, extractor, texts, corrupt=True)
        with pytest.raises(AssertionError, match="cross-parity"):
            _run_regen_main([
                "--in", inp2, "--out", out2, "--base-model", "tiny", "--center-layer", "4",
                "--save-layers", "3-5", "--window", "0",
                "--max-length", "64", "--batch-size", "3", "--chunk-size", "8",
            ])


def test_regenerate_bank_dry_run_no_model(monkeypatch):
    # --dry-run must not construct any extractor at all
    import multilayer_nla.regenerate_bank as rb

    def boom(_):
        raise AssertionError("model load attempted under --dry-run")

    monkeypatch.setattr(rb, "load_class", boom)
    with tempfile.TemporaryDirectory() as td:
        inp = f"{td}/av.parquet"
        flat = np.zeros(2 * D_MODEL, dtype=np.float32)
        pq.write_table(pa.table({
            "detokenized_text_truncated": pa.array(["a", "b"]),
            "n_raw_tokens": pa.array([1, 1], pa.int64()),
            "activation_vector": pa.FixedSizeListArray.from_arrays(pa.array(flat), D_MODEL),
            "activation_layer": pa.array([4, 4], pa.int64()),
        }), inp)
        _run_regen_main(["--in", inp, "--out", f"{td}/o.parquet", "--base-model", "x",
                         "--center-layer", "4", "--dry-run"])
