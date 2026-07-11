"""Slot-spec grammar for layers × token-positions AV-input conditions.

A condition is an ordered list of SLOTS, each a (layer, position-offset) pair:

    L24@0      layer 24 at the labeled position p          (the classic input)
    L24@-2     layer 24 at position p-2                    (needs window_L24, W >= 3)
    L23@0      layer 23 at p                               (multi-layer slot)

Condition strings (CLI):  name=SLOT,SLOT,...[|flag...]  joined by ';'

    single    = L24@0
    local     = L23@0,L24@0,L25@0          # 1 position x 3 layers  (§7 'local')
    dup3      = L24@0,L24@0,L24@0          # marker-count control   (§7 'duplicate')
    tok3      = L24@-2,L24@-1,L24@0        # 3 positions x 1 layer  (the multitoken arm)
    tokdup3   = L24@0,L24@0,L24@0          # == dup3: control for tok3 at matched k
    grid2x2   = L23@-1,L23@0,L25@-1,L25@0  # positions x layers grid (k=4)
    mean3     = L23@0,L24@0,L25@0|pool     # element-wise mean -> ONE k=1 slot
    tok3_shufctx = L24@-2,L24@-1,L24@0|shufctx   # eval-only control: all but the
                                                 # LAST slot come from another doc

Flags:
    pool     mean all slots into a single av_in_0 (k=1); RAW mean, norm-matched
             at inject time like any single slot.
    shufctx  EVAL-ONLY control (build_conditions refuses it for AV training
             files): non-final slots are doc-deranged across rows, final slot
             kept true. Evaluated with the PARENT condition's AV checkpoint —
             if window ≈ shufctx, the extra slots aren't used as this-doc
             context (the multitoken pre-registration's load-bearing control).

Slot order in the spec == av_in_* column order == prompt marker order ==
injection scan order. The AR reconstruction target is NEVER part of a
condition — it stays the fixed target triplet no matter what the AV sees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SLOT_RE = re.compile(r"^L(\d+)@(0|-\d+)$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
KNOWN_FLAGS = ("pool", "shufctx")


@dataclass(frozen=True)
class Condition:
    name: str
    slots: tuple  # ((layer, offset), ...) — offset <= 0, 0 == labeled position p
    pool: bool = False
    shufctx: bool = False

    @property
    def k(self) -> int:
        """Number of av_in_* slots this condition materializes (1 when pooled)."""
        return 1 if self.pool else len(self.slots)

    @property
    def layers(self) -> tuple:
        return tuple(l for l, _ in self.slots)

    @property
    def offsets(self) -> tuple:
        return tuple(o for _, o in self.slots)

    def describe(self) -> str:
        s = ",".join(f"L{l}@{o}" for l, o in self.slots)
        flags = "".join(f"|{f}" for f in ("pool", "shufctx") if getattr(self, f))
        return f"{self.name}={s}{flags}"


def parse_slot(tok: str):
    m = _SLOT_RE.match(tok.strip())
    if not m:
        raise ValueError(
            f"bad slot {tok!r} — expected L<layer>@<offset<=0>, e.g. L24@0 or L23@-2"
        )
    layer, off = int(m.group(1)), int(m.group(2))
    assert off <= 0, f"slot {tok!r}: offset must be <= 0 (0 = the labeled position p)"
    return layer, off


def parse_condition(entry: str) -> Condition:
    """'name=L23@0,L24@0|pool' -> Condition. Raises on any malformed part."""
    if "=" not in entry:
        raise ValueError(f"condition {entry!r} lacks 'name=' prefix")
    name, rhs = entry.split("=", 1)
    name = name.strip()
    assert _NAME_RE.match(name), f"bad condition name {name!r} (use [A-Za-z0-9_.-])"
    parts = [p.strip() for p in rhs.split("|")]
    slot_str, flags = parts[0], parts[1:]
    for f in flags:
        assert f in KNOWN_FLAGS, f"condition {name!r}: unknown flag {f!r} (known: {KNOWN_FLAGS})"
    slots = tuple(parse_slot(t) for t in slot_str.split(",") if t.strip())
    assert slots, f"condition {name!r} has no slots"
    cond = Condition(name=name, slots=slots,
                     pool="pool" in flags, shufctx="shufctx" in flags)
    assert not (cond.pool and cond.shufctx), f"{name!r}: pool+shufctx makes no sense together"
    if cond.shufctx:
        assert len(slots) >= 2, f"{name!r}: shufctx needs >= 2 slots (context + the true final)"
    return cond


def parse_conditions(spec: str) -> dict:
    """';'-joined condition entries -> {name: Condition} (order-preserving)."""
    out: dict = {}
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        c = parse_condition(entry)
        assert c.name not in out, f"duplicate condition name {c.name!r}"
        out[c.name] = c
    assert out, f"no conditions parsed from {spec!r}"
    return out


def validate_against_bank(cond: Condition, *, activation_layers: set,
                          window_layers: set, window_size: int) -> None:
    """A condition is buildable iff every slot's source exists in the bank:
    offset 0 needs activation_L{layer}; offset -j needs window_L{layer} with
    W > j (window slot W-1-j exists). Raises with the fix spelled out."""
    for layer, off in cond.slots:
        if off == 0:
            if layer not in activation_layers:
                raise SystemExit(
                    f"condition {cond.name!r}: slot L{layer}@0 needs activation_L{layer} — "
                    f"bank has {sorted(activation_layers)}. Re-run regenerate_bank with "
                    f"--save-layers covering {layer}."
                )
        else:
            j = -off
            if layer not in window_layers:
                raise SystemExit(
                    f"condition {cond.name!r}: slot L{layer}@{off} needs window_L{layer} — "
                    f"bank windows cover {sorted(window_layers)}. Re-run regenerate_bank "
                    f"with --window-layers covering {layer}."
                )
            if j >= window_size:
                raise SystemExit(
                    f"condition {cond.name!r}: slot L{layer}@{off} needs window size > {j}, "
                    f"bank has W={window_size}. Re-run regenerate_bank with --window > {j}."
                )


# The §7 sweep conditions expressed in the new grammar (all offsets 0) — kept as
# a reference vocabulary and for the legacy-reproduction path.
LEGACY_SWEEP = {
    "local": "local=L23@0,L24@0,L25@0",
    "duplicate": "duplicate=L24@0,L24@0,L24@0",
    "wide": "wide=L20@0,L24@0,L28@0",
    "single": "single=L24@0",
    "s2_19_21_23": "s2_19_21_23=L19@0,L21@0,L23@0",
    "s2_20_22_24": "s2_20_22_24=L20@0,L22@0,L24@0",
}

# The pre-registered position-vs-layer permutation grid. Structure:
#   * matched-k comparisons only (dup{k} is each k's marker-count control);
#   * lay3 vs tok3 — layer diversity vs position diversity at k=3;
#   * single -> tok3 -> tok5 (+ their dup controls) — dose-response over
#     window width;
#   * tok5 (contiguous, reach p-4) vs tok5w (log-spaced, reach p-7 == the full
#     W=8 bank window) — adjacency vs span at k=5. Position-space analog of
#     §7's stride-2 result (spread layers beat adjacent ones at matched k);
#     adjacent positions are maximally collinear, so log-spacing (dense near p,
#     sparse far back — matching how context is progressively summarized)
#     should buy more non-redundant information per slot;
#   * shufctx arms are EVAL-ONLY controls for the position conditions.
PERMUTATION_GRID = (
    "single=L24@0; "
    "dup3=L24@0,L24@0,L24@0; "
    "lay3=L23@0,L24@0,L25@0; "
    "tok3=L24@-2,L24@-1,L24@0; "
    "mix4=L23@-1,L23@0,L25@-1,L25@0; "
    "dup4=L24@0,L24@0,L24@0,L24@0; "
    "tok5=L24@-4,L24@-3,L24@-2,L24@-1,L24@0; "
    "tok5w=L24@-7,L24@-4,L24@-2,L24@-1,L24@0; "
    "dup5=L24@0,L24@0,L24@0,L24@0,L24@0; "
    "tok3_shufctx=L24@-2,L24@-1,L24@0|shufctx; "
    "tok5_shufctx=L24@-4,L24@-3,L24@-2,L24@-1,L24@0|shufctx; "
    "tok5w_shufctx=L24@-7,L24@-4,L24@-2,L24@-1,L24@0|shufctx"
)

# LAYER grid — exploits the full stored L19-29 band, ALL slots at position p
# (offset 0), so it needs only activation_L{k} (no windows). Every condition
# reconstructs the SAME fixed target (default [L23,L24,L25]@p); the input layers
# vary. Matched-k with dup controls; the head-to-heads mirror §7/§8:
#   * adjacency vs span at k=3: lay3 (adjacent 23-25) vs wide (20/24/28) vs
#     s2lo/s2hi (stride-2). §7 found span/stride beats adjacency at fixed k.
#   * saturation: single -> lay3 -> band5 (contiguous 5) each vs its dup control
#     — does more layers keep paying, or plateau (§8 said ~k=2-with-L24)?
#   * adjacency vs span at k=5: band5 (contiguous 21-25) vs spread5 (19..29
#     across the whole band).
#   * far2 (the extremes L19,L29) — do two maximally-separated layers beat the
#     adjacent pair lay2? decorrelation in the extreme.
# NB reconstructing a DIFFERENT target center is NOT here — that needs a
# retrained AR (AR_LAYER_TO_TARGET_COL is fixed to 23/24/25). This sweeps the
# INPUT only, which is free from the bank.
LAYER_GRID = (
    "single=L24@0; "
    "lay2=L23@0,L25@0; "
    "far2=L19@0,L29@0; "
    "dup2=L24@0,L24@0; "
    "dup3=L24@0,L24@0,L24@0; "
    "lay3=L23@0,L24@0,L25@0; "
    "wide=L20@0,L24@0,L28@0; "
    "s2lo=L19@0,L21@0,L23@0; "
    "s2hi=L20@0,L22@0,L24@0; "
    "dup5=L24@0,L24@0,L24@0,L24@0,L24@0; "
    "band5=L21@0,L22@0,L23@0,L24@0,L25@0; "
    "spread5=L19@0,L22@0,L24@0,L26@0,L29@0"
)
