"""Checkpoint selection (dev only) + the final result table for the §7 sweep.

Two modes, run either side of the one-shot test eval:

  --mode select   read the dev grid (evaluate_e2e summaries on rl_dev for every
                  AR ckpt x AV ckpt x condition) and choose, USING DEV ONLY:
                    * one shared AR ckpt = argmax over AR of the mean (over the four
                      conditions) of each condition's best-AV dev FVE;
                    * then, at that AR, the better AV ckpt per condition.
                  Writes selection.json + selection.env (sourceable by the sweep).

  --mode report   read the one-shot rl_test summaries for the SELECTED ckpts (+ the
                  AR-only gold dev/test FVE) and emit the compact result table.

Selection metric defaults to penalized end-to-end FVE (pen_fve_overall) — the
failure-honest headline. Nothing here ever reads a test summary during selection.

Dev summary filenames:  dev_<cond>_ar<arstep>_av<avstep>.json
Test summary filenames: test_<cond>.json
"""

import argparse
import glob
import json
import re
from pathlib import Path

CONDS = ("local", "duplicate", "wide", "single", "s2_19_21_23", "s2_20_22_24")
AV_INPUT_LAYERS = {"local": "23,24,25", "duplicate": "24,24,24", "wide": "20,24,28", "single": "24",
                   "s2_19_21_23": "19,21,23", "s2_20_22_24": "20,22,24"}
DEV_RE = re.compile(r"dev_(?P<cond>\w+)_ar(?P<ar>\d+)_av(?P<av>\d+)\.json$")


def _load(path):
    return json.loads(Path(path).read_text())


def load_dev_grid(dev_dir, metric):
    """grid[cond][ar_step][av_step] = metric value, from dev_*.json summaries."""
    grid = {}
    for p in sorted(glob.glob(str(Path(dev_dir) / "dev_*.json"))):
        m = DEV_RE.search(Path(p).name)
        if not m:
            continue
        s = _load(p)
        grid.setdefault(m["cond"], {}).setdefault(int(m["ar"]), {})[int(m["av"])] = s.get(metric)
    return grid


def select(grid, conds):
    """Return (chosen_ar_step, {cond: chosen_av_step}, diagnostics). Dev only."""
    ar_steps = sorted({ar for c in grid for ar in grid[c]})
    assert ar_steps, "no dev summaries found"
    for c in conds:
        assert grid.get(c), f"dev grid missing condition {c!r} — incomplete eval grid"
    # mean over conditions of each condition's best-AV dev metric, per AR ckpt
    mean_by_ar = {}
    for ar in ar_steps:
        best_per_cond = []
        for c in conds:
            avs = grid.get(c, {}).get(ar, {})
            vals = [v for v in avs.values() if v is not None]
            if vals:
                best_per_cond.append(max(vals))
        mean_by_ar[ar] = sum(best_per_cond) / len(best_per_cond) if best_per_cond else float("-inf")
    chosen_ar = max(ar_steps, key=lambda a: mean_by_ar[a])
    chosen_av = {}
    for c in conds:
        avs = grid.get(c, {}).get(chosen_ar, {})
        avs = {k: v for k, v in avs.items() if v is not None}
        chosen_av[c] = max(avs, key=lambda k: avs[k]) if avs else None
    missing = [c for c in conds if chosen_av.get(c) is None]
    assert not missing, (f"no valid dev metric for {missing} at AR step {chosen_ar} — the dev "
                         f"grid is incomplete (need AR{{500,1000}} x AV{{500,1000}} per condition)")
    return chosen_ar, chosen_av, {"mean_dev_metric_by_ar": mean_by_ar}


def _fmt(x, pct=True):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x * 100:.1f}" if pct else f"{x:.0f}"


def make_table(test_dir, selection, ar_gold_dev=None, ar_gold_test=None):
    lines = []
    lines.append("| condition | target layers | AV input layers | test FVE prev | test FVE centre "
                 "| test FVE next | test FVE overall | penalized FVE | extraction rate | mean tokens |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c in CONDS:
        p = Path(test_dir) / f"test_{c}.json"
        if not p.exists():
            lines.append(f"| {c} | 23,24,25 | {AV_INPUT_LAYERS[c]} | — | — | — | — | — | — | — |")
            continue
        s = _load(p)
        lines.append(
            f"| {c} | 23,24,25 | {AV_INPUT_LAYERS[c]} "
            f"| {_fmt(s.get('fve_prev'))} | {_fmt(s.get('fve_centre'))} | {_fmt(s.get('fve_next'))} "
            f"| {_fmt(s.get('fve_overall'))} | {_fmt(s.get('pen_fve_overall'))} "
            f"| {_fmt(s.get('successful_extraction_rate'))} | {_fmt(s.get('mean_generated_tokens'), pct=False)} |")
    out = ["## §7 SFT control sweep — held-out TEST results", "",
           f"Selected shared AR ckpt: step {selection.get('chosen_ar_step')}; "
           f"per-condition AV ckpt: {selection.get('chosen_av_step')}", "",
           "All conditions reconstruct the SAME fixed target [L23,L24,L25]; only the AV input varies.",
           "", *lines, ""]
    # AR-only gold held-out FVE (reconstructor ceiling, independent of AV / extraction)
    for label, path in (("ar_dev", ar_gold_dev), ("ar_test", ar_gold_test)):
        if path and Path(path).exists():
            g = _load(path)
            out += [f"### AR-only gold {label} FVE (gold explanation → shared AR → [L23,L24,L25])",
                    f"- prev/centre/next/overall = {_fmt(g.get('fve_prev'))} / {_fmt(g.get('fve_centre'))}"
                    f" / {_fmt(g.get('fve_next'))} / {_fmt(g.get('fve_overall'))}", ""]
    # shuffled control sanity (should collapse toward 0)
    shuf = [f"{c}={_fmt(_load(Path(test_dir) / f'test_{c}.json').get('shuffled_pen_fve_overall'))}"
            for c in CONDS if (Path(test_dir) / f"test_{c}.json").exists()]
    if shuf:
        out += ["### Shuffled-generation control (penalized FVE; must collapse toward 0)",
                "- " + ", ".join(shuf)]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["select", "report"], required=True)
    p.add_argument("--dev-dir", help="dir with dev_<cond>_ar<>_av<>.json (select)")
    p.add_argument("--test-dir", help="dir with test_<cond>.json (report)")
    p.add_argument("--metric", default="pen_fve_overall",
                   help="dev selection metric (default penalized end-to-end FVE)")
    p.add_argument("--conds", default=",".join(CONDS))
    p.add_argument("--selection", help="selection.json (report reads it; select writes it)")
    p.add_argument("--env-out", help="also write a sourceable selection.env (select)")
    p.add_argument("--ar-gold-dev", help="AR-only gold dev summary json (report, optional)")
    p.add_argument("--ar-gold-test", help="AR-only gold test summary json (report, optional)")
    p.add_argument("--out", help="output path (selection.json for select / table.md for report)")
    args = p.parse_args()
    conds = tuple(c.strip() for c in args.conds.split(","))

    if args.mode == "select":
        grid = load_dev_grid(args.dev_dir, args.metric)
        chosen_ar, chosen_av, diag = select(grid, conds)
        sel = {"metric": args.metric, "chosen_ar_step": chosen_ar,
               "chosen_av_step": chosen_av, "diagnostics": diag}
        out = args.out or args.selection or "selection.json"
        Path(out).write_text(json.dumps(sel, indent=2))
        print(f"[select] metric={args.metric}  chosen AR step={chosen_ar}  AV steps={chosen_av}")
        print(f"[select] {diag}  -> {out}")
        if args.env_out:
            lines = [f"CHOSEN_AR_STEP={chosen_ar}"]
            for c in conds:
                lines.append(f"CHOSEN_AV_{c.upper()}_STEP={chosen_av.get(c)}")
            Path(args.env_out).write_text("\n".join(lines) + "\n")
            print(f"[select] env -> {args.env_out}")
    else:
        sel = _load(args.selection) if args.selection else {}
        table = make_table(args.test_dir, sel, args.ar_gold_dev, args.ar_gold_test)
        out = args.out or "result_table.md"
        Path(out).write_text(table + "\n")
        print(table)
        print(f"\n[report] -> {out}")


if __name__ == "__main__":
    main()
