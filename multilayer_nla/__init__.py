"""Multi-layer / multi-position NLA — experiment package on top of the nla/ core.

Ported from the nanoNLA multilayer_nla + multitoken_nla packages (the §7/§8
sweep lineage) and adapted to EasyNLA's core (nla.utils.arch_adapters, the
short-circuit HFExtractor, arch-aware LoRA targets). Never mutates nla/.

Entry points:
    regenerate_bank        published labels -> multi-layer × multi-position bank (GPU)
    verify_bank            post-build integrity gate (CPU)
    verify_center_parity   multi-hook center tap == legacy single-layer (GPU, bitwise)
    verify_regen_parity    final-token gather == legacy final token (GPU, bitwise)
    build_from_published   bank -> av/ar/rl training parquets for a chosen center
    splits                 doc-level train/dev/test manifests
    train_av_multi         k-slot verbalizer SFT warm-start
    train_ar_multi         multi-tap reconstructor SFT warm-start
    train_rl_multi         single-GPU GRPO (legacy fixed 3-slot scheme)
    evaluate_e2e           held-out end-to-end FVE (+ bootstrap CIs, shuffled control)
    eval_ar_gold           AR-only gold ceiling (bottleneck localization)
"""
