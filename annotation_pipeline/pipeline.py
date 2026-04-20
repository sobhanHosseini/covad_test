"""
pipeline.py — Main orchestrator for the annotation pipeline.

Responsibilities:
  - Parse CLI arguments
  - Set up clients (Ollama, CLIP)
  - Coordinate all stages with checkpoint load/save
  - Implement --load_from_stage for fast holdout and CL scenarios
  - Print a clean summary at the end

Run as:
    python -m annotation_pipeline [args]
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from annotation_pipeline import checkpoints
from annotation_pipeline.clip_utils import load_clip, load_or_compute_clip_embeddings
from annotation_pipeline.dataset import discover_images, get_defect_types
from annotation_pipeline.stages import stage1, stage2, stage3, stage4
from annotation_pipeline import vocabulary, csv_builder

log = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VLM concept annotation pipeline for CONVAD (modular v6)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument("--dataset_path", required=True,
                   help="Root of the MVTec dataset (parent of category folders)")
    p.add_argument("--category",     required=True,
                   help="Category name, e.g. hazelnut, capsule, screw")
    p.add_argument("--save_path",    required=True,
                   help="Output CSV path, e.g. ./annotations/hazelnut_v6.csv")

    # ── VLM / Ollama ──────────────────────────────────────────────────────────
    p.add_argument("--model_name",   default="gemma4:e4b")
    p.add_argument("--ollama_host",  default="http://localhost:6000")

    # ── Checkpoint control ────────────────────────────────────────────────────
    p.add_argument(
        "--load_from_stage", type=int, default=0, choices=[0, 1, 2, 3],
        help=(
            "Load checkpoint from stage N and skip all earlier stages. "
            "0 = run everything from scratch (default). "
            "3 = load Stage 3 checkpoint and only rerun Stage 4 + CSV "
            "    (fastest for generating holdout variants). "
            "Checkpoints must already exist (run with 0 first)."
        ),
    )

    # ── Holdout ───────────────────────────────────────────────────────────────
    p.add_argument("--holdout_defect", default=None,
                   help="Defect type to hold out (Tier 3 concepts excluded from vocab).")

    # ── Sampling ──────────────────────────────────────────────────────────────
    p.add_argument("--n_normal_sample", type=int, default=None,
                   help="Subsample N normal images for Stage 1 (default: all).")
    p.add_argument("--n_annotate_sample", type=int, default=None,
                   help="Limit Stage 3 to N images per group (default: all).")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    p.add_argument("--min_concept_freq",  type=float, default=0.20,
                   help="Min fraction of normal images a concept must appear in.")
    p.add_argument("--max_concept_freq",  type=float, default=0.95)
    p.add_argument("--cluster_threshold", type=float, default=0.65,
                   help="Stage 2 dimension-aware clustering threshold.")
    p.add_argument("--skip_vlm_audit",   action="store_true",
                   help="Skip Stage 2d VLM self-audit.")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    p.add_argument("--n_normal_refs",           type=int, default=3,
                   help="CLIP-selected normal references per defect image (P1-A).")
    p.add_argument("--max_new_defect_concepts", type=int, default=6,
                   help="Max new defect concepts VLM extracts per defect image.")
    p.add_argument("--n_workers",               type=int, default=1,
                   help="Parallel workers for Stage 3 defect annotation (P2-B).")

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    p.add_argument("--skip_stage4", action="store_true",
                   help=(
                       "Skip Stage 4 (v2-mode): use raw defect concepts without clustering. "
                       "Produces more concepts, higher C-AUC but more redundancy."
                   ))
    p.add_argument("--stage4_cluster_threshold",     type=float, default=0.65)
    p.add_argument("--min_defect_freq",              type=float, default=0.20,
                   help="Cluster-union frequency threshold for keeping defect concepts.")
    p.add_argument("--min_defect_types_for_generic", type=int,   default=2,
                   help="Concepts spanning N+ defect types → promoted to Tier 2 generic.")
    p.add_argument("--use_visual_grounding",         action="store_true",
                   help="Enable CLIP visual grounding in Stage 4.")
    p.add_argument("--visual_weight",                type=float, default=0.60)

    # ── CL stub ───────────────────────────────────────────────────────────────
    p.add_argument("--append_defect", default=None,
                   help=(
                       "[CL mode] Name of a new defect type being added. "
                       "Loads Stage 3 checkpoint, appends new annotations, reruns Stage 4. "
                       "Requires --load_from_stage 3."
                   ))

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--debug",       action="store_true")

    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace):
    random.seed(args.random_seed)

    # ── Imports that require ollama (lazy — keeps module importable without it) ─
    from ollama import Client

    # ── Logging setup ─────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("═" * 60)
    log.info("CONVAD Annotation Pipeline  (modular v6)")
    log.info("═" * 60)
    log.info("Category     : %s", args.category)
    log.info("Dataset      : %s", args.dataset_path)
    log.info("Output CSV   : %s", args.save_path)
    log.info("Model        : %s @ %s", args.model_name, args.ollama_host)
    log.info("Load from    : Stage %d", args.load_from_stage)
    if args.holdout_defect:
        log.info("Holdout      : %s", args.holdout_defect)
    if args.append_defect:
        log.info("CL append    : %s", args.append_defect)

    checkpoints.log_checkpoint_status(args.save_path, args.category)

    # ── Clients ───────────────────────────────────────────────────────────────
    client = Client(host=args.ollama_host)
    clip_model, clip_preprocess, clip_device = load_clip()

    # ── Dataset discovery ─────────────────────────────────────────────────────
    image_groups = discover_images(args.dataset_path, args.category)
    defect_types = get_defect_types(image_groups)

    if args.holdout_defect and args.holdout_defect not in defect_types:
        raise ValueError(
            f"--holdout_defect '{args.holdout_defect}' not found. "
            f"Available: {defect_types}"
        )

    all_normal = image_groups["normal"]

    # ── CLIP embeddings (used by Stage 3 P1-A reference selection) ───────────
    # normal_embeddings: reference pool — which normal images look most like this defect?
    # defect_embeddings: query vectors — one per defect image
    normal_embeddings: dict = {}
    defect_embeddings: dict = {}

    if clip_model is not None and args.load_from_stage < 3:
        cache_path = str(checkpoints.clip_cache_path(args.save_path, args.category))
        normal_embeddings = load_or_compute_clip_embeddings(
            all_normal, clip_model, clip_preprocess, clip_device,
            cache_path=cache_path,
        )
        all_defect_paths = [
            p for k, v in image_groups.items()
            if k not in ("normal", "normal_test") for p in v
        ]
        if all_defect_paths:
            log.info("CLIP: computing embeddings for %d defect images...", len(all_defect_paths))
            from annotation_pipeline.clip_utils import compute_clip_embeddings
            defect_embeddings = compute_clip_embeddings(
                all_defect_paths, clip_model, clip_preprocess, clip_device
            )
    elif clip_model is not None:
        log.info("CLIP: skipping (Stage 3 loaded from checkpoint)")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1
    # ══════════════════════════════════════════════════════════════════════════
    if args.load_from_stage >= 1:
        raw_concepts = checkpoints.load_stage1(args.save_path, args.category)
        if raw_concepts is None:
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found for '{args.category}'. "
                f"Run with --load_from_stage 0 first."
            )
    else:
        log.info("\n" + "═" * 60)
        log.info("STAGE 1 — Per-image normal concept extraction")
        log.info("═" * 60)
        raw_concepts = stage1.run(
            client, args.model_name, all_normal, args.category,
            n_sample=args.n_normal_sample,
            random_seed=args.random_seed,
            debug=args.debug,
        )
        checkpoints.save_stage1(raw_concepts, args.save_path, args.category)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2
    # ══════════════════════════════════════════════════════════════════════════
    if args.load_from_stage >= 2:
        normal_concepts = checkpoints.load_stage2(args.save_path, args.category)
        if normal_concepts is None:
            raise FileNotFoundError(
                f"Stage 2 checkpoint not found for '{args.category}'. "
                f"Run with --load_from_stage 0 or 1 first."
            )
    else:
        log.info("\n" + "═" * 60)
        log.info("STAGE 2 — Normal concept post-processing")
        log.info("═" * 60)
        normal_concepts = stage2.run(
            client, args.model_name, raw_concepts, all_normal, args.category,
            min_freq=args.min_concept_freq,
            max_freq=args.max_concept_freq,
            cluster_threshold=args.cluster_threshold,
            run_vlm_audit=not args.skip_vlm_audit,
            debug=args.debug,
        )
        checkpoints.save_stage2(normal_concepts, args.save_path, args.category)

    # Stage 2b — always recomputed (it's just a list merge, instant)
    all_concepts = stage2.add_generic_concepts(normal_concepts)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3
    # ══════════════════════════════════════════════════════════════════════════
    if args.load_from_stage >= 3:
        all_annotations = checkpoints.load_stage3(args.save_path, args.category)
        if all_annotations is None:
            raise FileNotFoundError(
                f"Stage 3 checkpoint not found for '{args.category}'. "
                f"Run with --load_from_stage 0 first."
            )

        # ── CL mode: append new defect annotations ────────────────────────────
        if args.append_defect:
            if args.append_defect not in defect_types:
                raise ValueError(
                    f"--append_defect '{args.append_defect}' not found in {defect_types}"
                )
            log.info("\n[CL] Annotating new defect type: '%s'", args.append_defect)
            new_anns = stage3.run(
                client, args.model_name,
                {args.append_defect: image_groups[args.append_defect]},
                [args.append_defect],
                all_concepts, args.category,
                normal_embeddings={},
                defect_embeddings={},   # no CLIP in append mode (fast)
                n_normal_refs=1,
                max_new_defect_concepts=args.max_new_defect_concepts,
                n_workers=args.n_workers,
                n_annotate_sample=args.n_annotate_sample,
                debug=args.debug,
            )
            all_annotations = checkpoints.append_to_stage3(
                new_anns, args.save_path, args.category
            )
    else:
        log.info("\n" + "═" * 60)
        log.info("STAGE 3 — Per-image annotation")
        log.info("═" * 60)
        all_annotations = stage3.run(
            client, args.model_name,
            image_groups, defect_types, all_concepts, args.category,
            normal_embeddings=normal_embeddings,
            defect_embeddings=defect_embeddings,
            n_normal_refs=args.n_normal_refs,
            max_new_defect_concepts=args.max_new_defect_concepts,
            n_workers=args.n_workers,
            n_annotate_sample=args.n_annotate_sample,
            debug=args.debug,
        )
        checkpoints.save_stage3(all_annotations, args.save_path, args.category)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "═" * 60)
    log.info("STAGE 4 — Defect concept refinement")
    log.info("═" * 60)

    raw_defect_map = stage3.collect_defect_concepts(all_annotations)
    log.info("Raw defect concepts discovered:")
    for dt, concepts in raw_defect_map.items():
        log.info("  %-15s: %d concepts", dt, len(concepts))

    if args.skip_stage4:
        log.info("Stage 4 skipped (--skip_stage4 / v2-mode): using raw defect concepts")
        refined_defect_map = raw_defect_map
        discovered_generic: list[dict] = []
    else:
        refined_defect_map, discovered_generic = stage4.run(
            defect_concept_map=raw_defect_map,
            all_annotations=all_annotations,
            image_groups=image_groups,
            cluster_threshold=args.stage4_cluster_threshold,
            min_defect_freq=args.min_defect_freq,
            min_defect_types_for_generic=args.min_defect_types_for_generic,
            use_visual_grounding=args.use_visual_grounding,
            visual_weight=args.visual_weight,
            clip_model=clip_model if args.use_visual_grounding else None,
            clip_preprocess=clip_preprocess,
            clip_device=clip_device,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # VOCABULARY + CSV
    # ══════════════════════════════════════════════════════════════════════════
    final_vocab = vocabulary.build(
        normal_concepts=normal_concepts,
        generic_concepts=all_concepts[len(normal_concepts):],
        defect_concept_map=refined_defect_map,
        holdout_defect=args.holdout_defect,
        discovered_generic=discovered_generic,
    )

    df = csv_builder.build(
        dataset_path=args.dataset_path,
        category=args.category,
        image_groups=image_groups,
        all_annotations=all_annotations,
        final_vocab=final_vocab,
        holdout_defect=args.holdout_defect,
        random_seed=args.random_seed,
    )

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.save_path, index=False)
    log.info("\n✓ CSV saved → %s", args.save_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_tier1 = len(normal_concepts)
    n_tier2_fixed = len(all_concepts) - n_tier1
    n_tier2_disc  = len(discovered_generic)
    n_tier3 = sum(len(v) for v in refined_defect_map.values())

    log.info("\n" + "═" * 60)
    log.info("PIPELINE SUMMARY — %s", args.category)
    log.info("═" * 60)
    log.info("  Tier 1 normal concepts      : %d", n_tier1)
    log.info("  Tier 2 generic (fixed)      : %d", n_tier2_fixed)
    log.info("  Tier 2 generic (discovered) : %d", n_tier2_disc)
    log.info("  Tier 3 defect-specific      : %d", n_tier3)
    log.info("  Final vocabulary size       : %d", len(final_vocab))
    log.info("  Total images annotated      : %d", len(all_annotations))
    log.info("  CSV shape                   : %d rows × %d cols", *df.shape)
    log.info("  Holdout defect              : %s", args.holdout_defect or "none")
    log.info("  Stage 4                     : %s",
             "skipped (v2-mode)" if args.skip_stage4 else "on")
    log.info("  Loaded from stage           : %d", args.load_from_stage)
    log.info("  VLM audit (Stage 2d)        : %s",
             "off" if args.skip_vlm_audit else "on")
    log.info("  CLIP refs (Stage 3, P1-A)   : %d", args.n_normal_refs)
    log.info("  Visual grounding (Stage 4)  : %s",
             "on" if args.use_visual_grounding else "off")
    log.info("  Async workers (Stage 3)     : %d", args.n_workers)
    log.info("═" * 60)

    return df, normal_concepts, refined_defect_map