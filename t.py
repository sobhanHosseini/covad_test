"""
CONVAD — Hold-out Defect Experiment (v4-final)
===============================================
Sobhan Hosseini — MSc Thesis, University of Padova, 2025

Cross-referenced against the original CONVAD source code:
  trainers/trainer_cbm.py, trainer_stfpm.py
  evaluators/evaluator_cbm.py, evaluator_stfpm.py
  datasets/concept_dataset.py
  models/model_backbones.py, full_models.py
  main_scripts/cbm.py, stfpm.py, combined_branches.py
  utils/dataset_utils.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALL BUGS AND ISSUES FOUND (original v3 notebook vs source)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[BUG-1] Cell 7: BackboneModel missing required args (CRASH BUG)
  Original:  BackboneModel(num_classes=15, pretrained=True, bottleneck=False, backbone=BACKBONE)
  Problem:   BackboneModel.__init__ requires num_attr and freeze_parameters — no defaults.
             This raises TypeError and the notebook cannot even initialize.
  Fix:       Add num_attr=None, freeze_parameters=False. Use standard_model() factory instead,
             which is designed exactly for this use case.

[BUG-2] CBMEvaluator.evaluate() return mapping assumed wrong in v3 (found CORRECT on review)
  Source:    evaluator_cbm.py returns (auc_main, f1_main, mean_auc, f1_attr, main_preds) — 5 items.
  Notebook:  maps [0]=I_AUC, [1]=I_F1, [2]=C_AUC, [3]=C_F1 — CORRECT.
  Status:    No fix needed.

[BUG-3] Missing random seeds for full reproducibility
  Fix:       random.seed, np.random.seed, cuda seeds added globally at Cell 2.

[BUG-4] stratify= crash on tiny defect classes
  Fix:       _split_index_safe() with try/except fallback.

[BUG-5] drop_concepts filter absent — can cause extreme pos_weight values
  Source:    dataset_utils.py drops concepts in <10 images. With 70+ concepts from the new
             pipeline, some may appear in only 1-2 images → pos_weight = 279, numerically
             destabilising the BCE loss for that concept head.
  Fix:       MIN_CONCEPT_COUNT guard added. Default = 5 (softer than paper's 10 to avoid
             over-filtering with the richer new vocabulary). Set to 0 to disable.

[BUG-6] zero-positive concept imbalance returns ratio=0 from find_class_imbalance
  Source:    concept_dataset.py: num_positives==0 → imbalance_ratio.append(0)
             BCEWithLogitsLoss(pos_weight=tensor(0)) → model assigns zero weight to
             positives → effectively never predicts that concept as present.
  Fix:       Clamp per-concept imbalance ratio to a minimum of 1.0 before passing to trainer.

[BUG-7] STFPM printer pattern mismatch (pixel PRO format)
  Source:    evaluator_stfpm.py prints: "Pixel AUC = x, Pixel F1 = x, Pixel PRO = x, Pixel PR = x"
             All on one line, comma-separated.
  Fix:       Regex updated to match the actual printed format.

[BUG-8] Detection rate at fixed 0.5 threshold
  Fix:       Also compute at Youden-J optimal threshold from val set.

DATA PROTOCOL DIFFERENCES (paper vs notebook — intentional deviations):
  Paper split_dataframe: 80/10/10 on WHOLE dataframe (normal+anomalous mixed, stratified)
  Notebook:              70/10/20 separately for normal and anomalous
  Rationale:  Holdout experiment requires redistribution of images, so the paper's split
              cannot be used directly. 70/10/20 is adopted for consistency across all runs.
              The baseline uses the same 70/10/20 protocol so comparisons are apples-to-apples.
              If you want to reproduce the paper's numbers exactly, run the paper's CSV
              with split column set from the original MVTec split (all train/good → train,
              all test/* → test). This notebook does not do that.

CONFIRMED CORRECT vs source (no change needed):
  ✓ model.train_df — End2EndModel.state_dict() explicitly serialises/deserialises it.
  ✓ strict=False in teacher load_state_dict — safer than original stfpm.py (which omits it),
    matches combined_branches.py (the more recent reference).
  ✓ student pretrained=False — matches stfpm.py (combined_branches.py uses True, which is wrong
    for STFPM: student should start random so teacher-student gap is meaningful at test time).
  ✓ imbalance_main, _ = find_class_imbalance('main') — correctly unpacks (ratio, contamination).
  ✓ imbalance_attr = find_class_imbalance('attributes') — correctly gets list of per-concept ratios.
  ✓ CBMTrainer receives weight_attr as list — matches trainer_cbm.py: weight_attr[i] per concept.
  ✓ STFPMTrainer patience and scheduler params — accepted, confirmed in trainer_stfpm.py.
  ✓ STFPM val dataset includes anomalies (needed for pixel AUC early-stopping) — correct.
"""



















