# MA-CBM Development Notes

## LLM Parser — needs fixing before scaling to all 15 categories

**Issue:** `vocabulary_builder._parse_llm_grouping()` is unreliable.
- In the hazelnut proof-of-concept, qwen3.5:9b produced 17 groups in one run
  and only 1 group in another run, on the same 47 input concepts.
- Root cause: the LLM occasionally changes its output format — uses markdown
  code fences, omits some `---` separators, or restructures the block layout.
- The current regex splits on `\n---\n` which is too strict.

**Before running on all 15 categories, fix:**
1. Strip markdown code fences (` ```...``` `) before parsing.
2. Split on `---` with optional surrounding whitespace:
   `re.split(r'\s*---\s*', text)` instead of `r'\n---\n'`.
3. Add a fallback: if fewer than 5 groups are parsed, log the raw response
   and retry once with `gemma4:e4b` (which tends to follow format more strictly).
4. Consider adding "Output ONLY the formatted list, no preamble" to the prompt.

**Impact if not fixed:** embedding dedup handles ~38 singletons instead of ~20
canonicals, so near-duplicate pairs at similarity 0.70–0.74 are not merged and
appear as separate singletons in the final vocabulary (affects target of 15-25).

---

## Concept head training — TODO

- Train one binary linear classifier per concept (16 for hazelnut).
- Positives: anomaly patches where member atoms activate strongly.
- Negatives: normal patches + anomaly patches from other concepts.
- Evaluate: per-concept binary accuracy on held-out anomaly patches.
- Implementation: `mac/concepts/concept_head_trainer.py`
- Script: `mac/scripts/04_train_concept_heads.py`
