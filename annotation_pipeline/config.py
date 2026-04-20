"""
config.py — All constants, dimension maps, and prompt templates.

Nothing in here imports from the rest of the package.
Edit prompts and thresholds here; do not scatter them across stage files.
"""

# ── Visual dimension normalisation ────────────────────────────────────────────
DIMENSION_NORMALIZE: dict[str, str] = {
    "surface texture": "texture",    "surface_texture": "texture",
    "texture": "texture",
    "surface color": "color",        "surface_color": "color",
    "color": "color",
    "shape/geometry": "shape",       "shape_geometry": "shape",
    "shape": "shape",                "geometry": "shape",
    "material finish": "finish",     "material_finish": "finish",
    "finish": "finish",
    "structural integrity": "structure", "structural_integrity": "structure",
    "structure": "structure",        "structural": "structure",
    "visible surface markings": "marking", "visible_surface_markings": "marking",
    "marking": "marking",            "markings": "marking",
    "surface markings": "marking",
}

# Per-dimension clustering thresholds (higher = harder to merge = more concepts kept)
DIMENSION_THRESHOLDS: dict[str, float] = {
    "color":     0.75,   # color concepts are specific; merge only near-identical ones
    "texture":   0.72,
    "shape":     0.70,
    "finish":    0.72,
    "structure": 0.60,   # damage concepts have high naming variance; merge liberally
    "marking":   0.65,
    "unknown":   0.68,
}

# ── Tier 2 — fixed generic anomaly concepts (always in vocab, never held out) ─
GENERIC_ANOMALY_CONCEPTS: list[dict] = [
    {
        "name": "surface_irregularity",
        "description": (
            "Any visible irregularity, roughness or discontinuity on the surface "
            "that deviates from expected normal appearance"
        ),
        "visual_dimension": "texture",
        "tier": "generic",
    },
    {
        "name": "color_deviation",
        "description": (
            "Any unexpected change in color, discoloration, staining or "
            "abnormal pigmentation compared to normal appearance"
        ),
        "visual_dimension": "color",
        "tier": "generic",
    },
    {
        "name": "structural_discontinuity",
        "description": (
            "Any break, crack, hole, fracture or loss of structural continuity in the material"
        ),
        "visual_dimension": "structure",
        "tier": "generic",
    },
    {
        "name": "texture_inconsistency",
        "description": (
            "Any localised region where surface texture is inconsistent "
            "with the surrounding normal texture"
        ),
        "visual_dimension": "texture",
        "tier": "generic",
    },
    {
        "name": "unexpected_surface_pattern",
        "description": (
            "Any pattern, marking, deposit or foreign material on the surface "
            "that should not be present on a normal specimen"
        ),
        "visual_dimension": "marking",
        "tier": "generic",
    },
]

# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_NORMAL_EXTRACTION = """\
You are an industrial quality control inspector specialising in visual inspection \
of manufactured parts.

You are examining a defect-free, normal-quality {category} from a production line.

Your task: extract exactly 12 visual attribute concepts that characterise the \
appearance of THIS specific specimen.

Rules:
- Each concept is a 2-4 word noun phrase describing a specific, concrete visual attribute
- Cover these six dimensions: surface texture, surface color, shape/geometry, \
material finish, structural integrity, and visible surface markings
- Concepts must describe what IS visually present — not what is absent
- Concepts must be visually grounded: a camera could detect them in an image region
- Do NOT include the object name or category name in any concept
- Do NOT use vague terms like "good quality", "normal appearance", "standard condition"
- Use snake_case for all concept names (words joined by underscores, all lowercase)

Return ONLY a valid JSON array of objects. No preamble, no explanation, no markdown.

Format:
[
  {{
    "name": "smooth_shell_surface",
    "description": "The outer shell is uniformly smooth with no visible grooves or roughness",
    "visual_dimension": "texture"
  }},
  ...
]

Provide exactly 12 concepts covering all 6 visual dimensions listed above."""


PROMPT_VLM_MERGE = """\
You are a vocabulary curator for an industrial inspection system.

Below are groups of visual attribute concepts clustered as semantically similar.
For each group, choose ONE canonical concept name that:
1. Is the most general and precise representative of the group
2. Uses exactly 2-4 words in snake_case
3. Is visually concrete — a camera sensor could detect it in an image

Groups to merge:
{json_groups}

Return ONLY a valid JSON object: {{"original_name": "canonical_name", ...}}
No preamble, no explanation."""


PROMPT_NORMAL_CONCEPT_AUDIT = """\
You are auditing a normal concept vocabulary for an industrial inspection system.
Category: {category}

These concepts were extracted from defect-free images.
Remove a concept ONLY IF it explicitly references a defect or its absence:
  "absence_of_surface_cracks" → REMOVE  |  "crack_free_shell" → REMOVE
  "no_visible_damage" → REMOVE
KEEP concepts describing normal physical properties:
  "structural_integrity" → KEEP  |  "intact_shell_structure" → KEEP
  "smooth_surface" → KEEP

Concepts to audit:
{concepts_json}

Return ONLY a valid JSON object: {{"concept_name": "keep"/"remove", ...}}
No preamble, no explanation."""


PROMPT_DEFECT_ANNOTATION = """\
You are an industrial quality control inspector performing comparative visual inspection.

You are given {n_images} images in order:
  Images 1 to {n_refs}: Defect-free, normal-quality {category} reference specimens.
  Image {query_idx} (last): A {category} specimen from the defect category "{defect_type}".

Normal concept vocabulary (established from defect-free specimens):
{normal_dict_json}

PART A — Annotate each normal concept for the LAST image (the defective specimen).
Use the reference images for comparison.
  true  = the attribute is clearly visible and intact in the defective image
  false = the attribute has been disrupted, damaged, or is clearly absent

PART B — Extract NEW defect-specific concepts.
List at most {max_new} NEW visual attributes visible in the defective image
that are NOT in the normal vocabulary. Focus on the most visually distinctive ones.
Do NOT pad with vague or redundant concepts.
Each must be:
  - Visually concrete (detectable in a specific image region)
  - Different from all concepts in the normal vocabulary
  - A 2-4 word snake_case noun phrase

Return ONLY this JSON structure:
{{
  "normal_concept_annotations": {{"concept_name_1": true, "concept_name_2": false, ...}},
  "new_defect_concepts": [
    {{
      "name": "linear_shell_fracture",
      "description": "A visible linear crack along the outer shell surface",
      "visual_dimension": "structure"
    }}
  ],
  "defect_category": "{defect_type}"
}}

Annotate ALL {n_concepts} concepts from the normal vocabulary in PART A."""


PROMPT_NORMAL_IMAGE_ANNOTATION = """\
You are an expert evaluating an industrial image to detect anomalies.
I provide an image of a {category}.
The image has been classified as normal — no visible defect or issue.
Choose which concepts you see among: {concept_list}
Output ONLY a JSON object: {{"concept_1": true, "concept_2": false, ...}}\
"""
