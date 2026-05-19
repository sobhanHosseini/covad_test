"""
atom_selector.py — Score SAE atoms by their discrimination power between
anomaly patches and normal patches.

For every atom in the 4096-dimensional SAE dictionary we compute:
    discrimination_score = mean_activation_anomaly - mean_activation_normal

Atoms with a positive score AND at least *min_count* anomaly activations are
kept; the top-N are returned sorted by score descending.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def select_anomaly_relevant_atoms(
    anomaly_patches: list[dict[str, Any]],
    normal_codes: torch.Tensor,
    min_count: int = 10,
    top_n: int = 50,
    save_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Score every SAE atom and return the top-N anomaly-discriminative ones.

    For each of the 4096 atoms:
        * mean activation on anomaly patches
        * mean activation on normal patches
        * discrimination_score = mean_anomaly − mean_normal
        * anomaly_count = number of anomaly patches where the atom fires (> 0)

    An atom is kept when:
        * discrimination_score > 0  (fires more on anomalies than normals)
        * anomaly_count ≥ min_count  (has enough evidence)

    Results are sorted by discrimination_score descending and the top *top_n*
    are returned.

    Args:
        anomaly_patches: Output of ``extract_anomaly_patches`` — list of dicts
            each containing ``"sae_code"`` (torch.Tensor of shape (4096,)).
        normal_codes: Tensor of shape ``(M, 4096)`` — SAE codes for normal
            patches, as returned by ``load_normal_patches``.
        min_count: Minimum number of anomaly patches an atom must activate on.
        top_n: How many atoms to return.
        save_path: If given, saves results as JSON to this path.

    Returns:
        List of dicts (length ≤ top_n), sorted by discrimination_score desc::

            {
                "atom_id":              int,
                "discrimination_score": float,
                "anomaly_count":        int,
                "mean_anomaly":         float,
                "mean_normal":          float,
                "top_patch_indices":    list[int],  # indices into anomaly_patches
            }
    """
    if not anomaly_patches:
        raise ValueError("anomaly_patches is empty — run extract_anomaly_patches first.")

    print(f"[atom_selector] Anomaly patches: {len(anomaly_patches)}")
    print(f"[atom_selector] Normal codes:    {normal_codes.shape}")

    # Stack all anomaly SAE codes → (N_anomaly, 4096)
    anomaly_codes = torch.stack([p["sae_code"] for p in anomaly_patches], dim=0).float()

    n_atoms = anomaly_codes.shape[1]
    print(f"[atom_selector] Scoring {n_atoms} atoms …")

    normal_codes = normal_codes.float()

    # Per-atom mean activations
    mean_anomaly = anomaly_codes.mean(dim=0)   # (4096,)
    mean_normal  = normal_codes.mean(dim=0)    # (4096,)
    disc_scores  = mean_anomaly - mean_normal  # (4096,)

    # Anomaly-patch activation count per atom (fires = activation > 0)
    fires_on_anomaly = (anomaly_codes > 0).sum(dim=0)  # (4096,)

    # Filter: score > 0 AND count >= min_count
    mask = (disc_scores > 0) & (fires_on_anomaly >= min_count)
    candidate_ids = mask.nonzero(as_tuple=True)[0].tolist()
    print(f"[atom_selector] Candidates after filtering: {len(candidate_ids)}")

    # Sort by discrimination score descending, take top_n
    candidate_ids.sort(key=lambda i: disc_scores[i].item(), reverse=True)
    selected_ids = candidate_ids[:top_n]

    results: list[dict[str, Any]] = []
    for atom_id in selected_ids:
        # Top-patch indices: anomaly patches with highest activation for this atom
        acts = anomaly_codes[:, atom_id]  # (N_anomaly,)
        top_patch_indices = acts.argsort(descending=True)[:9].tolist()

        results.append({
            "atom_id":              int(atom_id),
            "discrimination_score": float(disc_scores[atom_id]),
            "anomaly_count":        int(fires_on_anomaly[atom_id]),
            "mean_anomaly":         float(mean_anomaly[atom_id]),
            "mean_normal":          float(mean_normal[atom_id]),
            "top_patch_indices":    top_patch_indices,
        })

    print(f"[atom_selector] Returning {len(results)} atoms")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[atom_selector] Saved → {save_path}")

    return results
