"""
Evaluation of culturally aware substitution using recipe signature matching.

Positive labels are derived from near-identical recipes that differ by exactly one ingredient (signature matches), representing realistic substitution opportunities.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from cuisine_substitution import (
    SAMPLE_RECIPES,
    EVAL_TOP,
    HIT_K,
    load_recipes,
    load_lookup,
    load_graph_with_foods,
    load_flavor_profiles,
    load_food_meta,
    recommend_substitutes,
)

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

SIGNATURE_RECIPE_LIMIT = 20000


def map_recipe_ingredients(recipes: pd.DataFrame, ing_lookup: Dict[str, int]) -> List[Set[int]]:
    mapped_sets: List[Set[int]] = []
    for ingredients in recipes["ingredient_list"]:
        mapped = {ing_lookup.get(ing) for ing in ingredients if ing_lookup.get(ing)}
        if len(mapped) >= 3:
            mapped_sets.append(mapped)
        else:
            mapped_sets.append(set())
    return mapped_sets


def build_signature_map(recipes: pd.DataFrame, ing_lookup: Dict[str, int]) -> Dict[Tuple[int, ...], Set[int]]:
    signature_map: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
    mapped_sets = map_recipe_ingredients(recipes, ing_lookup)
    for ingredient_set in mapped_sets:
        if len(ingredient_set) < 3:
            continue
        for ingredient in ingredient_set:
            key = tuple(sorted(ingredient_set - {ingredient}))
            if len(key) < 2:
                continue
            signature_map[key].add(ingredient)
    return signature_map


def plot_curves(labels: List[int], scores: List[float], prefix: str) -> Dict[str, float]:
    if not labels:
        return {"pr_auc": 0.0, "roc_auc": 0.0}
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)
    precision, recall, _ = precision_recall_curve(labels_arr, scores_arr)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(labels_arr, scores_arr)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Signature Substitution Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{prefix}_pr_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Signature Substitution ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{prefix}_roc_curve.png", dpi=300)
    plt.close()

    return {"pr_auc": float(pr_auc), "roc_auc": float(roc_auc)}


def evaluate_with_signatures(
    recipes: pd.DataFrame,
    ing_lookup: Dict[str, int],
    signature_map: Dict[Tuple[int, ...], Set[int]],
    graph_with_foods: pd.DataFrame,
    flavor_profiles: Dict[int, Dict[str, float]],
    food_groups: Dict[int, str],
    k: int = HIT_K,
) -> Dict[str, float]:
    hits = 0
    total = 0
    labels: List[int] = []
    scores: List[float] = []

    mapped_sets = map_recipe_ingredients(recipes, ing_lookup)
    for ingredients, (_, row) in zip(mapped_sets, recipes.iterrows()):
        if len(ingredients) < 3:
            continue
        cuisine_tags = row["cuisine_tags"]
        for target in ingredients:
            key = tuple(sorted(ingredients - {target}))
            positives = signature_map.get(key)
            if not positives:
                continue
            recs_full = recommend_substitutes(
                target,
                cuisine_tags,
                graph_with_foods,
                flavor_profiles,
                food_groups,
                k=k,
                return_all=True,
            )
            if not recs_full:
                continue
            recs = recs_full[:k]
            if any(cand_id in positives for cand_id, _, _ in recs):
                hits += 1
            total += 1
            for cand_id, _, score in recs_full[:EVAL_TOP]:
                labels.append(1 if cand_id in positives else 0)
                scores.append(score)

    metrics = {
        f"hit_at_{k}": hits / total if total else 0.0,
        "evaluated_targets": total,
    }
    metrics.update(plot_curves(labels, scores, "cuisine_substitution_signature"))
    return metrics


def main() -> None:
    recipes_full = load_recipes()
    ing_lookup = load_lookup()
    signature_map = build_signature_map(recipes_full.head(SIGNATURE_RECIPE_LIMIT), ing_lookup)

    eval_recipes = recipes_full.sample(n=min(SAMPLE_RECIPES, len(recipes_full)), random_state=42).reset_index(drop=True)
    graph_df = load_graph_with_foods()
    flavor_profiles = load_flavor_profiles()
    food_groups = load_food_meta()

    metrics = evaluate_with_signatures(
        eval_recipes,
        ing_lookup,
        signature_map,
        graph_df,
        flavor_profiles,
        food_groups,
        k=HIT_K,
    )
    print(metrics)


if __name__ == "__main__":
    main()
