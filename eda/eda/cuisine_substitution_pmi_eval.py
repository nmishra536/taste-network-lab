"""
Evaluation of culturally aware substitution using PMI-based positives.

Positive labels are ingredient pairs whose global PMI and average rating exceed thresholds, acting as proxy "good substitutions" regardless of whether they co-occur in the same recipe.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

PMI_SAMPLE_RECIPES = 1200

from cuisine_substitution import (
    HIT_K,
    EVAL_TOP,
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

PMI_THRESHOLD = 1.0
RATING_THRESHOLD = 4.0


def build_positive_pairs(graph_df: pd.DataFrame) -> Dict[int, Set[int]]:
    positives: Dict[int, Set[int]] = defaultdict(set)
    for _, row in graph_df.iterrows():
        pmi = row.get("pmi")
        rating = row.get("avg_rating")
        if pd.isna(pmi) or pd.isna(rating):
            continue
        if pmi < PMI_THRESHOLD or rating < RATING_THRESHOLD:
            continue
        a = int(row["food_id_a"])
        b = int(row["food_id_b"])
        positives[a].add(b)
        positives[b].add(a)
    return positives


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
    plt.title("PMI Substitution Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{prefix}_pr_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("PMI Substitution ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{prefix}_roc_curve.png", dpi=300)
    plt.close()

    return {"pr_auc": float(pr_auc), "roc_auc": float(roc_auc)}


def evaluate_with_positives(
    recipes: pd.DataFrame,
    ing_lookup: Dict[str, int],
    positives: Dict[int, Set[int]],
    graph_with_foods: pd.DataFrame,
    flavor_profiles: Dict[int, Dict[str, float]],
    food_groups: Dict[int, str],
    k: int = HIT_K,
    enforce_blocklist: bool = True,
) -> Dict[str, float]:
    hits = 0
    total = 0
    score_list: List[float] = []
    label_list: List[int] = []

    for _, row in recipes.iterrows():
        cuisine_tags = row["cuisine_tags"]
        mapped = [ing_lookup.get(ing) for ing in row["ingredient_list"] if ing_lookup.get(ing)]
        ingredient_set = set(mapped)
        if len(ingredient_set) < 2:
            continue

        for target in ingredient_set:
            positive_set = positives.get(target)
            if not positive_set:
                continue
            recs_full = recommend_substitutes(
                target,
                cuisine_tags,
                graph_with_foods,
                flavor_profiles,
                food_groups,
                k=k,
                enforce_blocklist=enforce_blocklist,
                return_all=True,
            )
            if not recs_full:
                continue
            recs = recs_full[:k]
            if any(cand_id in positive_set for cand_id, _, _ in recs):
                hits += 1
            total += 1

            for cand_id, _, score in recs_full[:EVAL_TOP]:
                label_list.append(1 if cand_id in positive_set else 0)
                score_list.append(score)

    metrics = {
        f"hit_at_{k}": hits / total if total else 0.0,
        "evaluated_targets": total,
    }
    metrics.update(plot_curves(label_list, score_list, "cuisine_substitution_pmi"))
    return metrics


def main() -> None:
    recipes = load_recipes(PMI_SAMPLE_RECIPES)
    ing_lookup = load_lookup()
    graph_df = load_graph_with_foods()
    flavor_profiles = load_flavor_profiles()
    food_groups = load_food_meta()
    positives = build_positive_pairs(graph_df)
    metrics = evaluate_with_positives(
        recipes,
        ing_lookup,
        positives,
        graph_df,
        flavor_profiles,
        food_groups,
        k=HIT_K,
        enforce_blocklist=False,
    )
    print(metrics)


if __name__ == "__main__":
    main()
