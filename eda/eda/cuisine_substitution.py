"""
Culturally aware ingredient substitution recommender.

Approach:
- Use Food.com cuisines + ingredient graph edges (PMI, ratings, novelty, chemistry overlap).
- Apply cuisine-specific blocklists (e.g., avoid beef/pork for Indian, avoid pork/alcohol for halal-ish cuisines).
- Score candidate substitutes per target ingredient.
- Evaluate with a leave-one-in heuristic: for each ingredient in a recipe, see if recommended substitutes include any other ingredient already in that recipe (proxy for plausibility). Report hit@k and share of blocked suggestions.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

RECIPES_PATH = PROCESSED_DIR / "recipes_ingredients_clean.csv"
ING_LOOKUP_PATH = PROCESSED_DIR / "ingredient_lookup.csv"
GRAPH_PATH = PROCESSED_DIR / "ingredient_graph.csv"
GRAPH_WITH_FOODS_PATH = PROCESSED_DIR / "ingredient_graph_with_foods.csv"
FOOD_META_PATH = PROCESSED_DIR / "food_compound_summary.csv"
FLAVOR_PROFILE_PATH = PROCESSED_DIR / "food_flavor_profile.csv"
FIGURES_PATH = BASE_DIR / "figures"
FIGURES_PATH.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42
SAMPLE_RECIPES = 1500
HIT_K = 5
EVAL_TOP = 20

ALL_CUISINES = [
    "unspecified",
    "north-american",
    "american",
    "european",
    "asian",
    "italian",
    "mexican",
    "southern-united-states",
    "african",
    "greek",
    "middle-eastern",
    "indian",
    "french",
    "chinese",
    "caribbean",
    "english",
    "central-american",
    "scandinavian",
    "spanish",
    "german",
    "cajun",
    "thai",
    "moroccan",
    "japanese",
    "irish",
    "portuguese",
    "swedish",
    "vietnamese",
    "hawaiian",
    "turkish",
    "austrian",
    "cuban",
    "danish",
    "filipino",
    "hungarian",
    "korean",
    "native-american",
    "puerto-rican",
]

HALAL_BLOCKS = {
    "pork",
    "bacon",
    "ham",
    "pepperoni",
    "prosciutto",
    "lard",
    "gelatin",
    "alcohol",
    "rum",
    "wine",
    "brandy",
    "vodka",
    "gin",
    "whiskey",
    "bourbon",
}

HINDU_BLOCKS = HALAL_BLOCKS | {"beef", "steak", "veal"}

KOSHER_BLOCKS = {
    "pork",
    "bacon",
    "ham",
    "pepperoni",
    "prosciutto",
    "lard",
    "gelatin",
    "shellfish",
    "shrimp",
    "lobster",
    "crab",
    "clam",
    "oyster",
    "scallop",
}

CUISINE_BLOCKLIST = {tag: set() for tag in ALL_CUISINES}
CUISINE_BLOCKLIST.update(
    {
        "indian": HINDU_BLOCKS,
        "middle-eastern": HALAL_BLOCKS,
        "moroccan": HALAL_BLOCKS,
        "turkish": HALAL_BLOCKS,
        "african": {"bushmeat"},
        "asian": set(),
        "italian": set(),
        "mexican": set(),
        "southern-united-states": set(),
        "greek": set(),
        "french": set(),
        "chinese": set(),
        "caribbean": set(),
        "english": set(),
        "central-american": set(),
        "scandinavian": set(),
        "spanish": set(),
        "german": set(),
        "cajun": set(),
        "thai": set(),
        "japanese": set(),
        "irish": set(),
        "portuguese": set(),
        "swedish": set(),
        "vietnamese": set(),
        "hawaiian": set(),
        "austrian": set(),
        "cuban": set(),
        "danish": set(),
        "filipino": set(),
        "hungarian": set(),
        "korean": set(),
        "native-american": set(),
        "puerto-rican": set(),
        "north-american": set(),
        "american": set(),
        "european": set(),
        "unspecified": set(),
    }
)

GLOBAL_BLOCK = {"rat", "bat", "cat"}


def normalize_name(name: str) -> str:
    return name.lower().strip() if isinstance(name, str) else ""


def load_recipes(n: Optional[int] = None) -> pd.DataFrame:
    converters = {"ingredient_list": eval, "cuisine_tags": eval}
    df = pd.read_csv(RECIPES_PATH, converters=converters)
    if n:
        df = df.sample(n=min(n, len(df)), random_state=RANDOM_STATE)
    return df.reset_index(drop=True)


def load_lookup() -> Dict[str, int]:
    df = pd.read_csv(ING_LOOKUP_PATH)
    df = df.dropna(subset=["food_id"])
    return dict(zip(df["ingredient"], df["food_id"].astype(int)))


def load_graph() -> pd.DataFrame:
    df = pd.read_csv(GRAPH_PATH)
    return df


def load_graph_with_foods() -> pd.DataFrame:
    df = pd.read_csv(GRAPH_WITH_FOODS_PATH)
    # Normalize names for matching blocklists
    for col in ["food_name_a", "food_name_b"]:
        df[col] = df[col].apply(normalize_name)
    return df


def load_food_meta() -> Dict[int, str]:
    if not FOOD_META_PATH.exists():
        return {}
    df = pd.read_csv(FOOD_META_PATH)
    return dict(zip(df["food_id"], df["group"]))


def load_flavor_profiles() -> Dict[int, Dict[str, float]]:
    if not FLAVOR_PROFILE_PATH.exists():
        return {}
    df = pd.read_csv(FLAVOR_PROFILE_PATH)
    profiles: Dict[int, Dict[str, float]] = defaultdict(dict)
    for _, row in df.iterrows():
        profiles[int(row["food_id"])][row["flavor"]] = float(row["tfidf"])
    return profiles


def cosine_from_profile(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[f] * b[f] for f in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def plot_curves(scores: List[float], labels: List[int], prefix: str) -> Dict[str, float]:
    if not scores:
        return {"pr_auc": 0.0, "roc_auc": 0.0}
    labels_array = np.array(labels)
    scores_array = np.array(scores)
    precision, recall, _ = precision_recall_curve(labels_array, scores_array)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(labels_array, scores_array)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Substitution Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / f"{prefix}_pr_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Substitution ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / f"{prefix}_roc_curve.png", dpi=300)
    plt.close()

    return {"pr_auc": float(pr_auc), "roc_auc": float(roc_auc)}


def block_for_cuisine(candidate_name: str, cuisine_tags: Iterable[str]) -> bool:
    cname = normalize_name(candidate_name)
    banned_terms = set(GLOBAL_BLOCK)
    for tag in cuisine_tags:
        banned_terms |= CUISINE_BLOCKLIST.get(normalize_name(tag), set())
    return any(term in cname for term in banned_terms)


def recommend_substitutes(
    target_food_id: int,
    cuisine_tags: Iterable[str],
    graph_with_foods: pd.DataFrame,
    flavor_profiles: Dict[int, Dict[str, float]],
    food_groups: Dict[int, str],
    k: int = 5,
    weights: Tuple[float, float, float, float, float, float, float] = (1.0, 0.5, 0.5, 0.2, 0.1, 1.0, 0.3),
    enforce_blocklist: bool = True,
    return_all: bool = False,
) -> List[Tuple[int, str, float]]:
    w_pmi, w_rating, w_chem, w_novelty, w_count, w_flavor, w_group = weights

    subset_a = graph_with_foods[graph_with_foods["food_id_a"] == target_food_id]
    subset_b = graph_with_foods[graph_with_foods["food_id_b"] == target_food_id]

    candidates = []
    for _, row in subset_a.iterrows():
        candidates.append(
            (
                int(row["food_id_b"]),
                row["food_name_b"],
                row.get("food_group_name_b"),
                row["pmi"],
                row["avg_rating"],
                row["chemistry_overlap"],
                row["novelty_score"],
                row["pair_count"],
            )
        )
    for _, row in subset_b.iterrows():
        candidates.append(
            (
                int(row["food_id_a"]),
                row["food_name_a"],
                row.get("food_group_name_a"),
                row["pmi"],
                row["avg_rating"],
                row["chemistry_overlap"],
                row["novelty_score"],
                row["pair_count"],
            )
        )

    target_flavor = flavor_profiles.get(target_food_id, {})
    target_group = food_groups.get(target_food_id)

    scored = []
    for cand_id, cand_name, cand_group, pmi, rating, chem, novelty, count in candidates:
        if enforce_blocklist and block_for_cuisine(cand_name, cuisine_tags):
            continue
        pmi = 0.0 if math.isnan(pmi) else pmi
        rating = 0.0 if math.isnan(rating) else rating
        chem = 0.0 if math.isnan(chem) else chem
        novelty = 0.0 if math.isnan(novelty) else novelty
        flavor_sim = cosine_from_profile(target_flavor, flavor_profiles.get(cand_id, {}))
        same_group = 1.0 if target_group and cand_group and target_group == cand_group else 0.0
        score = (
            w_pmi * pmi
            + w_rating * rating
            + w_chem * chem
            + w_novelty * novelty
            + w_count * math.log1p(count)
            + w_flavor * flavor_sim
            + w_group * same_group
        )
        scored.append((cand_id, cand_name, score))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored if return_all else scored[:k]


def evaluate_hit_rate(
    recipes: pd.DataFrame,
    ing_to_food: Dict[str, int],
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
        mapped = [ing_to_food.get(ing) for ing in row["ingredient_list"] if ing_to_food.get(ing) is not None]
        if len(mapped) < 2:
            continue
        ingredient_set = set(mapped)
        for target in ingredient_set:
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
            rec_ids = {rid for rid, _, _ in recs}
            if rec_ids & (ingredient_set - {target}):
                hits += 1
            total += 1
            positives = ingredient_set - {target}
            for cand_id, _, score in recs_full[:EVAL_TOP]:
                label = 1 if cand_id in positives else 0
                score_list.append(score)
                label_list.append(label)
    curve_metrics = plot_curves(score_list, label_list, "cuisine_substitution")
    return {
        "hit_at_%d" % k: hits / total if total else 0.0,
        "evaluated_targets": total,
        **curve_metrics,
    }


def main() -> None:
    recipes = load_recipes(SAMPLE_RECIPES)
    ing_to_food = load_lookup()
    graph_with_foods = load_graph_with_foods()
    flavor_profiles = load_flavor_profiles()
    food_groups = load_food_meta()
    metrics = evaluate_hit_rate(recipes, ing_to_food, graph_with_foods, flavor_profiles, food_groups, k=HIT_K)
    print(metrics)


if __name__ == "__main__":
    main()
