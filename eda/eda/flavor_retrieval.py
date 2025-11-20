"""
Dish-level flavor similarity baseline.

Approach:
- Map recipe ingredients to FooDB food_ids via ingredient_lookup.
- Build compact chemistry vectors per food from chemistry_vectors.csv.
- Select top-N compounds globally to form a fixed embedding size.
- Embed recipes by averaging their ingredient embeddings.
- Retrieve similar recipes via cosine similarity.
- Evaluate with a simple hit@k: does a test recipe retrieve training recipes that share at least one ingredient?
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

RECIPES_PATH = PROCESSED_DIR / "recipes_ingredients_clean.csv"
ING_LOOKUP_PATH = PROCESSED_DIR / "ingredient_lookup.csv"
CHEMISTRY_VECTORS_PATH = PROCESSED_DIR / "chemistry_vectors.csv"
FIGURES_PATH = BASE_DIR / "figures"
FIGURES_PATH.mkdir(exist_ok=True, parents=True)

# Configuration
TOP_COMPOUNDS = 500
SAMPLE_RECIPES = 3000  # limit to keep runtime reasonable
RANDOM_STATE = 42
HIT_K = 10


def load_recipes(n: Optional[int] = None) -> pd.DataFrame:
    converters = {"ingredient_list": eval}
    df = pd.read_csv(RECIPES_PATH, converters=converters)
    if n:
        df = df.sample(n=min(n, len(df)), random_state=RANDOM_STATE)
    return df.reset_index(drop=True)


def load_lookup() -> Dict[str, int]:
    df = pd.read_csv(ING_LOOKUP_PATH)
    df = df.dropna(subset=["food_id"])
    return dict(zip(df["ingredient"], df["food_id"].astype(int)))


def load_chemistry_vectors() -> pd.DataFrame:
    cols = ["food_id", "compound_id", "tfidf"]
    return pd.read_csv(CHEMISTRY_VECTORS_PATH, usecols=cols)


def select_top_compounds(chem_df: pd.DataFrame, top_n: int) -> List[int]:
    totals = chem_df.groupby("compound_id")["tfidf"].sum().nlargest(top_n)
    return totals.index.astype(int).tolist()


def build_food_embeddings(chem_df: pd.DataFrame, compound_ids: List[int]) -> Dict[int, np.ndarray]:
    compound_index = {cid: idx for idx, cid in enumerate(compound_ids)}
    vectors: Dict[int, np.ndarray] = {}
    filtered = chem_df[chem_df["compound_id"].isin(compound_index)]

    for food_id, group in filtered.groupby("food_id"):
        vec = np.zeros(len(compound_ids), dtype=np.float32)
        for _, row in group.iterrows():
            vec[compound_index[int(row["compound_id"])]] = float(row["tfidf"])
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        vectors[int(food_id)] = vec
    return vectors


def embed_recipe(ingredients: List[str], ing_to_food: Dict[str, int], food_vecs: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    vecs = []
    for ing in ingredients:
        food_id = ing_to_food.get(ing)
        if food_id is None:
            continue
        food_vec = food_vecs.get(food_id)
        if food_vec is not None:
            vecs.append(food_vec)
    if not vecs:
        return None
    mat = np.vstack(vecs)
    recipe_vec = mat.mean(axis=0)
    norm = np.linalg.norm(recipe_vec)
    if norm > 0:
        recipe_vec /= norm
    return recipe_vec


def build_recipe_embeddings(recipes: pd.DataFrame, ing_to_food: Dict[str, int], food_vecs: Dict[int, np.ndarray]) -> Tuple[np.ndarray, List[int]]:
    embeddings = []
    keep_indices = []
    for idx, row in recipes.iterrows():
        emb = embed_recipe(row["ingredient_list"], ing_to_food, food_vecs)
        if emb is not None:
            embeddings.append(emb)
            keep_indices.append(idx)
    if not embeddings:
        return np.array([]), []
    return np.vstack(embeddings), keep_indices


def recall_at_k(train_vecs: np.ndarray, test_vecs: np.ndarray, train_sets: List[set], test_sets: List[set], k: int) -> float:
    sims = cosine_similarity(test_vecs, train_vecs)
    hits = 0
    total = len(test_sets)
    for i, scores in enumerate(sims):
        top_idx = np.argsort(scores)[::-1][:k]
        neighbor_sets = [train_sets[j] for j in top_idx]
        if any(test_sets[i] & ns for ns in neighbor_sets):
            hits += 1
    return hits / total if total else 0.0


def plot_pr_roc(scores: List[float], labels: List[int], prefix: str) -> Dict[str, float]:
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
    plt.title("Flavor Retrieval Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / f"{prefix}_pr_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Flavor Retrieval ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / f"{prefix}_roc_curve.png", dpi=300)
    plt.close()

    return {"pr_auc": float(pr_auc), "roc_auc": float(roc_auc)}


def evaluate() -> Dict[str, float]:
    recipes = load_recipes(SAMPLE_RECIPES)
    ing_to_food = load_lookup()
    chem_df = load_chemistry_vectors()
    top_compounds = select_top_compounds(chem_df, TOP_COMPOUNDS)
    food_vecs = build_food_embeddings(chem_df, top_compounds)

    recipe_embeddings, keep_idx = build_recipe_embeddings(recipes, ing_to_food, food_vecs)
    if recipe_embeddings.size == 0:
        return {"recall_at_10": 0.0, "embedded_recipes": 0, "note": "No recipes could be embedded; improve ingredient->FooDB mapping."}

    recipes = recipes.iloc[keep_idx].reset_index(drop=True)
    train_df, test_df, train_vecs, test_vecs = train_test_split(
        recipes,
        recipe_embeddings,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    train_sets = [set(ings) for ings in train_df["ingredient_list"]]
    test_sets = [set(ings) for ings in test_df["ingredient_list"]]

    sims = cosine_similarity(test_vecs, train_vecs)
    score_list: List[float] = []
    label_list: List[int] = []
    for i, scores in enumerate(sims):
        for j, score in enumerate(scores):
            label = 1 if test_sets[i] & train_sets[j] else 0
            score_list.append(float(score))
            label_list.append(int(label))

    r_at_k = recall_at_k(train_vecs, test_vecs, train_sets, test_sets, HIT_K)
    curve_metrics = plot_pr_roc(score_list, label_list, "flavor_retrieval")
    return {
        "recall_at_10": r_at_k,
        "embedded_recipes": recipe_embeddings.shape[0],
        "train_size": len(train_df),
        "test_size": len(test_df),
        "top_compounds": TOP_COMPOUNDS,
        **curve_metrics,
    }


def main() -> None:
    metrics = evaluate()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    import json

    main()
