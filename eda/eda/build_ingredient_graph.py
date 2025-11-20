from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter, defaultdict
from difflib import get_close_matches
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FOODB_DIR = DATA_DIR / "foodb_2020_04_07_csv"
PROCESSED_DIR = DATA_DIR / "processed"

RECIPES_PATH = PROCESSED_DIR / "recipes_ingredients_clean.csv"
INTERACTIONS_PATH = PROCESSED_DIR / "recipe_interactions_clean.csv"
CONTENT_PATH = FOODB_DIR / "Content.csv"
FOOD_PATH = FOODB_DIR / "Food.csv"
FLAVOR_LOOKUP_PATH = PROCESSED_DIR / "compound_flavor_lookup.csv"

INGREDIENT_PATTERN = re.compile(r"[^a-z0-9\s/]")

FOOD_COLS = ["id", "name", "food_group", "food_subgroup", "category", "public_id"]
CONTENT_COLS = ["food_id", "source_id", "source_type", "standard_content"]

MANUAL_SYNONYMS = {
    "green onions": "green onion",
    "spring onion": "green onion",
    "scallions": "green onion",
    "scallion": "green onion",
    "chopped onion": "onion",
    "sweet onion": "onion",
    "yellow onion": "onion",
    "white onion": "onion",
    "red onion": "onion",
    "kosher salt": "salt",
    "sea salt": "salt",
    "coarse salt": "salt",
    "fine sea salt": "salt",
    "unsalted butter": "butter",
    "salted butter": "butter",
    "sweet butter": "butter",
    "whole milk": "milk",
    "2 percent milk": "milk",
    "skim milk": "milk",
    "evaporated milk": "milk",
    "black pepper": "pepper",
    "ground black pepper": "pepper",
    "freshly ground pepper": "pepper",
    "powdered sugar": "icing sugar",
    "confectioners sugar": "icing sugar",
    "icing sugar": "icing sugar",
    "all-purpose flour": "all purpose flour",
    "plain flour": "all purpose flour",
    "ap flour": "all purpose flour",
    "self rising flour": "self raising flour",
    "self-rising flour": "self raising flour",
    "extra virgin olive oil": "olive oil",
    "vegetable oil": "vegetable oil",
    "canola oil": "canola oil",
    "cooking spray": "vegetable oil cooking spray",
    "garlic": "garlic",
    "garlic clove": "garlic",
}

DESCRIPTORS = {
    "fresh",
    "dried",
    "ground",
    "chopped",
    "sliced",
    "minced",
    "crushed",
    "shredded",
    "grated",
    "cooked",
    "raw",
    "boneless",
    "skinless",
    "organic",
    "unsalted",
    "salted",
    "reduced",
    "fatfree",
    "lowfat",
    "large",
    "small",
    "medium",
    "extra",
    "freshly",
    "lean",
    "thickcut",
    "softened",
    "melted",
    "room",
}


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def normalize_token(token: str) -> str:
    if not isinstance(token, str):
        return ""
    cleaned = token.lower().strip()
    cleaned = cleaned.replace("-", " ")
    cleaned = INGREDIENT_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def simplify_token(token: str) -> str:
    words = token.split()
    words = [w for w in words if w not in DESCRIPTORS]
    simplified = " ".join(words) if words else token
    simplified = simplified.strip()
    # crude singularization
    if simplified.endswith("ies"):
        simplified = simplified[:-3] + "y"
    elif simplified.endswith("es") and len(simplified) > 4:
        simplified = simplified[:-2]
    elif simplified.endswith("s") and len(simplified) > 3:
        simplified = simplified[:-1]
    return simplified.strip()


def load_recipes() -> pd.DataFrame:
    if not RECIPES_PATH.exists():
        raise FileNotFoundError(f"Missing recipes at {RECIPES_PATH}")
    converters = {"ingredient_list": ast.literal_eval}
    recipes = pd.read_csv(RECIPES_PATH, converters=converters)
    recipes["ingredient_list"] = recipes["ingredient_list"].apply(
        lambda items: [normalize_token(item) for item in items if normalize_token(item)]
    )
    return recipes


def load_interactions() -> pd.DataFrame:
    if not INTERACTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing interactions at {INTERACTIONS_PATH}")
    interactions = pd.read_csv(INTERACTIONS_PATH)
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    return interactions


def load_food_lookup() -> pd.DataFrame:
    if not FOOD_PATH.exists():
        raise FileNotFoundError(f"Missing FooDB Food.csv at {FOOD_PATH}")
    food = pd.read_csv(FOOD_PATH, usecols=FOOD_COLS)
    food["normalized_name"] = food["name"].apply(normalize_token)
    food = food.dropna(subset=["normalized_name"])
    return food


def build_ingredient_lookup(recipes: pd.DataFrame, food: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    unique_ingredients = sorted({ing for items in recipes["ingredient_list"] for ing in items})
    manual_map = {normalize_token(k): normalize_token(v) for k, v in MANUAL_SYNONYMS.items()}

    normalized_to_food = (
        food.sort_values(["normalized_name", "food_group"])
        .drop_duplicates(subset=["normalized_name"])
        .set_index("normalized_name")
    )
    normalized_names = normalized_to_food.index.tolist()

    rows = []
    ingredient_to_food: Dict[str, int] = {}

    for ingredient in unique_ingredients:
        match_type = "unmatched"
        normalized = ingredient
        normalized_manual = manual_map.get(normalized, normalized)
        if normalized_manual != normalized:
            match_type = "manual"
            normalized = normalized_manual

        candidates = [
            normalized,
            simplify_token(normalized),
            simplify_token(simplify_token(normalized)),
        ]
        food_row = None
        for candidate in candidates:
            if candidate and candidate in normalized_to_food.index:
                potential = normalized_to_food.loc[candidate]
                food_row = potential.iloc[0] if isinstance(potential, pd.DataFrame) else potential
                match_type = "manual_synonym" if match_type == "manual" else "exact"
                break

        if food_row is None:
            fuzzy_target = candidates[-1] or normalized
            match = get_close_matches(fuzzy_target, normalized_names, n=1, cutoff=0.93)
            if match:
                potential = normalized_to_food.loc[match[0]]
                food_row = potential.iloc[0] if isinstance(potential, pd.DataFrame) else potential
                match_type = "fuzzy"

        if food_row is not None:
            rows.append(
                {
                    "ingredient": ingredient,
                    "normalized_ingredient": normalized,
                    "food_id": int(food_row["id"]),
                    "food_name": food_row["name"],
                    "food_group": food_row["food_group"],
                    "food_subgroup": food_row["food_subgroup"],
                    "category": food_row["category"],
                    "match_type": match_type,
                }
            )
            ingredient_to_food[ingredient] = int(food_row["id"])
        else:
            rows.append(
                {
                    "ingredient": ingredient,
                    "normalized_ingredient": normalized,
                    "food_id": math.nan,
                    "food_name": None,
                    "food_group": None,
                    "food_subgroup": None,
                    "category": None,
                    "match_type": "unmatched",
                }
            )

    lookup_df = pd.DataFrame(rows)
    lookup_df.to_csv(PROCESSED_DIR / "ingredient_lookup.csv", index=False)
    return lookup_df, ingredient_to_food


def load_compound_content() -> pd.DataFrame:
    if not CONTENT_PATH.exists():
        raise FileNotFoundError(f"Missing FooDB Content.csv at {CONTENT_PATH}")
    content = pd.read_csv(
        CONTENT_PATH,
        usecols=CONTENT_COLS,
        low_memory=False,
    )
    content["source_type"] = content["source_type"].astype(str).str.lower()
    compound_content = (
        content.loc[content["source_type"] == "compound", ["food_id", "source_id", "standard_content"]]
        .dropna(subset=["food_id", "source_id"])
        .copy()
    )
    compound_content["food_id"] = compound_content["food_id"].astype("int64")
    compound_content["source_id"] = compound_content["source_id"].astype("int64")
    compound_content["standard_content"] = pd.to_numeric(compound_content["standard_content"], errors="coerce")
    compound_content = compound_content.dropna(subset=["standard_content"])
    return compound_content


def build_chemistry_vectors(compound_content: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, set]]:
    doc_freq = compound_content.groupby("source_id")["food_id"].nunique()
    num_foods = compound_content["food_id"].nunique()
    idf = np.log((1 + num_foods) / (1 + doc_freq)) + 1.0

    compound_content = compound_content.join(idf.rename("idf"), on="source_id")
    compound_content["tf"] = np.log1p(compound_content["standard_content"].clip(lower=0))
    compound_content["tfidf"] = compound_content["tf"] * compound_content["idf"]

    chemistry_vectors = compound_content[["food_id", "source_id", "tfidf"]].copy()
    chemistry_vectors.rename(columns={"source_id": "compound_id"}, inplace=True)
    chemistry_vectors.to_csv(PROCESSED_DIR / "chemistry_vectors.csv", index=False)

    compound_sets: Dict[int, set] = {}
    for food_id, group in chemistry_vectors.groupby("food_id"):
        top_compounds = group.sort_values("tfidf", ascending=False).head(200)["compound_id"].astype(int)
        compound_sets[int(food_id)] = set(top_compounds.tolist())

    return chemistry_vectors, compound_sets


def build_flavor_profiles(compound_content: pd.DataFrame) -> pd.DataFrame:
    if not FLAVOR_LOOKUP_PATH.exists():
        return pd.DataFrame()

    flavor_lookup = pd.read_csv(FLAVOR_LOOKUP_PATH)
    if "canonical_flavor" not in flavor_lookup.columns:
        flavor_lookup = flavor_lookup.rename(columns={"flavor_label": "canonical_flavor"})
    flavor_lookup = flavor_lookup.dropna(subset=["canonical_flavor"])

    flavor_df = compound_content.merge(
        flavor_lookup[["compound_id", "canonical_flavor"]],
        left_on="source_id",
        right_on="compound_id",
        how="left",
    )
    flavor_df = flavor_df.dropna(subset=["canonical_flavor"])
    if flavor_df.empty:
        return flavor_df

    num_foods = compound_content["food_id"].nunique()
    doc_freq = flavor_df.groupby("canonical_flavor")["food_id"].nunique()
    idf = np.log((1 + num_foods) / (1 + doc_freq)) + 1.0

    flavor_df = flavor_df.join(idf.rename("idf"), on="canonical_flavor")
    flavor_df["tf"] = np.log1p(flavor_df["standard_content"].clip(lower=0))
    flavor_df["score"] = flavor_df["tf"] * flavor_df["idf"]

    profile = (
        flavor_df.groupby(["food_id", "canonical_flavor"])["score"]
        .sum()
        .reset_index()
        .rename(columns={"canonical_flavor": "flavor", "score": "tfidf"})
    )
    profile.to_csv(PROCESSED_DIR / "food_flavor_profile.csv", index=False)
    return profile


def compute_recipe_ratings(recipes: pd.DataFrame, interactions: pd.DataFrame) -> pd.Series:
    recipe_avg = interactions.groupby("recipe_id")["rating"].mean()
    recipes = recipes.merge(recipe_avg.rename("mean_rating"), on="recipe_id", how="left")
    recipes["mean_rating"] = recipes["mean_rating"].fillna(recipes["mean_rating"].median())
    return recipes["mean_rating"]


def build_pair_stats(
    recipes: pd.DataFrame,
    mean_ratings: pd.Series,
    ingredient_to_food: Dict[str, int],
    compound_sets: Dict[int, set],
) -> pd.DataFrame:
    ingredient_freq = Counter()
    pair_counts = Counter()
    pair_rating_sums = Counter()

    valid_recipe_mask = []
    for idx, ingredients in enumerate(recipes["ingredient_list"]):
        mapped = sorted({ingredient_to_food.get(ing) for ing in ingredients if ingredient_to_food.get(ing)})
        mapped = [ing for ing in mapped if ing is not None]
        valid_recipe_mask.append(mapped)
        for food_id in set(mapped):
            ingredient_freq[food_id] += 1
        if len(mapped) < 2:
            continue
        rating = mean_ratings.iloc[idx]
        for a, b in combinations(mapped, 2):
            key = tuple(sorted((a, b)))
            pair_counts[key] += 1
            pair_rating_sums[key] += rating

    total_recipes = len(recipes)

    rows = []
    for (a, b), count in pair_counts.items():
        freq_a = ingredient_freq[a]
        freq_b = ingredient_freq[b]
        pmi = math.log((count * total_recipes) / (freq_a * freq_b)) if freq_a and freq_b else math.nan
        avg_rating = pair_rating_sums[(a, b)] / count if count else math.nan
        novelty = math.log1p(total_recipes / count)

        compounds_a = compound_sets.get(a, set())
        compounds_b = compound_sets.get(b, set())
        if compounds_a and compounds_b:
            intersection = compounds_a & compounds_b
            union = compounds_a | compounds_b
            chemistry_overlap = len(intersection) / len(union) if union else 0.0
            shared_compounds = len(intersection)
        else:
            chemistry_overlap = math.nan
            shared_compounds = 0

        rows.append(
            {
                "food_id_a": a,
                "food_id_b": b,
                "pair_count": count,
                "freq_a": freq_a,
                "freq_b": freq_b,
                "pmi": pmi,
                "avg_rating": avg_rating,
                "novelty_score": novelty,
                "chemistry_overlap": chemistry_overlap,
                "shared_compounds": shared_compounds,
            }
        )

    pair_df = pd.DataFrame(rows)
    if not pair_df.empty:
        pair_df = pair_df.sort_values("pair_count", ascending=False)
    pair_df.to_csv(PROCESSED_DIR / "ingredient_graph.csv", index=False)
    pd.Series(ingredient_freq).rename("recipe_frequency").to_csv(
        PROCESSED_DIR / "ingredient_recipe_frequency.csv", header=True
    )
    return pair_df


def attach_food_metadata(pair_df: pd.DataFrame, food_lookup: pd.DataFrame, ingredient_lookup: pd.DataFrame) -> pd.DataFrame:
    food_meta = food_lookup.rename(
        columns={
            "id": "food_id",
            "name": "food_name",
            "food_group": "food_group_name",
        }
    )[["food_id", "food_name", "food_group_name"]]
    pair_df = pair_df.merge(
        food_meta.rename(columns=lambda c: f"{c}_a"),
        left_on="food_id_a",
        right_on="food_id_a",
        how="left",
    )
    pair_df = pair_df.merge(
        food_meta.rename(columns=lambda c: f"{c}_b"),
        left_on="food_id_b",
        right_on="food_id_b",
        how="left",
    )
    pair_df.to_csv(PROCESSED_DIR / "ingredient_graph_with_foods.csv", index=False)
    return pair_df


def main() -> None:
    ensure_dirs()
    recipes = load_recipes()
    interactions = load_interactions()
    food = load_food_lookup()
    lookup_df, ingredient_to_food = build_ingredient_lookup(recipes, food)
    compound_content = load_compound_content()
    chemistry_vectors, compound_sets = build_chemistry_vectors(compound_content)
    build_flavor_profiles(compound_content)
    mean_ratings = compute_recipe_ratings(recipes, interactions)
    pair_df = build_pair_stats(recipes, mean_ratings, ingredient_to_food, compound_sets)
    attach_food_metadata(pair_df, food, lookup_df)

    summary = {
        "ingredients_total": len(lookup_df),
        "ingredients_matched": int(lookup_df["food_id"].notna().sum()),
        "match_rate": float(lookup_df["food_id"].notna().mean()),
        "pairs": len(pair_df),
    }
    with (PROCESSED_DIR / "ingredient_graph_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("Ingredient lookup, chemistry vectors, and graph statistics generated.")


if __name__ == "__main__":
    main()
