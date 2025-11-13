from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FOODB_DIR = DATA_DIR / "foodb_2020_04_07_csv"
RECIPES_PATH = DATA_DIR / "RAW_recipes.csv"
INTERACTIONS_PATH = DATA_DIR / "RAW_interactions.csv"
PROCESSED_DIR = DATA_DIR / "processed"

FOOD_COLS = ["id", "name", "food_group", "food_subgroup", "category", "public_id"]
COMPOUND_COLS = [
    "id",
    "public_id",
    "name",
    "kingdom",
    "superklass",
    "klass",
    "subklass",
    "moldb_mono_mass",
    "moldb_inchi",
]
CONTENT_COLS = ["food_id", "source_id", "source_type", "standard_content"]
FLAVOR_COLS = ["id", "name", "flavor_group"]
COMP_FLAVOR_COLS = ["compound_id", "flavor_id"]
CANONICAL_FLAVOR_PATH = BASE_DIR / "flavor_canonical_lookup.csv"
RECIPE_COLS = [
    "id",
    "name",
    "submitted",
    "minutes",
    "tags",
    "nutrition",
    "n_steps",
    "ingredients",
    "n_ingredients",
]

CUISINE_TOKENS = {
    "african",
    "american",
    "asian",
    "austrian",
    "british",
    "cajun",
    "caribbean",
    "central-american",
    "chinese",
    "cuban",
    "danish",
    "eastern-european",
    "english",
    "european",
    "filipino",
    "french",
    "german",
    "greek",
    "hawaiian",
    "hungarian",
    "indian",
    "irish",
    "italian",
    "japanese",
    "korean",
    "latin-american",
    "mediterranean",
    "mexican",
    "middle-eastern",
    "moroccan",
    "native-american",
    "north-american",
    "persian",
    "portuguese",
    "puerto-rican",
    "scandinavian",
    "southern-united-states",
    "spanish",
    "swedish",
    "thai",
    "turkish",
    "vietnamese",
}

INGREDIENT_PATTERN = re.compile(r"[^a-z0-9\s/]")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_literal_eval(value: str | float | int) -> List[str]:
    if isinstance(value, float) and math.isnan(value):
        return []
    if not isinstance(value, str):
        return []
    stripped = value.strip()
    if not stripped:
        return []
    try:
        parsed = ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def normalize_ingredient(token: str) -> str:
    cleaned = token.lower().strip()
    cleaned = cleaned.replace("-", " ")
    cleaned = INGREDIENT_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def load_foodb_tables() -> Dict[str, pd.DataFrame]:
    if not FOODB_DIR.exists():
        raise FileNotFoundError(f"Expected FoodDB directory at {FOODB_DIR}")

    food = pd.read_csv(FOODB_DIR / "Food.csv", usecols=FOOD_COLS)
    compound_dtypes = {
        "public_id": "string",
        "name": "string",
    "kingdom": "string",
    "superklass": "string",
    "klass": "string",
    "subklass": "string",
    "moldb_mono_mass": "string",
    "moldb_inchi": "float64",
}
    compound = pd.read_csv(FOODB_DIR / "Compound.csv", usecols=COMPOUND_COLS, dtype=compound_dtypes)
    compound["mono_mass"] = pd.to_numeric(compound["moldb_mono_mass"], errors="coerce")
    compound["mono_mass"] = compound["mono_mass"].fillna(compound["moldb_inchi"])
    content = pd.read_csv(
        FOODB_DIR / "Content.csv",
        usecols=CONTENT_COLS,
        low_memory=False,
    )

    content["source_type"] = content["source_type"].astype(str).str.lower()
    compound_content = (
        content.loc[content["source_type"] == "compound", ["food_id", "source_id", "standard_content"]]
        .dropna(subset=["food_id", "source_id"])
        .copy()
    )
    compound_content["food_id"] = compound_content["food_id"].astype(np.int64)
    compound_content["source_id"] = compound_content["source_id"].astype(np.int64)
    compound_content["standard_content"] = pd.to_numeric(
        compound_content["standard_content"],
        errors="coerce",
    )
    del content

    flavor = pd.read_csv(FOODB_DIR / "Flavor.csv", usecols=FLAVOR_COLS)
    comp_flavor = pd.read_csv(FOODB_DIR / "CompoundsFlavor.csv", usecols=COMP_FLAVOR_COLS)
    flavor_lookup = flavor.rename(
        columns={
            "id": "flavor_id",
            "name": "flavor_name",
        }
    )
    if CANONICAL_FLAVOR_PATH.exists():
        canonical = pd.read_csv(CANONICAL_FLAVOR_PATH)
        canonical = canonical.rename(columns={"id": "flavor_id", "canonical_name": "canonical_flavor_name"})
        flavor_lookup = flavor_lookup.merge(
            canonical[["flavor_id", "canonical_flavor_name"]],
            on="flavor_id",
            how="left",
        )
    else:
        flavor_lookup["canonical_flavor_name"] = pd.NA
    flavor_lookup["flavor_label"] = flavor_lookup["canonical_flavor_name"].fillna(flavor_lookup["flavor_name"])
    flavor_lookup = flavor_lookup[["flavor_id", "flavor_name", "flavor_group", "flavor_label"]]
    comp_flavor = comp_flavor.merge(flavor_lookup, on="flavor_id", how="left")

    return {
        "food": food,
        "compound": compound,
        "compound_content": compound_content,
        "compound_flavor": comp_flavor,
    }


def load_recipe_assets() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RECIPES_PATH.exists():
        raise FileNotFoundError(f"Missing recipes at {RECIPES_PATH}")
    if not INTERACTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing interactions at {INTERACTIONS_PATH}")

    list_converters = {col: safe_literal_eval for col in ["tags", "nutrition", "ingredients"]}
    recipes = pd.read_csv(
        RECIPES_PATH,
        usecols=RECIPE_COLS,
        converters=list_converters,
        parse_dates=["submitted"],
    )
    interactions = pd.read_csv(INTERACTIONS_PATH, parse_dates=["date"])
    return recipes, interactions


def clean_recipes(recipes: pd.DataFrame) -> pd.DataFrame:
    df = recipes.copy()
    df["submitted"] = pd.to_datetime(df["submitted"], errors="coerce")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    df["n_ingredients"] = pd.to_numeric(df["n_ingredients"], errors="coerce")
    df["n_steps"] = pd.to_numeric(df["n_steps"], errors="coerce")
    df["ingredient_list"] = df["ingredients"].apply(
        lambda lst: [normalize_ingredient(token) for token in lst if normalize_ingredient(token)]
    )
    df["ingredient_count"] = df["ingredient_list"].apply(len)
    df["cuisine_tags"] = df["tags"].apply(extract_cuisines)
    df["primary_cuisine"] = df["cuisine_tags"].apply(lambda tags: tags[0] if tags else "unspecified")
    df = df.rename(columns={"id": "recipe_id"})
    keep_cols = [
        "recipe_id",
        "name",
        "submitted",
        "minutes",
        "n_steps",
        "n_ingredients",
        "ingredient_count",
        "ingredient_list",
        "cuisine_tags",
        "primary_cuisine",
    ]
    return df[keep_cols]


def extract_cuisines(tags: Iterable[str]) -> List[str]:
    cuisines = []
    for tag in tags:
        lowered = tag.lower()
        if lowered in CUISINE_TOKENS or lowered.endswith("-cuisine"):
            cuisines.append(lowered)
    if not cuisines:
        cuisines.append("unspecified")
    return sorted(set(cuisines))


def clean_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    df = interactions.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").clip(lower=0, upper=5)
    keep_cols = ["user_id", "recipe_id", "date", "rating"]
    return df[keep_cols]


def build_compound_food_links(compound_content: pd.DataFrame) -> pd.DataFrame:
    return compound_content[["food_id", "source_id"]].drop_duplicates()


def summarize_foodb(food: pd.DataFrame, compound: pd.DataFrame, compound_content: pd.DataFrame) -> Dict:
    food_counts = compound_content.groupby("food_id")["source_id"].nunique()
    compound_counts = compound_content.groupby("source_id")["food_id"].nunique()

    return {
        "foods": int(food.shape[0]),
        "compounds": int(compound.shape[0]),
        "compound_food_edges": int(len(compound_content)),
        "avg_compounds_per_food": float(food_counts.mean()),
        "median_compounds_per_food": float(food_counts.median()),
        "avg_foods_per_compound": float(compound_counts.mean()),
        "median_foods_per_compound": float(compound_counts.median()),
        "food_group_distribution": food["food_group"].fillna("Unknown").value_counts().to_dict(),
        "compound_class_distribution": compound["klass"].fillna("Unknown").value_counts().to_dict(),
    }


def summarize_recipes(recipes: pd.DataFrame, ingredient_counts: Counter) -> Dict:
    submitted = recipes["submitted"].dropna()
    cuisine_counts = recipes["primary_cuisine"].value_counts().to_dict()
    return {
        "recipes": int(len(recipes)),
        "date_range": [
            str(submitted.min().date()) if not submitted.empty else None,
            str(submitted.max().date()) if not submitted.empty else None,
        ],
        "median_minutes": float(recipes["minutes"].median()),
        "median_ingredients": float(recipes["ingredient_count"].median()),
        "cuisine_distribution": cuisine_counts,
        "ingredient_frequency": dict(ingredient_counts),
    }


def summarize_interactions(interactions: pd.DataFrame) -> Dict:
    return {
        "interactions": int(len(interactions)),
        "unique_users": int(interactions["user_id"].nunique()),
        "unique_recipes": int(interactions["recipe_id"].nunique()),
        "rating_distribution": interactions["rating"].fillna(-1).value_counts().sort_index().to_dict(),
    }


def write_clean_outputs(
    recipes: pd.DataFrame,
    interactions: pd.DataFrame,
    food: pd.DataFrame,
    compound_content: pd.DataFrame,
    compound_flavor: pd.DataFrame,
    ingredient_counts: Counter,
) -> None:
    ensure_dir(PROCESSED_DIR)

    recipes.to_csv(PROCESSED_DIR / "recipes_ingredients_clean.csv", index=False)
    interactions.to_csv(PROCESSED_DIR / "recipe_interactions_clean.csv", index=False)

    food_profile = (
        compound_content.groupby("food_id")["source_id"]
        .nunique()
        .reset_index(name="compound_count")
        .merge(food, left_on="food_id", right_on="id", how="left")
        .rename(
            columns={
                "id": "food_id",
                "name": "food_name",
                "food_group": "group",
                "food_subgroup": "subgroup",
            }
        )
    )
    food_profile = food_profile[
        ["food_id", "food_name", "group", "subgroup", "category", "compound_count"]
    ].sort_values("compound_count", ascending=False)
    food_profile.to_csv(PROCESSED_DIR / "food_compound_summary.csv", index=False)

    flavor_links = compound_flavor[["compound_id", "flavor_label", "flavor_group"]].dropna()
    flavor_links = flavor_links.rename(columns={"flavor_label": "canonical_flavor"})
    flavor_links.to_csv(PROCESSED_DIR / "compound_flavor_lookup.csv", index=False)

    ingredient_df = (
        pd.DataFrame(ingredient_counts.items(), columns=["ingredient", "count"])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    ingredient_df.to_csv(PROCESSED_DIR / "ingredient_frequency.csv", index=False)


def plot_food_group_distribution(food: pd.DataFrame, fig_dir: Path) -> None:
    counts = food["food_group"].fillna("Unknown").value_counts()
    plt.figure(figsize=(10, 6))
    counts.plot(kind="bar", color="#5DA5DA")
    plt.ylabel("Number of foods")
    plt.xlabel("Food group")
    plt.title("FooDB food-group distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "food_group_distribution.png", dpi=300)
    plt.close()


def plot_compound_class_distribution(compound: pd.DataFrame, fig_dir: Path) -> None:
    counts = compound["klass"].fillna("Unknown").value_counts()
    plt.figure(figsize=(12, 6))
    counts.plot(kind="bar", color="#60BD68")
    plt.ylabel("Compound count")
    plt.xlabel("Compound class (klass)")
    plt.title("Compound class coverage (all classes)")
    plt.yscale("log")
    plt.xticks(rotation=60, ha="right", fontsize=4)
    plt.tight_layout()
    plt.savefig(fig_dir / "compound_class_distribution.png", dpi=300)
    plt.close()


def plot_food_compound_distribution(food_counts: pd.Series, fig_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(food_counts, bins=60, color="#4C72B0", alpha=0.85)
    plt.xlabel("Unique compounds per food")
    plt.ylabel("Food count")
    plt.title("Distribution of compound diversity per food")
    plt.tight_layout()
    plt.savefig(fig_dir / "compound_diversity_hist.png", dpi=300)
    plt.close()


def plot_mass_vs_coverage(compound_counts: pd.Series, compound: pd.DataFrame, fig_dir: Path) -> None:
    merged = (
        compound_counts.reset_index()
        .rename(columns={"source_id": "compound_id", "food_id": "food_count"})
        .merge(
            compound[["id", "name", "mono_mass", "klass"]],
            left_on="compound_id",
            right_on="id",
            how="left",
        )
    )
    merged["mono_mass"] = pd.to_numeric(merged["mono_mass"], errors="coerce")
    merged = merged.dropna(subset=["mono_mass"])

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        merged["mono_mass"],
        merged["food_count"],
        c=merged["food_count"],
        cmap="viridis",
        alpha=0.6,
        s=25,
    )
    plt.xlabel("Monoisotopic mass")
    plt.ylabel("Foods containing compound")
    plt.title("Compound mass vs. prevalence")
    plt.colorbar(scatter, label="Food coverage")
    plt.tight_layout()
    plt.savefig(fig_dir / "mass_vs_coverage.png", dpi=300)
    plt.close()


def plot_flavor_group_heatmap(
    compound_food: pd.DataFrame,
    compound_flavor: pd.DataFrame,
    food: pd.DataFrame,
    fig_dir: Path,
) -> None:
    flavor_links = compound_food.merge(
        compound_flavor[["compound_id", "flavor_group"]],
        left_on="source_id",
        right_on="compound_id",
        how="left",
    )
    food_lookup = food[["id", "food_group"]].rename(columns={"id": "food_id"})
    flavor_links = flavor_links.merge(food_lookup, on="food_id", how="left")
    flavor_links["food_group"] = flavor_links["food_group"].fillna("Unknown")
    flavor_links["flavor_group"] = flavor_links["flavor_group"].fillna("Unlabeled")

    pivot = (
        flavor_links.groupby(["food_group", "flavor_group"])["food_id"]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )

    plt.figure(figsize=(max(10, pivot.shape[1] * 0.5), max(6, pivot.shape[0] * 0.4)))
    im = plt.imshow(pivot, cmap="YlGnBu")
    plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(pivot.shape[0]), pivot.index)
    cbar = plt.colorbar(im)
    cbar.set_label("Unique foods")
    plt.title("Food groups vs. flavor families (all categories)")
    plt.tight_layout()
    plt.savefig(fig_dir / "flavor_heatmap.png", dpi=300)
    plt.close()


def plot_flavor_distribution(
    compound_food: pd.DataFrame,
    compound_flavor: pd.DataFrame,
    fig_dir: Path,
    top_n: int = 25,
) -> None:
    flavor_links = compound_food.merge(
        compound_flavor[["compound_id", "flavor_label"]],
        left_on="source_id",
        right_on="compound_id",
        how="left",
    )
    flavor_counts = (
        flavor_links.dropna(subset=["flavor_label"])
        .groupby("flavor_label")["food_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    if flavor_counts.empty:
        return

    top_counts = flavor_counts.head(top_n)
    ranks = np.arange(1, len(flavor_counts) + 1)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [1.6, 1]},
    )

    # Left panel: labeled top flavors
    axes[0].barh(
        top_counts.index[::-1],
        top_counts.values[::-1],
        color=plt.cm.PuRd(np.linspace(0.3, 0.9, len(top_counts))),
    )
    axes[0].set_xlabel("Unique foods carrying flavor")
    axes[0].set_title(f"Top {len(top_counts)} flavor descriptors")

    # Right panel: full distribution
    axes[1].plot(
        ranks,
        flavor_counts.values,
        color="#4E79A7",
        linewidth=2,
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Flavor rank (log scale)")
    axes[1].set_ylabel("Unique foods (log scale)")
    axes[1].set_title(f"Flavor coverage distribution (all {len(flavor_counts)} descriptors)")

    plt.tight_layout()
    plt.savefig(fig_dir / "top_flavors.png", dpi=300)
    plt.close()


def plot_recipe_ingredient_hist(recipes: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(recipes["ingredient_count"].dropna(), bins=30, color="#F17CB0", alpha=0.85)
    plt.xlabel("Ingredient count per recipe")
    plt.ylabel("Recipe count")
    plt.title("Recipe ingredient-length distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "recipe_ingredient_hist.png", dpi=300)
    plt.close()


def plot_recipe_minutes_distribution(recipes: pd.DataFrame, fig_dir: Path) -> None:
    clipped = recipes["minutes"].clip(upper=recipes["minutes"].quantile(0.99))
    plt.figure(figsize=(8, 5))
    plt.hist(clipped.dropna(), bins=40, color="#B2912F", alpha=0.85)
    plt.xlabel("Minutes (capped at 99th percentile)")
    plt.ylabel("Recipe count")
    plt.title("Recipe prep-time distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "recipe_minutes_hist.png", dpi=300)
    plt.close()


def plot_recipe_submission_trend(recipes: pd.DataFrame, fig_dir: Path) -> None:
    submitted = recipes["submitted"].dropna()
    if submitted.empty:
        return
    yearly = submitted.dt.to_period("Y").value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    plt.plot(yearly.index.astype(str), yearly.values, color="#4E79A7", marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Recipes added")
    plt.xlabel("Year")
    plt.title("Recipe submissions over time")
    plt.tight_layout()
    plt.savefig(fig_dir / "recipe_submission_trend.png", dpi=300)
    plt.close()


def plot_rating_distribution(interactions: pd.DataFrame, fig_dir: Path) -> None:
    counts = interactions["rating"].fillna(-1).astype(int).value_counts().sort_index()
    labels = [str(idx) for idx in counts.index]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, counts.values, color="#E15759")
    plt.xlabel("Rating (0-5)")
    plt.ylabel("Count")
    plt.title("Full rating distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "rating_distribution.png", dpi=300)
    plt.close()


def plot_cuisine_distribution(recipes: pd.DataFrame, fig_dir: Path) -> None:
    counts = recipes["primary_cuisine"].value_counts()
    plt.figure(figsize=(max(8, len(counts) * 0.4), 6))
    counts.plot(kind="bar", color="#59A14F")
    plt.ylabel("Recipe count")
    plt.xlabel("Cuisine tag (all detected categories)")
    plt.title("Cuisine coverage in Food.com recipes")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "cuisine_distribution.png", dpi=300)
    plt.close()


def plot_ingredient_rank_curve(ingredient_counts: Counter, fig_dir: Path) -> None:
    if not ingredient_counts:
        return
    ordered = ingredient_counts.most_common()
    ranks = np.arange(1, len(ordered) + 1)
    freqs = [count for _, count in ordered]
    plt.figure(figsize=(8, 5))
    plt.plot(ranks, freqs, color="#9C755F")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Ingredient rank (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title("Ingredient frequency curve (all ingredients)")
    plt.tight_layout()
    plt.savefig(fig_dir / "ingredient_frequency_curve.png", dpi=300)
    plt.close()


def main() -> None:
    ensure_dir(PROCESSED_DIR)
    fig_dir = BASE_DIR / "figures"
    ensure_dir(fig_dir)

    # FooDB assets
    tables = load_foodb_tables()
    food = tables["food"]
    compound = tables["compound"]
    compound_content = tables["compound_content"]
    compound_flavor = tables["compound_flavor"]

    # Recipe + interaction assets
    recipes_raw, interactions_raw = load_recipe_assets()
    recipes_clean = clean_recipes(recipes_raw)
    interactions_clean = clean_interactions(interactions_raw)

    ingredient_counts = Counter()
    for items in recipes_clean["ingredient_list"]:
        ingredient_counts.update(items)

    write_clean_outputs(
        recipes_clean,
        interactions_clean,
        food,
        compound_content,
        compound_flavor,
        ingredient_counts,
    )

    compound_food = build_compound_food_links(compound_content)
    food_counts = compound_content.groupby("food_id")["source_id"].nunique()
    compound_counts = compound_content.groupby("source_id")["food_id"].nunique()

    # Visualization suite
    plot_food_group_distribution(food, fig_dir)
    plot_compound_class_distribution(compound, fig_dir)
    plot_food_compound_distribution(food_counts, fig_dir)
    plot_mass_vs_coverage(compound_counts, compound, fig_dir)
    plot_flavor_group_heatmap(compound_food, compound_flavor, food, fig_dir)
    plot_flavor_distribution(compound_food, compound_flavor, fig_dir)
    plot_recipe_ingredient_hist(recipes_clean, fig_dir)
    plot_recipe_minutes_distribution(recipes_clean, fig_dir)
    plot_recipe_submission_trend(recipes_clean, fig_dir)
    plot_rating_distribution(interactions_clean, fig_dir)
    plot_cuisine_distribution(recipes_clean, fig_dir)
    plot_ingredient_rank_curve(ingredient_counts, fig_dir)

    summary = {
        "foodb": summarize_foodb(food, compound, compound_content),
        "recipes": summarize_recipes(recipes_clean, ingredient_counts),
        "interactions": summarize_interactions(interactions_clean),
    }

    with (BASE_DIR / "eda_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("EDA complete. Metrics saved to eda_metrics.json")


if __name__ == "__main__":
    main()
