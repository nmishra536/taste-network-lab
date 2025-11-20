from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

RECIPES_PATH = PROCESSED_DIR / "recipes_ingredients_clean.csv"
ING_LOOKUP_PATH = PROCESSED_DIR / "ingredient_lookup.csv"
CHEMISTRY_VECTORS_PATH = PROCESSED_DIR / "chemistry_vectors.csv"
OUTPUT_EMBED_PATH = PROCESSED_DIR / "recipe_flavor_embeddings.parquet"

FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

TOP_COMPOUNDS = 600
MAX_RECIPES = 6000
RANDOM_STATE = 42
UMAP_NEIGHBORS = 25
UMAP_COMPONENTS = (2, 3)
KMEANS_CLUSTERS = 10


def load_recipes(limit: Optional[int] = None) -> pd.DataFrame:
    converters = {"ingredient_list": eval, "cuisine_tags": eval}
    df = pd.read_csv(RECIPES_PATH, converters=converters)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=RANDOM_STATE)
    return df.reset_index(drop=True)


def load_lookup() -> Dict[str, int]:
    df = pd.read_csv(ING_LOOKUP_PATH)
    df = df.dropna(subset=["food_id"])
    return dict(zip(df["ingredient"], df["food_id"].astype(int)))


def load_chemistry_vectors() -> pd.DataFrame:
    return pd.read_csv(CHEMISTRY_VECTORS_PATH, usecols=["food_id", "compound_id", "tfidf"])


def select_compounds(chem_df: pd.DataFrame, top_n: int) -> List[int]:
    totals = chem_df.groupby("compound_id")["tfidf"].sum().nlargest(top_n)
    return totals.index.astype(int).tolist()


def build_food_vectors(chem_df: pd.DataFrame, compounds: List[int]) -> Dict[int, np.ndarray]:
    index = {cid: idx for idx, cid in enumerate(compounds)}
    filtered = chem_df[chem_df["compound_id"].isin(index)]
    vectors: Dict[int, np.ndarray] = {}
    for food_id, group in filtered.groupby("food_id"):
        vec = np.zeros(len(compounds), dtype=np.float32)
        for _, row in group.iterrows():
            vec[index[int(row["compound_id"])]] = float(row["tfidf"])
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        vectors[int(food_id)] = vec
    return vectors


def embed_recipe(ingredients: List[str], ing_map: Dict[str, int], food_vecs: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    vecs = []
    for ing in ingredients:
        food_id = ing_map.get(ing)
        if food_id is None:
            continue
        vec = food_vecs.get(food_id)
        if vec is not None:
            vecs.append(vec)
    if not vecs:
        return None
    mat = np.vstack(vecs)
    recipe_vec = mat.mean(axis=0)
    norm = np.linalg.norm(recipe_vec)
    if norm > 0:
        recipe_vec /= norm
    return recipe_vec


def build_embeddings(recipes: pd.DataFrame, ing_map: Dict[str, int], food_vecs: Dict[int, np.ndarray]) -> Tuple[pd.DataFrame, np.ndarray]:
    embeddings = []
    meta_rows = []
    for _, row in recipes.iterrows():
        emb = embed_recipe(row["ingredient_list"], ing_map, food_vecs)
        if emb is None:
            continue
        embeddings.append(emb)
        meta_rows.append(
            {
                "recipe_id": row["recipe_id"],
                "name": row["name"],
                "primary_cuisine": row["primary_cuisine"],
            }
        )
    if not embeddings:
        return pd.DataFrame(), np.array([])
    meta_df = pd.DataFrame(meta_rows)
    emb_array = np.vstack(embeddings)
    meta_df["embedding_index"] = range(len(meta_df))
    return meta_df, emb_array


def compute_umap(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    reducer = UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        n_components=n_components,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    return reducer.fit_transform(embeddings)


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, float]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels) if len(set(labels)) > 1 else -1.0
    return labels, score


def plot_umap(meta: pd.DataFrame, umap_coords: np.ndarray, filename: str, title: str) -> None:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=meta["cluster"],
        cmap="tab20",
        s=8,
        alpha=0.8,
    )
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300)
    plt.close()


def plot_umap_3d(meta: pd.DataFrame, coords_3d: np.ndarray, filename: str, title: str) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(
        coords_3d[:, 0],
        coords_3d[:, 1],
        coords_3d[:, 2],
        c=meta["cluster"],
        cmap="tab20",
        s=8,
        alpha=0.8,
    )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    fig.colorbar(p, ax=ax, label="Cluster")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300)
    plt.close()


def plot_cluster_cuisine_heatmap(meta: pd.DataFrame) -> None:
    group_counts = (
        meta.groupby(["cluster", "primary_cuisine"])
        .size()
        .reset_index(name="count")
    )
    heatmap_df = group_counts.pivot_table(
        index="cluster",
        columns="primary_cuisine",
        values="count",
        fill_value=0,
    )
    heatmap_df.to_csv(PROCESSED_DIR / "cluster_cuisine_counts.csv")

    plt.figure(figsize=(max(10, heatmap_df.shape[1] * 0.4), 6))
    sns.heatmap(
        np.log1p(heatmap_df.values),
        cmap="YlGnBu",
        cbar_kws={"label": "log1p(recipe count)"},
    )
    plt.xticks(
        np.arange(len(heatmap_df.columns)) + 0.5,
        heatmap_df.columns,
        rotation=60,
        ha="right",
        fontsize=8,
    )
    plt.yticks(
        np.arange(len(heatmap_df.index)) + 0.5,
        heatmap_df.index,
        rotation=0,
    )
    plt.title("Cluster vs. Cuisine Coverage")
    plt.xlabel("Cuisine tag")
    plt.ylabel("Cluster id")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "recipe_cluster_cuisine_heatmap.png", dpi=300)
    plt.close()


def main() -> None:
    recipes = load_recipes(MAX_RECIPES)
    ing_map = load_lookup()
    chem_df = load_chemistry_vectors()
    top_compounds = select_compounds(chem_df, TOP_COMPOUNDS)
    food_vecs = build_food_vectors(chem_df, top_compounds)
    meta_df, emb_array = build_embeddings(recipes, ing_map, food_vecs)
    if emb_array.size == 0:
        print("No embeddings generated; improve ingredient mapping.")
        return

    labels, silhouette = cluster_embeddings(emb_array, KMEANS_CLUSTERS)
    meta_df["cluster"] = labels

    results = {"clusters": KMEANS_CLUSTERS, "silhouette": float(silhouette), "embeddings": len(meta_df)}
    print(results)

    coords_2d = compute_umap(emb_array, 2)
    coords_3d = compute_umap(emb_array, 3)

    meta_df["umap2_x"] = coords_2d[:, 0]
    meta_df["umap2_y"] = coords_2d[:, 1]
    meta_df["umap3_x"] = coords_3d[:, 0]
    meta_df["umap3_y"] = coords_3d[:, 1]
    meta_df["umap3_z"] = coords_3d[:, 2]
    meta_df.to_parquet(OUTPUT_EMBED_PATH, index=False)

    plot_umap(meta_df, coords_2d, "recipe_flavor_umap_2d.png", "Recipe Flavor Clusters (UMAP 2D)")
    plot_umap_3d(meta_df, coords_3d, "recipe_flavor_umap_3d.png", "Recipe Flavor Clusters (UMAP 3D)")
    plot_cluster_cuisine_heatmap(meta_df)


if __name__ == "__main__":
    main()
