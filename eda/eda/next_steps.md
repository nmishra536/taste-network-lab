# Next Steps - Ingredient Pairing Recommender

## 1. Ingredient identity resolution
1. Build a lookup table that maps each normalized Food.com token (from `recipes_ingredients_clean.csv`) to a FooDB `public_id`, `food_group`, and flavor family. Start with exact matches, then add synonym rules (scallion = green onion, chickpea = garbanzo bean, etc.).
2. Flag tokens that do not exist in FooDB and decide whether to drop them, map them to broader parents, or extend FooDB with stubs so the graph stays connected.
3. Save the mapping decisions as `data/processed/ingredient_lookup.csv` and reference it inside `eda.py` so future runs stay deterministic.

## 2. Feature engineering
1. **Chemistry vectors**: pivot `food_compound_summary.csv` plus FooDB concentration values into TF-IDF-weighted embeddings per ingredient. Persist to `data/processed/chemistry_vectors.parquet`.
2. **Co-occurrence and novelty**: from `recipes_ingredients_clean.csv`, compute pair counts, PMI, and novelty scores (inverse document frequency). Save to `data/processed/recipe_pair_stats.parquet`.
3. **Acceptance priors**: aggregate `recipe_interactions_clean.csv` to per-recipe average rating and per-pair rating lift while correcting for the 71% five-star skew.
4. **Cuisine context**: explode cuisine tags and compute per-cuisine ingredient frequencies plus acceptance rates to support context-aware filtering.

## 3. Graph construction and modeling
1. Merge chemistry similarity, co-occurrence strength, novelty, and acceptance into a single edge list (`data/processed/ingredient_graph.parquet`) that tracks each component weight.
2. Implement baseline recommenders:
   - Chemistry-only nearest neighbors with novelty filtering.
   - Hybrid scoring: `score = w1*chemistry + w2*cooccurrence + w3*acceptance + w4*novelty`.
3. Train a Node2Vec (or similar) embedding on the weighted graph and compare retrieval quality vs. the hand-crafted scores.

## 4. Evaluation and reporting
1. Create validation splits from Food.com recipes (train on 80%, test on 20%) and log hit@k/MAP plus novelty metrics on held-out ingredient pairs.
2. Visualize the ingredient network (PyVis or Plotly) with filters for food group, flavor group, and cuisine so stakeholders can inspect all categories interactively.
3. Promote this `eda/` folder into a public GitHub repo once the final milestone begins; keep cleaned datasets and write-ups under an `eda/` subdirectory for portfolio reuse.
4. Continue logging any deleted placeholders and newly created artifacts inside `eda_summary.md` and `narrative_summary.md` for an auditable history.
