from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "foodb_2020_04_07_csv"
FLAVOR_CSV = DATA_DIR / "Flavor.csv"
LOOKUP_OUT = BASE_DIR / "flavor_canonical_lookup.csv"
SUMMARY_OUT = BASE_DIR / "flavor_canonical_summary.csv"

SYNONYMS: Dict[str, str] = {
    "cheesy": "cheese",
    "cheddar": "cheese",
    "parmesan": "cheese",
    "cheddar-like": "cheese",
    "buttery": "butter",
    "butterscotch": "butter",
    "fatty": "fat",
    "fatness": "fat",
    "creamy": "cream",
    "creaminess": "cream",
    "milky": "milk",
    "meaty": "meat",
    "meatiness": "meat",
    "fishy": "fish",
    "nutty": "nut",
    "nutlike": "nut",
    "earthy": "earth",
    "herbaceous": "herb",
    "herbal": "herb",
    "spicy": "spice",
    "spiciness": "spice",
    "peppery": "pepper",
    "pepperiness": "pepper",
    "smoky": "smoke",
    "smokey": "smoke",
    "smoked": "smoke",
    "garlicky": "garlic",
    "oniony": "onion",
    "sulfury": "sulfur",
    "sulfurous": "sulfur",
    "tropical": "tropics",
    "savory": "umami",
    "umami": "umami",
    "sweetness": "sweet",
    "sweetish": "sweet",
    "bitterish": "bitter",
    "bitterness": "bitter",
    "pungency": "pungent",
    "fruity": "fruit",
    "fruitiness": "fruit",
    "citrusy": "citrus",
    "citrus-like": "citrus",
    "vanillary": "vanilla",
    "buttermilk": "butter milk",
}

SUFFIX_RULES = [
    ("iness", "y"),
    ("ness", ""),
    ("iness", "y"),
    ("ously", "ous"),
    ("ously", "ous"),
    ("edly", ""),
    ("ing", ""),
    ("edly", ""),
    ("ed", ""),
    ("ly", ""),
    ("y", "e"),
    ("ies", "y"),
    ("es", ""),
    ("s", ""),
]


def clean_token(token: str) -> str:
    token = SYNONYMS.get(token, token)
    for suffix, repl in SUFFIX_RULES:
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            token = token[: -len(suffix)] + repl
            break
    token = SYNONYMS.get(token, token)
    return token


def canonicalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    if not name:
        return ""
    name = SYNONYMS.get(name, name)
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    tokens = [clean_token(tok) for tok in name.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        tokens = [name]
    canonical = " ".join(tokens)
    return canonical.strip()


def main() -> None:
    if not FLAVOR_CSV.exists():
        raise FileNotFoundError(f"Flavor table missing at {FLAVOR_CSV}")

    flavors = pd.read_csv(FLAVOR_CSV, usecols=["id", "name", "flavor_group"])
    flavors["canonical_name"] = flavors["name"].apply(canonicalize)
    flavors.to_csv(LOOKUP_OUT, index=False)

    summary = (
        flavors.groupby("canonical_name")
        .agg(
            flavor_count=("id", "count"),
            example=("name", lambda s: ", ".join(sorted(set(s))[:5])),
            top_group=(
                "flavor_group",
                lambda s: s.mode().iat[0] if not s.mode().empty else "",
            ),
        )
        .reset_index()
        .sort_values("flavor_count", ascending=False)
    )
    summary.to_csv(SUMMARY_OUT, index=False)

    print(f"Wrote lookup to {LOOKUP_OUT}")
    print(f"Wrote summary to {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
