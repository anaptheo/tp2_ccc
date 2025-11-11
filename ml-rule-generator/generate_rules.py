#!/usr/bin/env python3
"""
Generate association rules from the Spotify CSV playlists (optimized + instrumented).

Usage (example):
  python scripts/generate_rules.py --inputs ../2023_spotify_ds1.csv ../2023_spotify_ds2.csv \
    --min-support 0.05 --min-confidence 0.5 --sample 1000 --out rules.pkl

Improvements:
  - Added timing checkpoints for each stage
  - Uses sparse=True for TransactionEncoder to reduce memory
  - Prints progress and dataset stats
  - Optionally limits itemset size for speed
"""
import argparse
import pickle
import json
import time
from pathlib import Path
from typing import List

import pandas as pd

try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
except Exception:
    raise ImportError("Install dependencies: pip install mlxtend pandas")


# --- Utility timing helper ---
def timed(label: str):
    """Context-like timing print helper"""
    print(f"[{label}] starting...")
    t0 = time.time()
    def done():
        print(f"[{label}] done in {time.time() - t0:.2f}s")
    return done


# --- Data loading ---
def load_transactions(files: List[Path], track_column="track_uri", pid_column="pid",
                      sample: int | None = None, aggregate: str = "track"):
    end = timed("Loading CSVs")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    end()

    if aggregate == "artist":
        if "artist_name" not in df.columns:
            raise ValueError("Requested artist aggregation but no artist_name column found")
        track_column = "artist_name"

    if track_column not in df.columns:
        track_column = "track_name" if "track_name" in df.columns else None
        if track_column is None:
            raise ValueError("No track column found (neither track_uri nor track_name present)")

    if pid_column not in df.columns:
        raise ValueError("No pid column found in input CSVs")

    print(f"Grouping by '{pid_column}' to form transactions...")
    grouped = df.groupby(pid_column)[track_column].apply(lambda s: s.astype(str).tolist())
    transactions = grouped.tolist()

    if sample is not None:
        transactions = transactions[:sample]
        print(f"Sampling first {sample} playlists...")

    print(f"Loaded {len(transactions)} transactions, avg length={sum(map(len, transactions)) / len(transactions):.1f}")
    return transactions


# --- Frequent itemset mining ---
def mine_rules(transactions: List[List[str]],
               min_support=0.01,
               min_confidence=0.5,
               method="fpgrowth",
               prefilter=True,
               max_itemset_size=None):
    from collections import Counter

    n_transactions = len(transactions)
    min_count = max(1, int(min_support * n_transactions))

    print(f"Prefilter={prefilter}, min_support={min_support}, transactions={n_transactions}")
    if prefilter:
        print("Counting item frequencies...")
        flat = Counter()
        for t in transactions:
            flat.update(set(t))
        frequent_items = {item for item, cnt in flat.items() if cnt >= min_count}

        print(f"Keeping {len(frequent_items)} frequent items out of {len(flat)}")
        filtered_tx = [[it for it in t if it in frequent_items] for t in transactions]
        filtered_tx = [t for t in filtered_tx if t]
    else:
        filtered_tx = transactions

    # One-hot encoding (sparse)
    end = timed("TransactionEncoder (sparse)")
    te = TransactionEncoder()
    te_ary = te.fit(filtered_tx).transform(filtered_tx, sparse=True)
    df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    end()

    print(f"Encoded matrix shape: {df.shape}")

    # Mining
    end = timed(f"Running {method}")
    if method == "apriori":
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    else:
        frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    end()

    if frequent_itemsets.empty:
        print("No frequent itemsets found with given min_support.")
        return frequent_itemsets, pd.DataFrame()

    # Optional filter for large itemsets
    if max_itemset_size:
        frequent_itemsets = frequent_itemsets[
            frequent_itemsets["itemsets"].apply(lambda x: len(x) <= max_itemset_size)
        ]

    print(f"Found {len(frequent_itemsets)} frequent itemsets")

    # Generate rules
    end = timed("Generating association rules")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    end()

    print(f"Generated {len(rules)} rules")
    return frequent_itemsets, rules


# --- Save helpers ---
def save_rules(obj, out_path: Path):
    with open(out_path, "wb") as f:
        pickle.dump(obj, f)


def save_rules_json(rules_df: pd.DataFrame, out_path: Path):
    simplified = []
    for _, row in rules_df.iterrows():
        simplified.append({
            "antecedents": list(row["antecedents"]),
            "consequents": list(row["consequents"]),
            "support": float(row.get("support", 0)),
            "confidence": float(row.get("confidence", 0)),
            "lift": float(row.get("lift", 0)),
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(simplified, f, ensure_ascii=False, indent=2)


# --- Main CLI ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="Input CSV file paths")
    p.add_argument("--min-support", type=float, default=0.01)
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--out", type=str, default="model/rules.pkl")
    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--max-itemset-size", type=int, default=None,
                   help="Limit the max size of itemsets (e.g., 3)")
    args = p.parse_args()

    end = timed("Full pipeline")

    input_paths = [Path(x) for x in args.inputs]
    transactions = load_transactions(input_paths, sample=args.sample)

    frequent_itemsets, rules = mine_rules(
        transactions,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        method="fpgrowth",
        prefilter=True,
        max_itemset_size=args.max_itemset_size,
    )

    out_path = Path(args.out)
    save_rules({"frequent_itemsets": frequent_itemsets, "rules": rules}, out_path)
    print(f"Saved pickle to {out_path}")

    if args.out_json:
        save_rules_json(rules, Path(args.out_json))
        print(f"Saved JSON rules to {args.out_json}")

    end()


if __name__ == "__main__":
    main()


