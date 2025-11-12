#!/usr/bin/env python3
"""
Generate association rules from the Spotify CSV playlists (optimized + instrumented)
and notify the Flask recommender service when done.

Usage:
  python scripts/generate_rules.py

Environment Variables:
  INPUTS, MIN_SUPPORT, MIN_CONFIDENCE, SAMPLE, OUT, OUT_JSON, MAX_ITEMSET_SIZE
  FRONTEND_IP: Base URL for recommender API (e.g., http://localhost:50001)
  DATASET_NAME: Optional label for the dataset version
"""
import argparse
import pickle
import json
import time
import os
import requests
from pathlib import Path
from typing import List

import pandas as pd

try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
except Exception:
    raise ImportError("Install dependencies: pip install mlxtend pandas mlxtend")


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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(obj, f)


def save_rules_json(rules_df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
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


# --- Environment variable parsing ---
def get_config_from_env():
    """Parse configuration from environment variables"""
    config = {}

    inputs_str = os.getenv('INPUTS')
    if not inputs_str:
        raise ValueError("INPUTS environment variable is required")
    config['inputs'] = [x.strip() for x in inputs_str.split(',')]

    config['min_support'] = float(os.getenv('MIN_SUPPORT', '0.01'))
    config['min_confidence'] = float(os.getenv('MIN_CONFIDENCE', '0.5'))

    sample_str = os.getenv('SAMPLE')
    config['sample'] = int(sample_str) if sample_str else None

    config['out'] = os.getenv('OUT', 'model/rules.pkl')
    config['out_json'] = os.getenv('OUT_JSON')
    config['max_itemset_size'] = int(os.getenv('MAX_ITEMSET_SIZE', '3'))

    config['dataset_name'] = os.getenv('DATASET_NAME', 'default')
    return config


# --- Main ---
def main():
    try:
        config = get_config_from_env()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1

    print("Configuration loaded:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    end = timed("Full pipeline")

    input_paths = [Path(x) for x in config['inputs']]
    transactions = load_transactions(input_paths, sample=config['sample'])

    frequent_itemsets, rules = mine_rules(
        transactions,
        min_support=config['min_support'],
        min_confidence=config['min_confidence'],
        method="fpgrowth",
        prefilter=True,
        max_itemset_size=config['max_itemset_size'],
    )

    out_path = Path(config['out'])
    save_rules({"frequent_itemsets": frequent_itemsets, "rules": rules}, out_path)
    print(f"Saved pickle to {out_path}")

    if config['out_json']:
        save_rules_json(rules, Path(config['out_json']))
        print(f"Saved JSON rules to {config['out_json']}")

    end()


if __name__ == "__main__":
    main()


