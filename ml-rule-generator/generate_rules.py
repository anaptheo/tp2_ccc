#!/usr/bin/env python3
"""
Generate association rules from Spotify playlist CSVs (track_name only)
and notify the Flask API when rules are ready.

Environment Variables:
  DATASET_PATH     comma-separated CSV URLs
  MIN_SUPPORT
  MIN_CONFIDENCE
  SAMPLE
  OUT
  OUT_JSON
  MAX_ITEMSET_SIZE
  FRONTEND_IP
  DATASET_VERSION
"""

import os
import json
import time
import pickle
import tempfile
from pathlib import Path
from typing import List

import pandas as pd
import requests

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules


# ------------------------------------------------------------
# Download CSV
# ------------------------------------------------------------
def download_csv(url: str) -> pd.DataFrame:
    print(f"⬇ Downloading: {url}")

    try:
        resp = requests.get(url, timeout=60, verify=False)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        df = pd.read_csv(tmp_path)
        os.unlink(tmp_path)

        print(f"   ✔ Loaded {len(df)} rows")
        return df

    except Exception as e:
        print(f"❌ Failed to download: {e}")
        raise


# ------------------------------------------------------------
# Load transactions (track_name grouped by PID)
# ------------------------------------------------------------
def load_transactions(urls: List[str], sample=None) -> List[List[str]]:
    dfs = [download_csv(u) for u in urls]
    df = pd.concat(dfs, ignore_index=True)

    if "pid" not in df.columns:
        raise ValueError("CSV missing 'pid' column")

    if "track_name" not in df.columns:
        raise ValueError("CSV missing 'track_name' column")

    df["track_name"] = df["track_name"].astype(str).str.strip()

    grouped = df.groupby("pid")["track_name"].apply(list)
    transactions = grouped.tolist()

    if sample:
        transactions = transactions[:sample]

    print(f"📦 Total playlists (transactions): {len(transactions)}")
    return transactions


# ------------------------------------------------------------
# Mine itemsets & rules
# ------------------------------------------------------------
def mine_rules(
    transactions: List[List[str]],
    min_support: float,
    min_confidence: float,
    max_itemset_size: int,
):
    print("🧾 Encoding transactions...")
    te = TransactionEncoder()
    te_matrix = te.fit(transactions).transform(transactions, sparse=True)
    df = pd.DataFrame.sparse.from_spmatrix(te_matrix, columns=te.columns_)

    print(f"   ✔ Encoded matrix: {df.shape}")

    print("⚡ Mining frequent itemsets with FPGrowth...")
    itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

    if max_itemset_size:
        itemsets = itemsets[itemsets["itemsets"].apply(lambda s: len(s) <= max_itemset_size)]

    print(f"   ✔ Found {len(itemsets)} itemsets")

    print("✨ Generating association rules...")
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    print(f"   ✔ {len(rules)} rules")

    # convert frozensets → lists
    rules["antecedents"] = rules["antecedents"].apply(list)
    rules["consequents"] = rules["consequents"].apply(list)

    return itemsets, rules


# ------------------------------------------------------------
# Save pickle for Flask API
# ------------------------------------------------------------
def save_pickle(itemsets, rules, out_path: Path):
    obj = {"frequent_itemsets": itemsets, "rules": rules}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(obj, f)

    print(f"💾 Saved pickle to {out_path}")


# ------------------------------------------------------------
# Save JSON (optional)
# ------------------------------------------------------------
def save_json(rules, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rules_list = []

    for _, row in rules.iterrows():
        rules_list.append({
            "antecedents": row["antecedents"],
            "consequents": row["consequents"],
            "support": float(row["support"]),
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rules_list, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved JSON to {out_path}")


# ------------------------------------------------------------
# Notify Flask API
# ------------------------------------------------------------
def notify_frontend(ip: str, rules_path: str, dataset_version: str):
    url = f"http://{ip}/reload_rules"
    payload = {
        "rules_path": rules_path,
        "dataset_version": dataset_version,
    }

    print(f"🔔 Notifying Flask API at {url}")
    try:
        r = requests.post(url, json=payload, timeout=10)
        print(f"   ✔ Response: {r.status_code} {r.text}")
    except Exception as e:
        print(f"❌ Notification failed: {e}")


# ------------------------------------------------------------
# Load config
# ------------------------------------------------------------
def get_config():
    cfg = {}

    ds = os.getenv("DATASET_PATH")
    if not ds:
        raise ValueError("Missing env: DATASET_PATH")

    cfg["inputs"] = [x.strip() for x in ds.split(",")]
    cfg["min_support"] = float(os.getenv("MIN_SUPPORT", "0.01"))
    cfg["min_confidence"] = float(os.getenv("MIN_CONFIDENCE", "0.2"))
    cfg["sample"] = int(os.getenv("SAMPLE", "0")) or None
    cfg["out"] = Path(os.getenv("OUT", "/model/rules.pkl"))
    cfg["out_json"] = Path(os.getenv("OUT_JSON", "/model/rules.json"))
    cfg["max_itemset_size"] = int(os.getenv("MAX_ITEMSET_SIZE", "3"))
    cfg["frontend_ip"] = os.getenv("FRONTEND_IP")
    cfg["dataset_version"] = os.getenv("DATASET_VERSION", "unknown")

    return cfg


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    cfg = get_config()

    print("⚙ Config:")
    print(json.dumps({k: str(v) for k, v in cfg.items()}, indent=2))

    tx = load_transactions(cfg["inputs"], cfg["sample"])

    itemsets, rules = mine_rules(
        tx,
        cfg["min_support"],
        cfg["min_confidence"],
        cfg["max_itemset_size"],
    )

    save_pickle(itemsets, rules, cfg["out"])
    save_json(rules, cfg["out_json"])

    if cfg["frontend_ip"]:
        notify_frontend(cfg["frontend_ip"], str(cfg["out"]), cfg["dataset_version"])

    print("🏁 Done.")
