from flask import Flask, request, jsonify
import pickle
import csv
import pandas as pd

app = Flask(__name__)

# -------------------------------------------------------
# 1. Load song metadata + flexible name lookup
# -------------------------------------------------------

def load_song_metadata(csv_path):
    df = pd.read_csv(csv_path)
    df["track_name"] = df["track_name"].astype(str).str.strip()
    df["artist_name"] = df["artist_name"].astype(str).str.strip()
    df["track_name_lower"] = df["track_name"].str.lower()

    metadata = {
        row["track_name"]: {
            "title": row["track_name"],
            "artist": row["artist_name"]
        }
        for _, row in df.iterrows()
    }

    return metadata, df


# -------------------------------------------------------
# 2. Load rules pickle
# -------------------------------------------------------

def load_rules_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    rules_df = data.get("rules")
    if rules_df is None:
        raise ValueError("Pickle file does not contain 'rules' DataFrame")
    return rules_df


# -------------------------------------------------------
# 3. Song lookup (substring search) + DEBUG
# -------------------------------------------------------

def resolve_song_name(user_text: str):
    query = user_text.lower().strip()

    print("\n========== DEBUG resolve_song_name ==========", flush=True)
    print(f"User text: '{user_text}' -> Query: '{query}'", flush=True)

    matches = lookup_df[
        lookup_df["track_name_lower"].str.contains(query, na=False)
    ]

    if matches.empty:
        print("No substring matches found!", flush=True)
    else:
        print("Matches:", matches["track_name"].unique().tolist()[:20], flush=True)

    print("========== END DEBUG resolve_song_name ==========\n", flush=True)

    return matches["track_name"].unique().tolist()


# -------------------------------------------------------
# 4. Recommendation logic
# -------------------------------------------------------

def recommend_multi(song_ids, rules_df, metadata, top_k=10):
    candidates = []

    # Strong multi-antecedent match
    for _, row in rules_df.iterrows():
        antecedents = set(row['antecedents'])
        consequents = set(row['consequents'])

        if antecedents.issubset(song_ids):
            for c in consequents:
                candidates.append((c, row['confidence'], row['lift']))

    # Fallback
    if not candidates:
        for song_id in song_ids:
            for _, row in rules_df.iterrows():
                if song_id in row['antecedents']:
                    for c in row['consequents']:
                        candidates.append((c, row['confidence'], row['lift']))

    # Sort + dedupe
    candidates.sort(key=lambda x: x[1] * x[2], reverse=True)

    seen = set()
    recs = []
    for c, conf, lift in candidates:
        if c not in seen and c not in song_ids:
            seen.add(c)
            info = metadata.get(c, {"title": c, "artist": "Unknown"})
            recs.append({
                "title": info.get("title", c),
            })

    return recs[:top_k]


# -------------------------------------------------------
# 5. Initialization
# -------------------------------------------------------

RULES_PATH = "/model/rules.pkl"
SONG_CSV_PATH = "/data/2023_spotify_songs.csv"

print("🔄 Loading rules and metadata...", flush=True)
rules_df = load_rules_pickle(RULES_PATH)
metadata, lookup_df = load_song_metadata(SONG_CSV_PATH)
print(f"✅ Loaded {len(rules_df)} rules and {len(metadata)} songs.", flush=True)

current_rules_path = RULES_PATH
current_dataset_version = "v0"


# -------------------------------------------------------
# 6. API: POST /api/recommender + DEBUG
# -------------------------------------------------------

@app.route("/api/recommender", methods=["POST"])
def recommend():
    data = request.get_json()
    input_songs = data.get("songs", [])

    print("\n=========== DEBUG /api/recommender ===========", flush=True)
    print("Raw input songs:", input_songs, flush=True)

    song_ids = set()

    for text in input_songs:
        matches = resolve_song_name(text)
        print(f"Resolved '{text}' -> {matches}", flush=True)

        if not matches:
            print(f"⚠️ No match for '{text}'", flush=True)
            continue

        for m in matches:
            song_ids.add(m)

    print("Final resolved song IDs:", list(song_ids), flush=True)
    print("=========== END DEBUG /api/recommender ===========\n", flush=True)

    if not song_ids:
        return jsonify({"error": "No valid songs found in request."}), 400

    recs = recommend_multi(song_ids, rules_df, metadata)
    return jsonify({"recommendations": recs})


# -------------------------------------------------------
# 7. GET /api/rules?song
# -------------------------------------------------------

@app.get("/api/rules")
def api_rules_for_song():
    song = request.args.get("song")
    if not song:
        return jsonify({"error": "missing song parameter"}), 400

    matches = resolve_song_name(song)
    if not matches:
        return jsonify({"error": "song not found"}), 404

    out = {}
    for m in matches:
        subset = rules_df[rules_df["antecedents"].apply(lambda a: m in a)]
        out[m] = [
            {
                "antecedents": list(row["antecedents"]),
                "consequents": list(row["consequents"]),
                "confidence": row["confidence"],
                "lift": row["lift"],
            }
            for _, row in subset.iterrows()
        ]

    return jsonify(out)


# -------------------------------------------------------
# 8. Reload rules endpoint
# -------------------------------------------------------

@app.route("/reload_rules", methods=["POST"])
def reload_rules():
    global rules_df, current_rules_path, current_dataset_version

    data = request.get_json()
    rules_path = data.get("rules_path")
    dataset_version = data.get("dataset_version")

    if not rules_path:
        return jsonify({"error": "Missing 'rules_path' parameter"}), 400

    try:
        rules_df = load_rules_pickle(rules_path)
        current_rules_path = rules_path
        current_dataset_version = dataset_version or "unknown"

        print(f"✅ Reloaded {len(rules_df)} rules from {rules_path}")
        return jsonify({
            "message": "Rules successfully reloaded.",
            "rules_path": rules_path,
            "dataset_version": current_dataset_version,
            "num_rules": len(rules_df)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------
# 9. Diagnostics
# -------------------------------------------------------

@app.route("/check_rules", methods=["GET"])
def check_rules():
    return jsonify({
        "rules_path": current_rules_path,
        "dataset_version": current_dataset_version,
        "num_rules": len(rules_df)
    })


@app.route("/get_rules", methods=["GET"])
def get_rules():
    sample_titles = lookup_df["track_name"].tolist()[:50]
    sample_rules = []
    for i, (_, row) in enumerate(rules_df.iterrows()):
        if i >= 10:
            break
        sample_rules.append({
            "antecedents": list(row["antecedents"]),
            "consequents": list(row["consequents"]),
            "confidence": row["confidence"],
            "lift": row["lift"],
        })

    return jsonify({
        "num_songs": len(metadata),
        "num_rules": len(rules_df),
        "sample_titles": sample_titles,
        "sample_rules": sample_rules,
    })


# -------------------------------------------------------
# 9b. NEW DEBUG ENDPOINT: /debug/lookup
# -------------------------------------------------------

@app.get("/debug/lookup")
def debug_lookup():
    q = request.args.get("q", "")
    matches = resolve_song_name(q)
    return {"query": q, "matches": matches}


# -------------------------------------------------------
# 9c. FIXED DEBUG SONG LIST
# -------------------------------------------------------

@app.get("/debug/songs")
def list_songs():
    return {
        "songs": sorted(lookup_df["track_name"].unique().tolist())[:200]
    }


# -------------------------------------------------------
# 10. Start server
# -------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50001)
