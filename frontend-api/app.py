from flask import Flask, request, jsonify
import pickle
import csv

app = Flask(__name__)

# ---------- 1. Load song metadata ----------
def load_song_metadata(csv_path):
    """Load metadata using only track_name as the ID."""
    lookup = {}
    title_to_title = {}

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("track_name", "").strip()
            artist = row.get("artist_name", "").strip()

            if title:
                lookup[title] = {
                    "title": title,
                    "artist": artist,
                }
                title_to_title[title.lower()] = title

    return lookup, title_to_title

# ---------- 2. Load pickled rules ----------
def load_rules_pickle(path):
    """Load preprocessed rules saved with pickle (expects dict with 'rules' DataFrame)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    rules_df = data.get("rules")
    if rules_df is None:
        raise ValueError("Pickle file does not contain 'rules' DataFrame")
    return rules_df

# ---------- 3. Recommendation logic ----------
def recommend_multi(song_ids, rules_df, metadata, top_k=10):
    candidates = []

    # Iterate over DataFrame rows
    for _, row in rules_df.iterrows():
        antecedents = set(row['antecedents'])
        consequents = set(row['consequents'])

        if antecedents.issubset(song_ids):
            for c in consequents:
                candidates.append((c, row['confidence'], row['lift']))

    # Fallback: individual antecedent matches if no subset matches
    if not candidates:
        for song_id in song_ids:
            for _, row in rules_df.iterrows():
                if song_id in row['antecedents']:
                    for c in row['consequents']:
                        candidates.append((c, row['confidence'], row['lift']))

    # Sort by confidence * lift
    candidates.sort(key=lambda x: x[1] * x[2], reverse=True)

    # Deduplicate and format
    seen = set()
    recs = []
    for c, conf, lift in candidates:
        if c not in seen and c not in song_ids:
            seen.add(c)
            song_info = metadata.get(c, {"title": c, "artist": "Unknown", "album": "", "genre": ""})
            recs.append({
                "track_id": c,
                "title": song_info.get("title", c),
                "artist": song_info.get("artist", "Unknown"),
                "album": song_info.get("album", ""),
                "genre": song_info.get("genre", ""),
                "confidence": conf,
                "lift": lift
            })

    return recs[:top_k]

# ---------- 4. Initialize data at startup ----------
RULES_PATH = "/model/rules.pkl"
print("🔄 Loading rules and metadata...")
rules_df = load_rules_pickle(RULES_PATH)
metadata, title_to_id = load_song_metadata("/data/2023_spotify_songs.csv")
print(f"✅ Loaded {len(rules_df)} rules and {len(metadata)} songs.")

# Track current active rules info
current_rules_path = RULES_PATH
current_dataset_version = "v0"

# ---------- 5. API endpoints ----------
@app.route("/api/recommender", methods=["POST"])
def recommend():
    data = request.get_json()
    input_songs = data.get("songs", [])
    print("INPUT SONGS:", input_songs, flush=True)

    song_ids = set()

    for s in input_songs:
        key = s.strip().lower()
        if key in title_to_id:
            song_ids.add(title_to_id[key])   # this returns the proper track_name
        else:
            print(f"⚠️ No match for input '{s}'", flush=True)

    if not song_ids:
        return jsonify({"error": "No valid songs found in request."}), 400

    recs = recommend_multi(song_ids, rules_df, metadata)
    return jsonify({"recommendations": recs})


# ---------- 5.1 Reload rules endpoint ----------
@app.route("/reload_rules", methods=["POST"])
def reload_rules():
    global rules_df, current_rules_path, current_dataset_version

    data = request.get_json()
    rules_path = data.get("rules_path")
    dataset_version = data.get("dataset_version")

    if not rules_path:
        return jsonify({"error": "Missing 'rules_path' parameter"}), 400
    if not dataset_version:
        return jsonify({"error": "Missing 'dataset_version' parameter"}), 400

    try:
        new_rules = load_rules_pickle(rules_path)
        rules_df = new_rules
        current_rules_path = rules_path
        current_dataset_version = dataset_version
        print(f"✅ Reloaded rules from {rules_path} (dataset: {dataset_version}), total {len(rules_df)} rules.")
        return jsonify({
            "message": "Rules successfully reloaded.",
            "rules_path": rules_path,
            "dataset_version": dataset_version,
            "num_rules": len(rules_df)
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to reload rules: {str(e)}"}), 500

# ---------- 5.2 Check rules endpoint ----------
@app.route("/check_rules", methods=["GET"])
def check_rules():
    """Return current rules configuration and statistics"""
    return jsonify({
        "rules_path": current_rules_path,
        "dataset_version": current_dataset_version,
        "num_rules": len(rules_df),
        "status": "active"
    }), 200

@app.route("/get_rules", methods=["GET"])
def get_rules():
    sample_titles = list(title_to_id.keys())[:50]
    sample_rules = []
    for i, (_, row) in enumerate(rules_df.iterrows()):
        if i >= 10:
            break
        sample_rules.append({
            "antecedents": list(row['antecedents']),
            "consequents": list(row['consequents']),
            "confidence": row["confidence"],
            "lift": row["lift"],
        })

    return jsonify({
        "num_songs": len(metadata),
        "num_rules": len(rules_df),
        "sample_titles": sample_titles,
        "sample_rules": sample_rules,
    }), 200
# ---------- 6. Run server ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50001)