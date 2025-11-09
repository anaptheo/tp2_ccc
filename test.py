import json
import csv

# ---------- 1. Load CSV metadata ----------
def load_song_metadata(csv_paths):
    """Load one or more Spotify CSVs into a lookup dictionary."""
    lookup = {}

    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    for path in csv_paths:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                track_id = row.get("track_uri", "").strip()
                title = row.get("track_name", "").strip()
                if track_id:
                    lookup[track_id] = title

    return lookup

# ---------- 2. Load rules ----------
with open("rules_broader2.json", "r", encoding="utf-8") as f:
    rules = json.load(f)

# ---------- 3. Load CSV metadata ----------
metadata = load_song_metadata(["2023_spotify_ds1.csv", "2023_spotify_ds2.csv"])

# ---------- 4. Collect unique antecedent track URIs ----------
antecedent_set = set()
for rule in rules:
    for track_id in rule["antecedents"]:
        antecedent_set.add(track_id)

# ---------- 5. Print track URI + name ----------
for track_id in sorted(antecedent_set):
    title = metadata.get(track_id, "Unknown")
    print(f"{track_id} -> {title}")

# ---------- 6. Print count ----------
print(f"\nTotal unique antecedent tracks: {len(antecedent_set)}")
