import csv

def load_csv_tracks(csv_paths):
    """Load one or more CSVs and return a dictionary of track_uri -> track_name."""
    tracks = {}
    
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    
    for path in csv_paths:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                track_id = row.get("track_uri", "").strip()
                track_name = row.get("track_name", "").strip()
                if track_id and track_id not in tracks:
                    tracks[track_id] = track_name
    return tracks

# Paths to your CSVs
csv_files = ["2023_spotify_ds1.csv", "2023_spotify_ds2.csv"]

# Load all tracks
all_tracks = load_csv_tracks(csv_files)

# Print track URI -> name
for track_id, track_name in sorted(all_tracks.items(), key=lambda x: x[1]):  # sorted by name
    print(f"{track_id} -> {track_name}")

# Count unique tracks
print(f"\nTotal unique Spotify tracks: {len(all_tracks)}")
