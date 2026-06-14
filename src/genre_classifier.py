import json


def classify_genre(features):
    tempo = features["tempo_bpm"]
    energy = features["avg_energy"]
    brightness = features["brightness"]

    if tempo < 120 and brightness < 1800:
        predicted_genre = "contemporary"
    elif tempo > 125 and energy > 0.22:
        predicted_genre = "edm"
    elif 100 <= tempo <= 140 and energy > 0.15:
        predicted_genre = "pop"
    elif (
        (70 <= tempo <= 115 or 140 <= tempo <= 195)
        and energy > 0.10
        and brightness > 1600
    ):
        predicted_genre = "hiphop"
    elif tempo < 130 and energy < 0.16 and brightness > 1600:
        predicted_genre = "jazz"
    else:
        predicted_genre = "unknown"

    return {
        "song_name": features["song_name"],
        "predicted_genre": predicted_genre,
        "features_used": {
            "tempo_bpm": features["tempo_bpm"],
            "avg_energy": features["avg_energy"],
            "brightness": features["brightness"],
            "num_beats": features["num_beats"]
        }
    }


def save_genre_result(genre_data, output_folder):
    output_folder.mkdir(exist_ok=True)

    song_stem = genre_data["song_name"].replace(".mp3", "")
    output_file = output_folder / f"{song_stem}_genre.json"

    with open(output_file, "w") as f:
        json.dump(genre_data, f, indent=4)

    return genre_data