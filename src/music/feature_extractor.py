import json
import librosa


def analyze_song(audio_file, output_folder):
    output_folder.mkdir(exist_ok=True)

    y, sr = librosa.load(audio_file)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)

    tempo_value = (
        float(tempo[0])
        if hasattr(tempo, "__len__")
        else float(tempo)
    )

    rms = librosa.feature.rms(y=y)[0]
    avg_energy = float(rms.mean())

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )[0]
    brightness = float(spectral_centroid.mean())

    song_features = {
        "song_name": audio_file.name,
        "tempo_bpm": round(tempo_value, 2),
        "num_beats": len(beat_times),
        "avg_energy": round(avg_energy, 4),
        "brightness": round(brightness, 2),
        "beat_times": beat_times.tolist()
    }

    output_file = output_folder / f"{audio_file.stem}_features.json"

    with open(output_file, "w") as f:
        json.dump(song_features, f, indent=4)

    return song_features