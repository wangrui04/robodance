import json
import librosa
import numpy as np

from pathlib import Path

music_folder = Path("music")
output_folder = Path("output")

output_folder.mkdir(exist_ok=True)

WINDOW_SIZE = 1.0  # seconds

for audio_file in sorted(music_folder.glob("*.mp3")):
    print(f"\nProcessing {audio_file.name}")

    y, sr = librosa.load(audio_file)
    duration = librosa.get_duration(y=y, sr=sr)

    states = []

    num_windows = int(duration // WINDOW_SIZE)

    previous_energy = None
    previous_brightness = None

    for i in range(num_windows):
        start_time = i * WINDOW_SIZE
        end_time = start_time + WINDOW_SIZE

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        segment = y[start_sample:end_sample]

        if len(segment) == 0:
            continue

        # Energy
        rms = librosa.feature.rms(y=segment)[0]
        energy = float(np.mean(rms))

        # Brightness
        spectral_centroid = librosa.feature.spectral_centroid(
            y=segment,
            sr=sr
        )[0]
        brightness = float(np.mean(spectral_centroid))

        # Change from previous second
        if previous_energy is None:
            energy_change = 0.0
            brightness_change = 0.0
        else:
            energy_change = energy - previous_energy
            brightness_change = brightness - previous_brightness

        # Simple movement quality label
        if energy < 0.03:
            movement_quality = "still"
        elif energy < 0.08:
            movement_quality = "smooth"
        elif energy < 0.15:
            movement_quality = "flowing"
        else:
            movement_quality = "strong"

        state = {
            "time": round(start_time, 2),
            "energy": round(energy, 4),
            "brightness": round(brightness, 2),
            "energy_change": round(energy_change, 4),
            "brightness_change": round(brightness_change, 2),
            "movement_quality": movement_quality
        }

        states.append(state)

        previous_energy = energy
        previous_brightness = brightness

    output_data = {
        "song_name": audio_file.name,
        "duration": round(duration, 2),
        "window_size": WINDOW_SIZE,
        "states": states
    }

    output_file = output_folder / f"{audio_file.stem}_music_state.json"

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Saved: {output_file}")