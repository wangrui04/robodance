import json
import librosa
import numpy as np

WINDOW_SIZE = 1.0


def generate_music_state(audio_file, output_folder):
    output_folder.mkdir(exist_ok=True)

    y, sr = librosa.load(audio_file, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
    _, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        onset_envelope=onset_envelope,
        trim=False
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_strengths = onset_envelope[beat_frames] if len(beat_frames) else []
    max_strength = float(np.max(beat_strengths)) if len(beat_strengths) else 0.0
    threshold = (
        float(np.percentile(beat_strengths, 60))
        if len(beat_strengths)
        else 0.0
    )

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

        rms = librosa.feature.rms(y=segment)[0]
        energy = float(np.mean(rms))

        spectral_centroid = librosa.feature.spectral_centroid(
            y=segment,
            sr=sr
        )[0]
        brightness = float(np.mean(spectral_centroid))

        if previous_energy is None:
            energy_change = 0.0
            brightness_change = 0.0
        else:
            energy_change = energy - previous_energy
            brightness_change = brightness - previous_brightness

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
        "beat_strength_threshold": round(threshold, 4),
        "beats": [
            {
                "time": round(float(time), 3),
                "strength": round(
                    float(strength / max_strength) if max_strength else 0.0,
                    4
                ),
                "is_strong": bool(strength >= threshold),
            }
            for time, strength in zip(beat_times, beat_strengths)
        ],
        "states": states
    }

    output_file = output_folder / f"{audio_file.stem}_music_state.json"

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    return output_data