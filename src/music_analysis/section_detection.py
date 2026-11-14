import json
import librosa
import numpy as np

from pathlib import Path

music_folder = Path("music")
output_folder = Path("output")

output_folder.mkdir(exist_ok=True)


def merge_short_sections(boundary_times, duration, min_section_length=8.0):
    section_times = [0.0] + list(boundary_times) + [duration]
    merged = [section_times[0]]

    for time in section_times[1:]:
        if time - merged[-1] >= min_section_length:
            merged.append(time)

    if merged[-1] != duration:
        merged[-1] = duration

    return merged


for audio_file in sorted(music_folder.glob("*.mp3")):
    print(f"\nProcessing {audio_file.name}")

    y, sr = librosa.load(audio_file)
    duration = librosa.get_duration(y=y, sr=sr)

    # Extract features that reflect changes in harmony/tone/timbre
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Combine features
    features = np.vstack([
        chroma,
        mfcc,
        centroid
    ])

    # Make similar frames easier to compare
    features = librosa.util.normalize(features, axis=1)

    # Self-similarity matrix
    similarity = librosa.segment.recurrence_matrix(
        features,
        mode="affinity",
        metric="cosine",
        sym=True
    )

    # Convert similarity into a novelty curve
    novelty = librosa.segment.path_enhance(similarity)

    novelty_curve = np.mean(1 - novelty, axis=0)

    # Pick peaks in novelty curve as section boundaries
    peaks = librosa.util.peak_pick(
        novelty_curve,
        pre_max=20,
        post_max=20,
        pre_avg=40,
        post_avg=40,
        delta=0.03,
        wait=40
    )

    boundary_times = librosa.frames_to_time(peaks, sr=sr)

    # Remove boundaries too close to start/end
    boundary_times = [
        float(t)
        for t in boundary_times
        if 5.0 < t < duration - 5.0
    ]

    # Merge sections that are too short
    section_times = merge_short_sections(
        boundary_times,
        duration,
        min_section_length=8.0
    )

    sections = []

    for i in range(len(section_times) - 1):
        start = round(section_times[i], 2)
        end = round(section_times[i + 1], 2)

        sections.append({
            "section_id": i + 1,
            "start": start,
            "end": end,
            "duration": round(end - start, 2)
        })

    section_data = {
        "song_name": audio_file.name,
        "duration": round(duration, 2),
        "num_sections": len(sections),
        "sections": sections
    }

    print(json.dumps(section_data, indent=4))

    output_file = output_folder / f"{audio_file.stem}_sections.json"

    with open(output_file, "w") as f:
        json.dump(section_data, f, indent=4)

    print(f"Saved: {output_file}")