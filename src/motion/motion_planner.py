
import json
from pathlib import Path

import numpy as np


BOUNDARY = {
    "x_min": -2.0,
    "x_max": 2.0,
    "y_min": -2.0,
    "y_max": 2.0,
}

STYLE_OPTIONS = {
    "contemporary": [
        "s_wave", "left_curve", "right_curve",
        "backward_pulse", "rotation"
    ],
    "edm": [
        "forward_pulse", "rotation", "s_wave",
        "right_curve", "backward_pulse"
    ],
    "pop": [
        "forward_pulse", "left_curve", "s_wave",
        "backward_pulse", "right_curve"
    ],
    "hiphop": [
        "backward_pulse", "rotation", "forward_pulse",
        "s_wave", "left_curve"
    ],
    "jazz": [
        "s_wave", "right_curve", "rotation",
        "left_curve", "backward_pulse"
    ],
    "unknown": [
        "s_wave", "forward_pulse", "left_curve",
        "backward_pulse", "rotation", "right_curve"
    ],
}


def state_at_time(states, time):
    if not states:
        return {
            "energy": 0.0,
            "brightness": 0.0,
            "energy_change": 0.0,
            "movement_quality": "still",
        }

    return min(
        states,
        key=lambda state: abs(state["time"] - time)
    )



def select_accents(beats, strength_threshold, min_movement_gap):
    if not beats:
        return []

    # Sort beats chronologically.
    beats = sorted(beats, key=lambda b: float(b["time"]))

    # Use a relative threshold based on this song,
    # rather than requiring a fixed absolute strength.
    strengths = np.array([
        float(beat["strength"]) for beat in beats
    ])

    relative_threshold = float(np.percentile(strengths, 60))

    # Keep the user's threshold as an optional upper limit.
    effective_threshold = min(
        strength_threshold,
        relative_threshold
    )

    candidates = []

    for i, beat in enumerate(beats):
        strength = float(beat["strength"])

        left = (
            float(beats[i - 1]["strength"])
            if i > 0 else -1
        )
        right = (
            float(beats[i + 1]["strength"])
            if i + 1 < len(beats) else -1
        )

        if (
            strength >= effective_threshold
            and strength >= left
            and strength >= right
        ):
            candidates.append(beat)

    # Select one strong local accent per time window.
    selected = []

    window_start = 0.0
    last_time = float(beats[-1]["time"])

    while window_start <= last_time:

        window = [
            beat for beat in candidates
            if window_start <= float(beat["time"])
            < window_start + min_movement_gap
        ]

        if window:
            strongest = max(
                window,
                key=lambda beat: float(beat["strength"])
            )

            selected.append(strongest)

        window_start += min_movement_gap

    return selected


def choose_motion(state, genre, event_index, previous_action):
    options = STYLE_OPTIONS.get(
        genre,
        STYLE_OPTIONS["unknown"]
    )

    energy = float(state.get("energy", 0.0))
    energy_change = float(state.get("energy_change", 0.0))
    quality = state.get("movement_quality", "flowing")

    if quality == "still" and energy < 0.015:
        return "rotation"

    if energy_change > 0.08:
        preferred = "forward_pulse"
    elif energy_change < -0.08:
        preferred = "backward_pulse"
    elif quality == "smooth":
        preferred = "s_wave"
    else:
        preferred = options[event_index % len(options)]

    if preferred == previous_action:
        alternatives = [
            action for action in options
            if action != previous_action
        ]
        return alternatives[event_index % len(alternatives)]

    return preferred


def generate_motion_timeline(
    music_data,
    genre_data,
    output_folder,
    strength_threshold=0.65,
    stop_seconds=0.22,
    min_movement_gap=1.5,
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    genre = genre_data["predicted_genre"]
    duration = float(music_data["duration"])
    beats = music_data.get("beats", [])

    selected_beats = select_accents(
        beats,
        strength_threshold,
        min_movement_gap
    )

    events = []
    previous_action = None

    for i, beat in enumerate(selected_beats):
        start = float(beat["time"])

        next_start = (
            float(selected_beats[i + 1]["time"])
            if i + 1 < len(selected_beats)
            else duration
        )

        end = max(
            start,
            min(duration, next_start - stop_seconds)
        )

        if end <= start:
            continue

        state = state_at_time(
            music_data.get("states", []),
            start
        )

        action = choose_motion(
            state,
            genre,
            i,
            previous_action
        )

        events.append({
            "time": round(start, 4),
            "end_time": round(end, 4),
            "duration": round(end - start, 4),
            "beat_strength": beat["strength"],
            "action": action,
            "source": {
                "energy": state["energy"],
                "brightness": state["brightness"],
                "energy_change": state["energy_change"],
                "movement_quality": state["movement_quality"],
                "genre": genre,
            },
        })

        previous_action = action

    data = {
        "song_name": music_data["song_name"],
        "duration": duration,
        "predicted_genre": genre,
        "pre_beat_stop": stop_seconds,
        "boundary": BOUNDARY,
        "num_detected_beats": len(beats),
        "num_strong_beats": len(selected_beats),
        "num_motion_events": len(events),
        "motion_timeline": events,
    }

    output_file = (
        output_folder
        / f"{Path(data['song_name']).stem}_motion_timeline.json"
    )

    output_file.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )

    return data
