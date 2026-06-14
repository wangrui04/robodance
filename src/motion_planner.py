import json

MOTION_COOLDOWN = 6.0


def choose_motion(state, genre=None):
    energy = state["energy"]
    energy_change = state["energy_change"]
    brightness = state["brightness"]
    movement_quality = state["movement_quality"]

    if energy < 0.005:
        return "end_pose"

    if movement_quality == "still":
        return "pause"

    if genre == "contemporary":
        if energy_change > 0.08:
            return "expand_slowly"
        if energy_change < -0.08:
            return "contract_slowly"
        if energy > 0.25:
            return "large_flowing_arc"
        if movement_quality == "smooth" and energy > 0.05:
            return "slow_sway"
        if movement_quality == "smooth":
            return "hold"
        return "flowing_arc"

    if movement_quality == "smooth" and energy > 0.05:
        return "slow_sway"

    if energy_change > 0.08:
        return "expand_movement"

    if energy_change < -0.08:
        return "contract_movement"

    if energy > 0.25:
        return "strong_forward_pulse"

    if brightness > 2500 and energy > 0.12:
        return "sharp_turn"

    if movement_quality == "smooth":
        return "hold"

    if movement_quality == "flowing":
        return "flowing_arc"

    if movement_quality == "strong":
        return "strong_pulse"

    return "hold"


def generate_motion_timeline(music_data, genre_data, output_folder):
    output_folder.mkdir(exist_ok=True)

    genre = genre_data["predicted_genre"]

    motion_timeline = []
    last_motion_time = -MOTION_COOLDOWN
    previous_action = None

    for state in music_data["states"]:
        current_time = state["time"]

        if current_time < 3.0:
            continue

        action = choose_motion(state, genre)

        if action == previous_action:
            continue

        if current_time - last_motion_time < MOTION_COOLDOWN:
            continue

        motion_event = {
            "time": current_time,
            "action": action,
            "source": {
                "energy": state["energy"],
                "brightness": state["brightness"],
                "energy_change": state["energy_change"],
                "movement_quality": state["movement_quality"],
                "genre": genre
            }
        }

        motion_timeline.append(motion_event)

        previous_action = action
        last_motion_time = current_time

    song_stem = music_data["song_name"].replace(".mp3", "")

    output_data = {
        "song_name": music_data["song_name"],
        "duration": music_data["duration"],
        "predicted_genre": genre,
        "motion_cooldown": MOTION_COOLDOWN,
        "num_motion_events": len(motion_timeline),
        "motion_timeline": motion_timeline
    }

    output_file = output_folder / f"{song_stem}_motion_timeline.json"

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    return output_data