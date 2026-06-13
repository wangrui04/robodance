import json
from pathlib import Path

input_folder = Path("output")
output_folder = Path("output")

MOTION_COOLDOWN = 6.0


def choose_motion(state):
    energy = state["energy"]
    energy_change = state["energy_change"]
    brightness = state["brightness"]
    movement_quality = state["movement_quality"]

    if energy < 0.005:
        return "end_pose"

    if movement_quality == "still":
        return "pause"

    if energy_change > 0.08:
        return "expand_movement"

    if energy_change < -0.08:
        return "contract_movement"

    if energy > 0.25:
        return "strong_forward_pulse"

    if brightness > 2500 and energy > 0.12:
        return "sharp_turn"

    if movement_quality == "smooth":
        return "slow_sway"

    if movement_quality == "flowing":
        return "flowing_arc"

    if movement_quality == "strong":
        return "strong_pulse"

    return "hold"


for input_file in sorted(input_folder.glob("*_music_state.json")):
    print(f"\nProcessing {input_file.name}")

    with open(input_file, "r") as f:
        music_data = json.load(f)

    motion_timeline = []
    last_motion_time = -MOTION_COOLDOWN
    previous_action = None

    for state in music_data["states"]:
        current_time = state["time"]
        action = choose_motion(state)

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
                "movement_quality": state["movement_quality"]
            }
        }

        motion_timeline.append(motion_event)

        previous_action = action
        last_motion_time = current_time

    output_data = {
        "song_name": music_data["song_name"],
        "duration": music_data["duration"],
        "motion_cooldown": MOTION_COOLDOWN,
        "num_motion_events": len(motion_timeline),
        "motion_timeline": motion_timeline
    }

    output_file = output_folder / input_file.name.replace(
        "_music_state.json",
        "_motion_timeline.json"
    )

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Saved: {output_file}")
    print(f"Generated {len(motion_timeline)} motion events")