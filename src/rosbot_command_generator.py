import json


def action_to_rosbot_command(event):
    action = event["action"]
    speed = event["speed"]
    amplitude = event["amplitude"]
    duration = event["duration"]

    if action in ["slow_sway"]:
        linear_x = 0.0
        angular_z = 0.35

    elif action in ["flowing_arc", "large_flowing_arc"]:
        linear_x = 0.12 * speed * amplitude
        angular_z = 0.25

    elif action in ["strong_pulse", "strong_forward_pulse"]:
        linear_x = 0.25 * speed * amplitude
        angular_z = 0.0

    elif action in ["expand_movement", "expand_slowly"]:
        linear_x = 0.18 * speed * amplitude
        angular_z = 0.35

    elif action in ["contract_movement", "contract_slowly"]:
        linear_x = -0.12 * speed * amplitude
        angular_z = -0.25

    elif action == "sharp_turn":
        linear_x = 0.0
        angular_z = 0.9

    elif action in ["pause", "end_pose", "hold"]:
        linear_x = 0.0
        angular_z = 0.0

    else:
        linear_x = 0.05
        angular_z = 0.0

    return {
        "time": event["time"],
        "action": action,
        "linear_x": round(linear_x, 3),
        "angular_z": round(angular_z, 3),
        "duration": duration
    }


def generate_rosbot_commands(motion_parameters, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)

    commands = []

    for event in motion_parameters["motion_parameters"]:
        command = action_to_rosbot_command(event)
        commands.append(command)

    output_data = {
        "song_name": motion_parameters["song_name"],
        "duration": motion_parameters["duration"],
        "predicted_genre": motion_parameters["predicted_genre"],
        "num_commands": len(commands),
        "commands": commands
    }

    song_stem = motion_parameters["song_name"].replace(".mp3", "")
    output_file = output_folder / f"{song_stem}_rosbot_commands.json"

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    return output_data