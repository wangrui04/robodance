import matplotlib.pyplot as plt


def apply_motion(x, y, heading, event):
    action = event["action"]
    speed = event["speed"]
    amplitude = event["amplitude"]

    if action in ["strong_forward_pulse", "strong_pulse"]:
        x += speed * amplitude
    elif action in ["slow_sway", "flowing_arc", "large_flowing_arc"]:
        y += amplitude
    elif action == "sharp_turn":
        heading += 45
    elif action in ["expand_movement", "expand_slowly"]:
        x += amplitude
        y += amplitude
    elif action in ["contract_movement", "contract_slowly"]:
        x -= amplitude
        y -= amplitude

    return x, y, heading


def visualize_motion(motion_parameters, visualization_folder):
    visualization_folder.mkdir(parents=True, exist_ok=True)

    x, y, heading = 0.0, 0.0, 0.0
    xs = [x]
    ys = [y]

    for event in motion_parameters["motion_parameters"]:
        x, y, heading = apply_motion(x, y, heading, event)
        xs.append(x)
        ys.append(y)

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys, marker="o")
    plt.title(
        f"{motion_parameters['song_name']} | "
        f"{motion_parameters['predicted_genre']} | "
        f"{motion_parameters['num_motion_events']} events"
    )
    plt.xlabel("X movement")
    plt.ylabel("Y movement")
    plt.grid(True)
    plt.axis("equal")

    song_stem = motion_parameters["song_name"].replace(".mp3", "")
    output_file = visualization_folder / f"{song_stem}_path.png"

    plt.savefig(output_file)
    plt.close()

    return output_file