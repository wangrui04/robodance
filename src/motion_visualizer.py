
import math
from pathlib import Path

import matplotlib.pyplot as plt

from rosbot_command_generator import integrate


def visualize_motion(commands_data, visualization_folder):

    visualization_folder = Path(visualization_folder)
    visualization_folder.mkdir(parents=True, exist_ok=True)

    commands = sorted(
        commands_data["commands"],
        key=lambda command: command["start_time"]
    )

    boundary = commands_data.get(
        "boundary",
        {
            "x_min": -2.0,
            "x_max": 2.0,
            "y_min": -2.0,
            "y_max": 2.0
        }
    )

    x, y, heading = 0.0, 0.0, 0.0

    xs = [x]
    ys = [y]

    for command in commands:

        v = float(command["linear_x"])
        w = float(command["angular_z"])
        duration = float(command["duration"])

        steps = max(1, math.ceil(duration / 0.03))
        dt = duration / steps

        for _ in range(steps):
            x, y, heading = integrate(
                x, y, heading, v, w, dt
            )

            xs.append(x)
            ys.append(y)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(
        xs,
        ys,
        color="#2DD4BF",
        linewidth=2,
        label="ROSbot trajectory"
    )

    ax.scatter(
        xs[0],
        ys[0],
        color="green",
        s=80,
        label="Start",
        zorder=5
    )

    ax.scatter(
        xs[-1],
        ys[-1],
        color="orange",
        s=80,
        label="End",
        zorder=5
    )

    # Draw boundary
    boundary_x = [
        boundary["x_min"],
        boundary["x_max"],
        boundary["x_max"],
        boundary["x_min"],
        boundary["x_min"]
    ]

    boundary_y = [
        boundary["y_min"],
        boundary["y_min"],
        boundary["y_max"],
        boundary["y_max"],
        boundary["y_min"]
    ]

    ax.plot(
        boundary_x,
        boundary_y,
        "r--",
        linewidth=2,
        label="Boundary"
    )

    ax.set_title(
        f"{commands_data['song_name']} | "
        f"{commands_data.get('predicted_genre', 'unknown')}"
    )

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.set_xlim(
        boundary["x_min"] - 0.5,
        boundary["x_max"] + 0.5
    )

    ax.set_ylim(
        boundary["y_min"] - 0.5,
        boundary["y_max"] + 0.5
    )

    song_stem = Path(
        commands_data["song_name"]
    ).stem

    output_file = (
        visualization_folder / f"{song_stem}_path.png"
    )

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_file
