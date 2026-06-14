import math
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def generate_intermediate_points(x, y, heading, event):
    points = []

    action = event["action"]
    speed = event["speed"]
    amplitude = event["amplitude"]
    turn_angle = math.radians(event["turn_angle"])
    curvature = event["curvature"]
    repeat = event["repeat"]
    motion_shape = event["motion_shape"]

    steps = max(6, int(event["duration"] * 4))

    for r in range(repeat):
        for i in range(steps):
            t = i / steps

            if motion_shape in ["arc", "large_arc", "slow_expand"]:
                heading += turn_angle / steps
                x += speed * amplitude * math.cos(heading) / steps
                y += speed * amplitude * math.sin(heading) / steps

            elif motion_shape in ["spiral_out", "spiral_in"]:
                heading += turn_angle / steps
                radius = amplitude * (t if motion_shape == "spiral_out" else 1 - t)
                x += speed * radius * math.cos(heading) / steps
                y += speed * radius * math.sin(heading) / steps

            elif motion_shape == "s_curve":
                side = math.sin(2 * math.pi * t)
                x += speed * 0.15 * math.cos(heading)
                y += curvature * amplitude * side * 0.08

            elif motion_shape in ["pulse", "strong_pulse"]:
                direction = 1 if t < 0.5 else -0.35
                x += direction * speed * amplitude * math.cos(heading) / steps
                y += direction * speed * amplitude * math.sin(heading) / steps

            elif motion_shape == "sharp_rotation":
                heading += turn_angle / steps

            elif motion_shape in ["pause", "end_pose"]:
                pass

            else:
                x += speed * amplitude * math.cos(heading) / steps
                y += speed * amplitude * math.sin(heading) / steps

            points.append((x, y, heading, action))

    return x, y, heading, points


def animate_motion(motion_parameters, animation_folder):
    animation_folder.mkdir(parents=True, exist_ok=True)

    x, y, heading = 0.0, 0.0, 0.0
    positions = [(x, y, heading, "start")]

    for event in motion_parameters["motion_parameters"]:
        x, y, heading, new_points = generate_intermediate_points(
            x,
            y,
            heading,
            event
        )
        positions.extend(new_points)

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_title(
        f"{motion_parameters['song_name']} | "
        f"{motion_parameters['predicted_genre']}"
    )
    ax.set_xlabel("X movement")
    ax.set_ylabel("Y movement")
    ax.grid(True)
    ax.axis("equal")

    margin = 1.0
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    path_line, = ax.plot([], [], linewidth=1)
    robot_dot, = ax.plot([], [], marker="o", markersize=10)
    heading_line, = ax.plot([], [], linewidth=2)
    action_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes
    )

    def init():
        path_line.set_data([], [])
        robot_dot.set_data([], [])
        heading_line.set_data([], [])
        action_text.set_text("")
        return path_line, robot_dot, heading_line, action_text

    def update(frame):
        current_xs = xs[:frame + 1]
        current_ys = ys[:frame + 1]

        x, y, heading, action = positions[frame]

        path_line.set_data(current_xs, current_ys)
        robot_dot.set_data([x], [y])

        heading_length = 0.5
        hx = x + heading_length * math.cos(heading)
        hy = y + heading_length * math.sin(heading)

        heading_line.set_data([x, hx], [y, hy])
        action_text.set_text(f"Action: {action}")

        return path_line, robot_dot, heading_line, action_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(positions),
        init_func=init,
        interval=120,
        blit=True
    )

    song_stem = motion_parameters["song_name"].replace(".mp3", "")
    output_file = animation_folder / f"{song_stem}_animation.gif"

    animation.save(
        output_file,
        writer=PillowWriter(fps=8)
    )

    plt.close()

    return output_file