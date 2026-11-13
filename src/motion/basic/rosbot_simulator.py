
import math
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import numpy as np


def simulate_rosbot(
    commands_data,
    audio_file,
    output_folder,
    fps=15,
    preview_seconds=15,
    save_video=True
):
    """
    Top-down ROSbot simulator using generated movement commands.

    Simulates:
    - Forward and backward movement
    - Beat-synchronized stops
    - Robot orientation
    - Movement trail
    - Current velocity
    - Original music synchronization

    This is a kinematic visualization, not a physics simulator.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    audio_file = Path(audio_file)
    song_stem = audio_file.stem

    commands = commands_data["commands"]

    if not commands:
        raise ValueError("No movement commands available.")

    if fps <= 0:
        raise ValueError("fps must be positive")

    full_duration = max(c["end_time"] for c in commands)

    duration = (
        min(full_duration, preview_seconds)
        if preview_seconds is not None and preview_seconds > 0
        else full_duration
    )

    # Precompute positions at the start of each command.
    command_positions = []
    x = 0.0
    y = 0.0
    theta = 0.0

    for command in commands:
        command_positions.append((x, y, theta))

        dt = command["duration"]
        v = command["linear_x"]
        w = command["angular_z"]

        if abs(w) < 1e-9:
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
        else:
            new_theta = theta + w * dt
            x += (v / w) * (
                math.sin(new_theta) - math.sin(theta)
            )
            y += -(v / w) * (
                math.cos(new_theta) - math.cos(theta)
            )
            theta = new_theta

    def robot_state(t):
        """
        Return x, y, heading, linear velocity,
        angular velocity and movement label.
        """

        if t < commands[0]["start_time"]:
            return 0.0, 0.0, 0.0, 0.0, 0.0, "WAITING"

        for i, command in enumerate(commands):
            start = command["start_time"]
            end = command["end_time"]

            if start <= t < end:
                dt = t - start

                px, py, heading = command_positions[i]
                v = command["linear_x"]
                w = command["angular_z"]

                if abs(w) < 1e-9:
                    px += v * math.cos(heading) * dt
                    py += v * math.sin(heading) * dt
                else:
                    new_heading = heading + w * dt

                    px += (v / w) * (
                        math.sin(new_heading) - math.sin(heading)
                    )
                    py += -(v / w) * (
                        math.cos(new_heading) - math.cos(heading)
                    )
                    heading = new_heading

                if v > 0:
                    label = "FORWARD"
                elif v < 0:
                    label = "BACKWARD"
                elif abs(w) > 1e-9:
                    label = "TURNING"
                else:
                    label = "STOP"

                return px, py, heading, v, w, label

            if i + 1 < len(commands):
                next_start = commands[i + 1]["start_time"]

                if end <= t < next_start:
                    # Hold the robot at the end of the command.
                    return robot_state(end - 1e-7)

        return x, y, theta, 0.0, 0.0, "FINISHED"

    # Compute simulation trajectory for plot bounds.
    sample_times = np.linspace(
        0,
        duration,
        max(2, int(duration * 20))
    )

    states = [robot_state(t) for t in sample_times]

    xs = np.array([s[0] for s in states])
    ys = np.array([s[1] for s in states])

    # Keep the robot large enough to see.
    robot_length = 0.25
    robot_width = 0.20

    x_padding = max(0.5, np.ptp(xs) * 0.3)
    y_padding = max(0.5, np.ptp(ys) * 0.3)

    x_center = (xs.min() + xs.max()) / 2
    y_center = (ys.min() + ys.max()) / 2

    half_width = max(
        (xs.max() - xs.min()) / 2 + x_padding,
        0.7
    )

    half_height = max(
        (ys.max() - ys.min()) / 2 + y_padding,
        0.7
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    fig.patch.set_facecolor("#101820")
    ax.set_facecolor("#172533")

    ax.set_xlim(
        x_center - half_width,
        x_center + half_width
    )

    ax.set_ylim(
        y_center - half_height,
        y_center + half_height
    )

    ax.set_aspect("equal", adjustable="box")

    ax.grid(
        True,
        color="#334155",
        linestyle="--",
        alpha=0.5
    )

    ax.tick_params(colors="white")

    ax.set_xlabel("X Position (m)", color="white")
    ax.set_ylabel("Y Position (m)", color="white")

    for spine in ax.spines.values():
        spine.set_color("#526579")

    ax.set_title(
        f"ROSbot Simulator | {song_stem}",
        color="white",
        fontsize=17,
        pad=18
    )

    # Robot body (centered around its local origin).
    body = FancyBboxPatch(
        (-robot_length / 2, -robot_width / 2),
        robot_length,
        robot_width,
        boxstyle="round,pad=0.01",
        facecolor="#2DD4BF",
        edgecolor="white",
        linewidth=2
    )

    ax.add_patch(body)

    # Four wheels in robot-local coordinates.
    wheels = []

    wheel_length = 0.075
    wheel_width = 0.035

    for wx in (-0.075, 0.075):
        for wy in (-0.115, 0.115):

            wheel = Rectangle(
                (
                    wx - wheel_length / 2,
                    wy - wheel_width / 2
                ),
                wheel_length,
                wheel_width,
                facecolor="#0F172A",
                edgecolor="#94A3B8",
                linewidth=1
            )

            ax.add_patch(wheel)
            wheels.append(wheel)

    # Forward-facing direction indicator.
    heading_arrow = FancyArrowPatch(
        (0, 0),
        (0.11, 0),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.5,
        color="white"
    )

    ax.add_patch(heading_arrow)

    # Robot movement trail.
    trail_line, = ax.plot(
        [],
        [],
        color="#FBBF24",
        linewidth=2,
        alpha=0.9
    )

    status_text = fig.text(
        0.5,
        0.06,
        "",
        ha="center",
        color="white",
        fontsize=12
    )

    direction_text = fig.text(
        0.5,
        0.90,
        "",
        ha="center",
        color="#2DD4BF",
        fontsize=15,
        fontweight="bold"
    )

    # Transform robot-local points into world coordinates.
    def transform_point(local_x, local_y, px, py, heading):
        return (
            px
            + local_x * math.cos(heading)
            - local_y * math.sin(heading),
            py
            + local_x * math.sin(heading)
            + local_y * math.cos(heading)
        )

    def update(frame):
        t = min(frame / fps, duration)

        px, py, heading, v, w, label = robot_state(t)

        from matplotlib.transforms import Affine2D

        transform = (
            Affine2D()
            .rotate(heading)
            .translate(px, py)
            + ax.transData
        )

        body.set_transform(transform)

        for wheel in wheels:
            wheel.set_transform(transform)

        arrow_start = transform_point(
            0, 0, px, py, heading
        )

        arrow_end = transform_point(
            0.11, 0, px, py, heading
        )

        heading_arrow.set_positions(
            arrow_start,
            arrow_end
        )

        if label == "FORWARD":
            color = "#2DD4BF"
        elif label == "BACKWARD":
            color = "#60A5FA"
        elif label == "STOP":
            color = "#F87171"
        else:
            color = "#CBD5E1"

        body.set_facecolor(color)
        direction_text.set_text(label)
        direction_text.set_color(color)

        # Display trajectory up to the current timestamp.
        visible = sample_times <= t

        trail_line.set_data(
            xs[visible],
            ys[visible]
        )

        status_text.set_text(
            f"Time: {t:.1f}s / {duration:.1f}s    "
            f"Speed: {v:+.2f} m/s    "
            f"Heading: {math.degrees(heading):.1f}°    "
            f"Position: ({px:+.2f}, {py:+.2f}) m"
        )

        return (
            body,
            *wheels,
            heading_arrow,
            trail_line,
            status_text,
            direction_text
        )

    total_frames = math.ceil(duration * fps)

    animation = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000 / fps,
        blit=False
    )

    output_file = (
        output_folder / f"{song_stem}_rosbot_simulation.mp4"
    )

    if not save_video:
        plt.show()
        return None

    with tempfile.TemporaryDirectory() as temp_dir:

        silent_video = Path(temp_dir) / "silent.mp4"

        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=2000,
            extra_args=["-pix_fmt", "yuv420p"]
        )

        try:
            animation.save(
                str(silent_video),
                writer=writer
            )
        finally:
            plt.close(fig)

        # Add original music.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(silent_video),
                "-i", str(audio_file),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                str(output_file)
            ],
            check=True
        )

    return output_file
