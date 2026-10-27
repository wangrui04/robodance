
import math
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyBboxPatch, Circle


def generate_motion_video(
    commands_data,
    audio_file,
    output_folder,
    fps=24,
    preview_seconds=None
):
    """
    Generate an MP4 animation synchronized with the original music.

    Shows:
    - Forward movement in green
    - Backward movement in blue
    - Stops in red
    - Robot position over time
    - Highlighted stationary intervals
    - Elapsed song time

    Requires FFmpeg installed and accessible from PATH.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    audio_file = Path(audio_file)
    song_stem = audio_file.stem

    commands = commands_data["commands"]

    if not commands:
        raise ValueError("No movement commands available.")

    if fps <= 0:
        raise ValueError("fps must be positive.")

    if preview_seconds is not None and preview_seconds <= 0:
        raise ValueError("preview_seconds must be positive or None.")

    full_duration = max(command["end_time"] for command in commands)

    duration = (
        min(full_duration, preview_seconds)
        if preview_seconds is not None
        else full_duration
    )

    # Calculate robot position at the beginning of each command.
    command_starts = []
    x = 0.0

    for command in commands:
        command_starts.append(x)
        x += command["linear_x"] * command["duration"]

    final_position = x

    def position_at_time(t):
        """
        Return the robot's position and movement state
        at a particular timestamp.
        """

        if t < commands[0]["start_time"]:
            return 0.0, "WAITING"

        for i, command in enumerate(commands):

            start = command["start_time"]
            end = command["end_time"]

            if start <= t < end:

                elapsed = t - start

                position = (
                    command_starts[i]
                    + command["linear_x"] * elapsed
                )

                velocity = command["linear_x"]

                if velocity > 0:
                    direction = "FORWARD"
                elif velocity < 0:
                    direction = "BACKWARD"
                else:
                    direction = "STOP"

                return position, direction

            # Handle any gaps between commands.
            if i + 1 < len(commands):
                next_start = commands[i + 1]["start_time"]

                if end <= t < next_start:
                    end_position = (
                        command_starts[i]
                        + command["linear_x"] * command["duration"]
                    )

                    return end_position, "STOP"

        return final_position, "FINISHED"

    # Generate position data for the graph.
    sample_times = np.linspace(
        0,
        duration,
        max(2, int(duration * 20))
    )

    sample_positions = np.array([
        position_at_time(t)[0]
        for t in sample_times
    ])

    min_x = float(np.min(sample_positions))
    max_x = float(np.max(sample_positions))

    padding = max(0.15, (max_x - min_x) * 0.15)

    x_min = min_x - padding
    x_max = max_x + padding

    # Create the visualization.
    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor("#101820")

    ax_robot = fig.add_axes([0.08, 0.52, 0.84, 0.34])
    ax_graph = fig.add_axes([0.08, 0.15, 0.84, 0.25])

    for ax in (ax_robot, ax_graph):
        ax.set_facecolor("#172533")
        ax.tick_params(colors="white")

        for spine in ax.spines.values():
            spine.set_color("#526579")

    # Robot animation panel.
    ax_robot.set_xlim(x_min, x_max)
    ax_robot.set_ylim(-0.5, 0.5)
    ax_robot.set_yticks([])

    ax_robot.set_xlabel(
        "Position (meters)",
        color="white"
    )

    ax_robot.axhline(
        0,
        color="#718096",
        linestyle="--",
        linewidth=1.5
    )

    robot_width = max(
        0.08,
        (x_max - x_min) * 0.06
    )

    robot_height = 0.18

    robot = FancyBboxPatch(
        (-robot_width / 2, -robot_height / 2),
        robot_width,
        robot_height,
        boxstyle="round,pad=0.02",
        facecolor="#2DD4BF",
        edgecolor="white",
        linewidth=1.5
    )

    ax_robot.add_patch(robot)

    left_wheel = Circle(
        (0, -robot_height / 2),
        robot_width * 0.12,
        color="#CBD5E1"
    )

    right_wheel = Circle(
        (0, robot_height / 2),
        robot_width * 0.12,
        color="#CBD5E1"
    )

    ax_robot.add_patch(left_wheel)
    ax_robot.add_patch(right_wheel)

    direction_text = ax_robot.text(
        0.5,
        0.92,
        "",
        transform=ax_robot.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        color="#2DD4BF",
        fontweight="bold"
    )

    position_text = ax_robot.text(
        0.5,
        0.08,
        "",
        transform=ax_robot.transAxes,
        ha="center",
        color="white",
        fontsize=11
    )

    # Position-versus-time graph.
    ax_graph.plot(
        sample_times,
        sample_positions,
        color="#64748B",
        linewidth=1.5,
        label="Robot position"
    )

    # Highlight STOP intervals in red.
    for command in commands:

        if command["linear_x"] == 0:

            stop_start = command["start_time"]
            stop_end = command["end_time"]

            if stop_start >= duration:
                continue

            ax_graph.axvspan(
                stop_start,
                min(stop_end, duration),
                color="#F87171",
                alpha=0.22
            )

    ax_graph.set_xlim(0, duration)

    graph_padding = max(
        0.05,
        (max_x - min_x) * 0.1
    )

    ax_graph.set_ylim(
        min_x - graph_padding,
        max_x + graph_padding
    )

    ax_graph.set_xlabel(
        "Time (seconds)",
        color="white"
    )

    ax_graph.set_ylabel(
        "Position (m)",
        color="white"
    )

    current_point, = ax_graph.plot(
        [],
        [],
        "o",
        color="#2DD4BF",
        markersize=8
    )

    progress_line = ax_graph.axvline(
        0,
        color="#FBBF24",
        linewidth=1.5
    )

    fig.text(
        0.5,
        0.95,
        f"RoboDance | {song_stem}",
        ha="center",
        color="white",
        fontsize=18,
        fontweight="bold"
    )

    time_text = fig.text(
        0.5,
        0.045,
        "",
        ha="center",
        color="#CBD5E1",
        fontsize=12
    )

    def update(frame):

        t = min(frame / fps, duration)

        position, direction = position_at_time(t)

        robot.set_x(position - robot_width / 2)

        left_wheel.center = (
            position,
            -robot_height / 2
        )

        right_wheel.center = (
            position,
            robot_height / 2
        )

        direction_text.set_text(direction)

        if direction == "FORWARD":
            color = "#2DD4BF"
        elif direction == "BACKWARD":
            color = "#60A5FA"
        elif direction == "STOP":
            color = "#F87171"
        else:
            color = "#CBD5E1"

        direction_text.set_color(color)
        robot.set_facecolor(color)

        position_text.set_text(
            f"Position: {position:+.3f} m"
        )

        current_point.set_data(
            [t],
            [position]
        )

        progress_line.set_xdata([t, t])

        time_text.set_text(
            f"{t:.1f} / {duration:.1f} seconds"
        )

        return (
            robot,
            left_wheel,
            right_wheel,
            direction_text,
            position_text,
            current_point,
            progress_line,
            time_text
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
        output_folder
        / f"{song_stem}_basic_animation.mp4"
    )

    # Render silent animation, then add original audio.
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
