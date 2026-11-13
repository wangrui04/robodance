
import math
import subprocess
import tempfile
from bisect import bisect_right
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.transforms import Affine2D


def integrate(x, y, heading, v, w, dt):
    if abs(w) < 1e-9:
        return (
            x + v * math.cos(heading) * dt,
            y + v * math.sin(heading) * dt,
            heading
        )

    new_heading = heading + w * dt

    return (
        x + v / w * (
            math.sin(new_heading) - math.sin(heading)
        ),
        y - v / w * (
            math.cos(new_heading) - math.cos(heading)
        ),
        new_heading
    )


def animate_motion(
    commands_data,
    audio_file,
    animation_folder,
    preview_seconds=30,
    fps=12
):
    animation_folder = Path(animation_folder)
    animation_folder.mkdir(parents=True, exist_ok=True)

    audio_file = Path(audio_file)

    commands = sorted(
        commands_data["commands"],
        key=lambda command: command["start_time"]
    )

    if not commands:
        raise ValueError("No ROSbot commands to animate.")

    duration = float(commands_data["duration"])

    if preview_seconds > 0:
        duration = min(duration, preview_seconds)

    if fps <= 0:
        raise ValueError("FPS must be positive.")

    boundary = commands_data.get(
        "boundary",
        {
            "x_min": -2,
            "x_max": 2,
            "y_min": -2,
            "y_max": 2
        }
    )

    # ==========================================
    # PRECOMPUTE ROBOT POSES
    # ==========================================

    segments = []
    x, y, heading = 0.0, 0.0, 0.0
    previous_end = 0.0

    for command in commands:

        start = float(command["start_time"])
        end = float(command["end_time"])

        v = float(command["linear_x"])
        w = float(command["angular_z"])

        if start > previous_end:
            segments.append({
                "start": previous_end,
                "end": start,
                "pose": (x, y, heading),
                "v": 0.0,
                "w": 0.0,
                "action": "STOP"
            })

        segments.append({
            "start": start,
            "end": end,
            "pose": (x, y, heading),
            "v": v,
            "w": w,
            "action": command["action"]
        })

        x, y, heading = integrate(
            x, y, heading, v, w, end - start
        )

        previous_end = end

    if previous_end < duration:
        segments.append({
            "start": previous_end,
            "end": duration,
            "pose": (x, y, heading),
            "v": 0.0,
            "w": 0.0,
            "action": "STOP"
        })

    segment_starts = [
        segment["start"] for segment in segments
    ]

    def state_at(t):
        index = bisect_right(segment_starts, t) - 1
        index = max(0, min(index, len(segments) - 1))

        segment = segments[index]

        dt = max(
            0.0,
            min(t, segment["end"]) - segment["start"]
        )

        px, py, ph = integrate(
            *segment["pose"],
            segment["v"],
            segment["w"],
            dt
        )

        return (
            px,
            py,
            ph,
            segment["v"],
            segment["w"],
            segment["action"]
        )

    # ==========================================
    # GENERATE FRAMES
    # ==========================================

    num_frames = max(1, math.ceil(duration * fps))

    frame_times = np.arange(num_frames) / fps

    states = [state_at(float(t)) for t in frame_times]

    xs = np.array([state[0] for state in states])
    ys = np.array([state[1] for state in states])

    # ==========================================
    # CREATE FIGURE
    # ==========================================

    fig, ax = plt.subplots(figsize=(9, 8))

    fig.patch.set_facecolor("#101827")
    ax.set_facecolor("#182438")

    ax.set_aspect("equal")

    ax.set_xlim(
        boundary["x_min"] - 0.5,
        boundary["x_max"] + 0.5
    )

    ax.set_ylim(
        boundary["y_min"] - 0.5,
        boundary["y_max"] + 0.5
    )

    ax.tick_params(colors="white")
    ax.set_xlabel("X position (m)", color="white")
    ax.set_ylabel("Y position (m)", color="white")

    ax.grid(
        True,
        alpha=0.15,
        color="white"
    )

    # Boundary
    boundary_box = Rectangle(
        (
            boundary["x_min"],
            boundary["y_min"]
        ),
        boundary["x_max"] - boundary["x_min"],
        boundary["y_max"] - boundary["y_min"],
        fill=False,
        edgecolor="#F87171",
        linewidth=2,
        linestyle="--"
    )

    ax.add_patch(boundary_box)

    # Robot dimensions (illustrative footprint)
    robot_length = 0.35
    robot_width = 0.25

    robot_body = Rectangle(
        (-robot_length / 2, -robot_width / 2),
        robot_length,
        robot_width,
        facecolor="#2DD4BF",
        edgecolor="white",
        linewidth=1.5,
        zorder=5
    )

    ax.add_patch(robot_body)

    wheels = []

    for wheel_x in (-0.11, 0.11):
        for wheel_y in (-0.16, 0.16):
            wheel = Rectangle(
                (wheel_x - 0.055, wheel_y - 0.035),
                0.11,
                0.07,
                facecolor="#090D15",
                edgecolor="#94A3B8",
                linewidth=1,
                zorder=6
            )
            ax.add_patch(wheel)
            wheels.append(wheel)

    heading_arrow = FancyArrowPatch(
        (0, 0),
        (0.3, 0),
        arrowstyle="-|>",
        mutation_scale=18,
        color="white",
        linewidth=2,
        zorder=7
    )

    ax.add_patch(heading_arrow)

    trail, = ax.plot(
        [],
        [],
        color="#2DD4BF",
        linewidth=2,
        alpha=0.8
    )

    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=11,
        family="monospace",
        bbox={
            "facecolor": "#101827",
            "alpha": 0.9,
            "edgecolor": "none",
            "pad": 10
        }
    )

    title = (
        f"{commands_data['song_name']}  |  "
        f"{commands_data.get('predicted_genre', 'unknown')}"
    )

    ax.set_title(
        title,
        color="white",
        fontsize=14,
        pad=15
    )

    # ==========================================
    # ANIMATE ACTUAL ROSBOT COMMANDS
    # ==========================================

    def update(frame):

        t = float(frame_times[frame])

        px, py, ph, v, w, action = states[frame]

        transform = (
            Affine2D()
            .rotate(ph)
            .translate(px, py)
            + ax.transData
        )

        robot_body.set_transform(transform)

        for wheel in wheels:
            wheel.set_transform(transform)

        heading_arrow.set_positions(
            (px, py),
            (
                px + 0.32 * math.cos(ph),
                py + 0.32 * math.sin(ph)
            )
        )

        trail.set_data(
            xs[:frame + 1],
            ys[:frame + 1]
        )

        if abs(v) < 1e-6 and abs(w) < 1e-6:
            color = "#F87171"
            movement = "STOPPED"
        elif v < 0:
            color = "#60A5FA"
            movement = "BACKWARD"
        elif abs(w) > 1e-6 and abs(v) < 1e-6:
            color = "#FBBF24"
            movement = "ROTATING"
        else:
            color = "#2DD4BF"
            movement = "MOVING"

        robot_body.set_facecolor(color)
        trail.set_color(color)

        status_text.set_text(
            f"Time:     {t:6.2f}s\n"
            f"Primitive: {action}\n"
            f"State:     {movement}\n"
            f"Linear:    {v:+.3f} m/s\n"
            f"Angular:   {w:+.3f} rad/s\n"
            f"Position: ({px:+.2f}, {py:+.2f})\n"
            f"Heading:   {math.degrees(ph):+.1f} deg"
        )

        return (
            robot_body,
            *wheels,
            heading_arrow,
            trail,
            status_text
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps,
        blit=False
    )

    # ==========================================
    # SAVE MP4 WITH ORIGINAL AUDIO
    # ==========================================

    song_stem = Path(
        commands_data["song_name"]
    ).stem

    output_file = (
        animation_folder
        / f"{song_stem}_animation_{int(duration)}s.mp4"
    )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            silent_video = (
                Path(temp_dir) / "silent.mp4"
            )

            animation.save(
                str(silent_video),
                writer=FFMpegWriter(
                    fps=fps,
                    codec="libx264",
                    extra_args=["-pix_fmt", "yuv420p"]
                )
            )

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel", "error",
                    "-i", str(silent_video),
                    "-i", str(audio_file),
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-t", str(duration),
                    "-movflags", "+faststart",
                    str(output_file)
                ],
                check=True
            )

    finally:
        plt.close(fig)

    return output_file
