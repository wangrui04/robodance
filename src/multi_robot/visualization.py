
"""Plot and animate multiple robots; analyze per-robot occupancy."""

from __future__ import annotations

import json
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

from motion.region_analysis import (
    REGION_CELLS,
    cell_for_position,
    region_for_position,
    safe_bounds,
)

from motion.rosbot_command_generator import integrate


# ==========================================
# ROBOT IDENTITY COLOURS
# ==========================================

# Used for trajectory PNGs and robot name labels.
# MP4 robot bodies use movement-state colours instead.

PALETTE = [
    "#2DD4BF",
    "#F59EAD",
    "#FBBF24",
    "#A78BFA",
    "#60A5FA",
    "#FB923C",
    "#84CC16",
    "#F472B6",
    "#22D3EE",
    "#E879F9",
]


def robot_colors(names):
    return {
        name: PALETTE[i % len(PALETTE)]
        for i, name in enumerate(names)
    }


# ==========================================
# RECONSTRUCT ROBOT TRAJECTORIES
# ==========================================

def segments_for_robot(robot, duration):

    pose = tuple(float(v) for v in robot["start_pose"])

    records = []
    previous_end = 0.0

    commands = sorted(
        robot["commands"],
        key=lambda c: c["start_time"]
    )

    for cmd in commands:

        start = float(cmd["start_time"])
        end = float(cmd["end_time"])

        if start > previous_end:
            records.append((
                previous_end,
                start,
                pose,
                0.0,
                0.0,
                "STOP"
            ))

        if end <= start:
            continue

        v = float(cmd["linear_x"])
        w = float(cmd["angular_z"])

        records.append((
            start,
            end,
            pose,
            v,
            w,
            cmd["action"]
        ))

        pose = integrate(
            *pose,
            v,
            w,
            end - start
        )

        previous_end = end

    if previous_end < duration:
        records.append((
            previous_end,
            duration,
            pose,
            0.0,
            0.0,
            "STOP"
        ))

    if not records:
        records.append((
            0.0,
            max(duration, 0.001),
            pose,
            0.0,
            0.0,
            "STOP"
        ))

    return records


def state_at_time(records, starts, time):

    index = bisect_right(starts, time) - 1

    index = max(
        0,
        min(index, len(records) - 1)
    )

    start, end, pose, v, w, action = records[index]

    dt = max(
        0.0,
        min(time, end) - start
    )

    x, y, heading = integrate(
        *pose,
        v,
        w,
        dt
    )

    return x, y, heading, v, w, action


def pose_at(records, starts, time):

    x, y, heading, v, w, action = state_at_time(
        records, starts, time
    )

    return (x, y, heading), action


def sample_robot(robot, duration, sample_dt=0.025):

    records = segments_for_robot(robot, duration)
    starts = [record[0] for record in records]

    count = max(
        1,
        math.ceil(duration / sample_dt)
    )

    times = [
        min(i * sample_dt, duration)
        for i in range(count + 1)
    ]

    if times[-1] != duration:
        times.append(duration)

    xs = []
    ys = []
    headings = []

    for t in times:

        pose, _ = pose_at(records, starts, t)

        xs.append(pose[0])
        ys.append(pose[1])
        headings.append(pose[2])

    return times, xs, ys, headings


def make_samples(data, dt=0.025):

    return {
        name: sample_robot(
            robot,
            float(data["duration"]),
            dt
        )
        for name, robot in data["robots"].items()
    }


# ==========================================
# STAGE VISUALIZATION HELPERS
# ==========================================

def draw_stage(ax, data):

    boundary = data["boundary"]
    margin = data["safety_margin"]

    xmin, xmax, ymin, ymax = safe_bounds(
        boundary, margin
    )

    ax.set_aspect("equal")

    ax.set_xlim(
        boundary["x_min"] - 0.25,
        boundary["x_max"] + 0.25
    )

    ax.set_ylim(
        boundary["y_min"] - 0.25,
        boundary["y_max"] + 0.25
    )

    ax.add_patch(
        Rectangle(
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
    )

    for i in range(5):

        ax.axvline(
            xmin + i * (xmax - xmin) / 4,
            color="gray",
            linestyle=":",
            alpha=0.4
        )

        ax.axhline(
            ymin + i * (ymax - ymin) / 4,
            color="gray",
            linestyle=":",
            alpha=0.4
        )

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.grid(alpha=0.1)


# ==========================================
# STATIC TRAJECTORY PNG
# ==========================================

def plot_multi_robot(data, folder):

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    robots = data["robots"]
    num_robots = len(robots)

    boundary = data["boundary"]
    samples = make_samples(data)

    colors = robot_colors(list(robots))

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot trajectories for all robots
    for name, (_, xs, ys, _) in samples.items():

        color = colors[name]

        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2,
            label=f"{name} trajectory"
        )

        # Start marker
        ax.scatter(
            xs[0],
            ys[0],
            color="green",
            edgecolor=color,
            linewidth=2,
            s=90,
            marker="o",
            zorder=5
        )

        # End marker
        ax.scatter(
            xs[-1],
            ys[-1],
            color="orange",
            edgecolor=color,
            linewidth=2,
            s=90,
            marker="o",
            zorder=5
        )

        ax.annotate(
            name,
            (xs[0], ys[0]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=9,
            color=color,
            fontweight="bold"
        )

    # Stage boundary
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

    # Legend
    ax.scatter([], [], color="green", s=80, label="Start")
    ax.scatter([], [], color="orange", s=80, label="End")

    ax.set_title(
        f"{data['song_name']} | "
        f"{data.get('predicted_genre', 'unknown')} | "
        f"{num_robots} Robots"
    )

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax.set_xlim(
        boundary["x_min"] - 0.5,
        boundary["x_max"] + 0.5
    )

    ax.set_ylim(
        boundary["y_min"] - 0.5,
        boundary["y_max"] + 0.5
    )

    output_file = (
        folder
        / f"{Path(data['song_name']).stem}_{num_robots}robots_path.png"
    )

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_file


# ==========================================
# REGION OCCUPANCY ANALYSIS
# ==========================================

def write_analysis(data, folder, step=0.025):

    if step <= 0:
        raise ValueError("step must be positive")

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    boundary = data["boundary"]
    margin = data["safety_margin"]
    duration = float(data["duration"])

    result = {
        "song_name": data["song_name"],
        "duration": duration,
        "num_robots": len(data["robots"]),
        "required_centre_separation": data[
            "required_centre_separation"
        ],
        "observed_minimum_separation": data[
            "observed_minimum_separation"
        ],
        "note": (
            "Residence time is estimated from simulated commands, "
            "not measured physical movement."
        ),
        "robots": {}
    }

    for name, robot in data["robots"].items():

        region_time = {
            r: 0.0 for r in REGION_CELLS
        }

        cell_time = {
            f"r{r+1}c{c+1}": 0.0
            for r in range(4)
            for c in range(4)
        }

        entries = {
            r: 0 for r in REGION_CELLS
        }

        transitions = []
        previous_region = None

        pose = tuple(robot["start_pose"])
        previous_end = 0.0

        def observe(start, end, velocity, angular):

            nonlocal pose, previous_region

            if end <= start:
                return

            count = max(
                1,
                math.ceil((end - start) / step)
            )

            dt = (end - start) / count

            for k in range(count):

                middle = integrate(
                    *pose,
                    velocity,
                    angular,
                    dt / 2
                )

                cell = cell_for_position(
                    middle[0],
                    middle[1],
                    boundary,
                    margin
                )

                region = region_for_position(
                    middle[0],
                    middle[1],
                    boundary,
                    margin
                )

                region_time[region] += dt

                cell_name = (
                    f"r{cell[0]+1}c{cell[1]+1}"
                )

                cell_time[cell_name] += dt

                if region != previous_region:

                    entries[region] += 1

                    transitions.append({
                        "time": round(
                            start + k * dt, 4
                        ),
                        "source": previous_region,
                        "destination": region,
                    })

                    previous_region = region

                pose = integrate(
                    *pose,
                    velocity,
                    angular,
                    dt
                )

        commands = sorted(
            robot["commands"],
            key=lambda c: c["start_time"]
        )

        for cmd in commands:

            start = float(cmd["start_time"])
            end = float(cmd["end_time"])

            if start > previous_end:
                observe(
                    previous_end,
                    start,
                    0.0,
                    0.0
                )

            observe(
                start,
                end,
                float(cmd["linear_x"]),
                float(cmd["angular_z"])
            )

            previous_end = max(previous_end, end)

        if previous_end < duration:
            observe(
                previous_end,
                duration,
                0.0,
                0.0
            )

        summary = {}

        for region in REGION_CELLS:

            chosen = [
                t for t in robot["targets"]
                if t["region"] == region
            ]

            summary[region] = {
                "selected_count": len(chosen),
                "reached_count": sum(
                    t["outcome"] == "reached"
                    for t in chosen
                ),
                "timeout_count": sum(
                    t["outcome"] == "timeout"
                    for t in chosen
                ),
                "blocked_count": sum(
                    t["outcome"] == "blocked"
                    for t in chosen
                ),
                "physical_entries": entries[region],
                "time_spent_seconds": round(
                    region_time[region], 3
                ),
                "time_spent_percent": (
                    round(
                        100 * region_time[region] / duration,
                        2
                    )
                    if duration else 0
                )
            }

        result["robots"][name] = {
            "region_summary": summary,
            "cell_occupancy_seconds": {
                k: round(v, 3)
                for k, v in cell_time.items()
            },
            "region_transitions": transitions,
            "target_history": robot["targets"],
            "yielded_seconds": robot["yielded_seconds"],
            "target_replans": robot["target_replans"],
        }

    path = (
        folder
        / f"{Path(data['song_name']).stem}_{len(data['robots'])}robots_analysis.json"
    )

    path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    return result, path


# ==========================================
# MULTI-ROBOT MP4 ANIMATION
# ==========================================

def animate_multi_robot(
    data,
    audio_file,
    folder,
    preview=30,
    fps=12
):
    """
    Animate multiple ROSbots using the same appearance and
    movement-state colours as the single-robot animator.

    STOPPED  = red
    BACKWARD = blue
    ROTATING = yellow
    MOVING   = turquoise
    """

    if fps <= 0:
        raise ValueError("FPS must be positive.")

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    duration = float(data["duration"])

    limit = (
        duration
        if preview <= 0
        else min(duration, float(preview))
    )

    if limit <= 0:
        raise ValueError("Animation duration must be positive.")

    names = list(data["robots"])

    if not names:
        raise ValueError("No robots to animate.")

    identity_colors = robot_colors(names)

    # ==========================================
    # PRECOMPUTE ROBOT MOTION
    # ==========================================

    prepared = {}

    for name in names:

        records = segments_for_robot(
            data["robots"][name],
            duration
        )

        starts = [
            record[0]
            for record in records
        ]

        prepared[name] = (records, starts)

    num_frames = max(
        1,
        math.ceil(limit * fps)
    )

    frame_times = np.arange(num_frames) / fps

    states = {
        name: [
            state_at_time(
                *prepared[name],
                float(t)
            )
            for t in frame_times
        ]
        for name in names
    }

    trails_data = {
        name: (
            np.array([
                state[0]
                for state in states[name]
            ]),
            np.array([
                state[1]
                for state in states[name]
            ])
        )
        for name in names
    }

    # ==========================================
    # CREATE FIGURE
    # ==========================================

    fig, ax = plt.subplots(figsize=(9, 8))

    # Same dark theme as the single-robot animator
    fig.patch.set_facecolor("#101827")
    ax.set_facecolor("#182438")

    ax.set_aspect("equal")

    boundary = data["boundary"]

    ax.set_xlim(
        boundary["x_min"] - 0.5,
        boundary["x_max"] + 0.5
    )

    ax.set_ylim(
        boundary["y_min"] - 0.5,
        boundary["y_max"] + 0.5
    )

    ax.tick_params(colors="white")

    ax.set_xlabel(
        "X position (m)",
        color="white"
    )

    ax.set_ylabel(
        "Y position (m)",
        color="white"
    )

    ax.grid(
        True,
        alpha=0.15,
        color="white"
    )

    # ==========================================
    # STAGE BOUNDARY
    # ==========================================

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

    # ==========================================
    # CREATE ROBOT MODELS
    # ==========================================

    robot_length = 0.35
    robot_width = 0.25

    robot_artists = {}

    for name in names:

        # Same body as single-robot animation
        robot_body = Rectangle(
            (
                -robot_length / 2,
                -robot_width / 2
            ),
            robot_length,
            robot_width,
            facecolor="#2DD4BF",
            edgecolor="white",
            linewidth=1.5,
            zorder=5
        )

        ax.add_patch(robot_body)

        # Four wheels
        wheels = []

        for wheel_x in (-0.11, 0.11):

            for wheel_y in (-0.16, 0.16):

                wheel = Rectangle(
                    (
                        wheel_x - 0.055,
                        wheel_y - 0.035
                    ),
                    0.11,
                    0.07,
                    facecolor="#090D15",
                    edgecolor="#94A3B8",
                    linewidth=1,
                    zorder=6
                )

                ax.add_patch(wheel)
                wheels.append(wheel)

        # Direction arrow
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

        # Movement trail
        trail, = ax.plot(
            [],
            [],
            color="#2DD4BF",
            linewidth=2,
            alpha=0.8
        )

        # Small robot identifier
        robot_label = ax.text(
            0,
            0,
            name.replace("_", " ").upper(),
            fontsize=8,
            color=identity_colors[name],
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=8
        )

        robot_artists[name] = {
            "body": robot_body,
            "wheels": wheels,
            "arrow": heading_arrow,
            "trail": trail,
            "label": robot_label,
        }

    # ==========================================
    # STATUS PANEL
    # ==========================================

    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=9,
        family="monospace",
        bbox={
            "facecolor": "#101827",
            "alpha": 0.9,
            "edgecolor": "none",
            "pad": 10
        },
        zorder=10
    )

    title = (
        f"{data['song_name']}  |  "
        f"{data.get('predicted_genre', 'unknown')}  |  "
        f"{len(names)} robots"
    )

    ax.set_title(
        title,
        color="white",
        fontsize=14,
        pad=15
    )

    # ==========================================
    # ANIMATE ROBOTS
    # ==========================================

    def update(frame):

        t = float(frame_times[frame])

        status_lines = [
            f"Time: {t:6.2f}s"
        ]

        artists = [status_text]

        for name in names:

            x, y, heading, v, w, action = states[name][frame]

            parts = robot_artists[name]

            body = parts["body"]
            wheels = parts["wheels"]
            arrow = parts["arrow"]
            trail = parts["trail"]
            label = parts["label"]

            # ----------------------------------
            # Position and rotation
            # ----------------------------------

            transform = (
                Affine2D()
                .rotate(heading)
                .translate(x, y)
                + ax.transData
            )

            body.set_transform(transform)
            artists.append(body)

            for wheel in wheels:
                wheel.set_transform(transform)
                artists.append(wheel)

            arrow.set_positions(
                (x, y),
                (
                    x + 0.32 * math.cos(heading),
                    y + 0.32 * math.sin(heading)
                )
            )

            artists.append(arrow)

            # ----------------------------------
            # Movement trail
            # ----------------------------------

            xs, ys = trails_data[name]

            trail.set_data(
                xs[:frame + 1],
                ys[:frame + 1]
            )

            artists.append(trail)

            label.set_position(
                (x, y + 0.29)
            )

            artists.append(label)

            # ----------------------------------
            # MOVEMENT STATE COLOURS
            # Same as single-robot animation
            # ----------------------------------

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

            body.set_facecolor(color)
            trail.set_color(color)

            # ----------------------------------
            # STATUS INFORMATION
            # ----------------------------------

            status_lines.extend([
                f"{name}: {action}",
                f"  State: {movement}",
                f"  Linear: {v:+.3f} m/s",
                f"  Angular: {w:+.3f} rad/s",
                f"  Position: ({x:+.2f}, {y:+.2f})",
                f"  Heading: {math.degrees(heading):+.1f} deg",
            ])

        status_text.set_text(
            "\n".join(status_lines)
        )

        return artists

    # ==========================================
    # CREATE ANIMATION
    # ==========================================

    animation = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps,
        blit=False
    )

    song_stem = Path(
        data["song_name"]
    ).stem

    output_file = (
        folder
        / f"{song_stem}_{len(names)}robots_{int(limit)}s.mp4"
    )

    # ==========================================
    # SAVE MP4 WITH ORIGINAL MUSIC
    # ==========================================

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
                    extra_args=[
                        "-pix_fmt",
                        "yuv420p"
                    ]
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
                    "-t", str(limit),
                    "-movflags", "+faststart",
                    str(output_file)
                ],
                check=True
            )

    finally:
        plt.close(fig)

    return output_file


# ==========================================
# BACKWARD-COMPATIBLE ALIASES
# ==========================================

def plot_duet(data, folder):
    return plot_multi_robot(data, folder)


def animate_duet(
    data,
    audio_file,
    folder,
    preview=30,
    fps=12
):
    return animate_multi_robot(
        data,
        audio_file,
        folder,
        preview=preview,
        fps=fps
    )
