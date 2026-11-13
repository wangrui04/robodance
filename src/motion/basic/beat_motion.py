
import json
import math
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np


def detect_beats(audio_file):
    """
    Detect beat timestamps from an audio file.
    """
    y, sr = librosa.load(audio_file, sr=22050)

    tempo, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        trim=False
    )

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr
    )

    tempo = float(np.asarray(tempo).item())
    duration = librosa.get_duration(y=y, sr=sr)

    return {
        "song_name": Path(audio_file).name,
        "tempo": tempo,
        "duration": float(duration),
        "beat_times": beat_times.tolist()
    }


def generate_beat_commands(
    beat_data,
    beats_per_move=2,
    speed=0.12,
    start_direction="forward",
    pause_beats=1
):
    """
    Generate forward/backward commands with pauses.

    Example with beats_per_move=2 and pause_beats=1:

        Forward  - 2 beats
        Stop     - 1 beat
        Backward - 2 beats
        Stop     - 1 beat
        Repeat

    All movement transitions follow detected beat timestamps.
    """

    if beats_per_move < 1:
        raise ValueError("beats_per_move must be >= 1")

    if pause_beats < 0:
        raise ValueError("pause_beats must be >= 0")

    if speed <= 0:
        raise ValueError("speed must be positive")

    if start_direction not in ("forward", "backward"):
        raise ValueError(
            "start_direction must be 'forward' or 'backward'"
        )

    beat_times = beat_data["beat_times"]
    duration = beat_data["duration"]

    if len(beat_times) < 2:
        raise ValueError("Not enough beats detected")

    beat_interval = float(np.median(np.diff(beat_times)))

    # Extend beat boundaries through the end of the song.
    boundaries = list(beat_times)

    while boundaries[-1] < duration:
        boundaries.append(boundaries[-1] + beat_interval)

    commands = []

    beat_index = 0
    direction = (
        1 if start_direction == "forward" else -1
    )

    def add_command(start, end, velocity, label):
        end = min(end, duration)

        if end <= start:
            return

        commands.append({
            "start_time": round(float(start), 3),
            "end_time": round(float(end), 3),
            "direction": label,
            "linear_x": round(float(velocity), 3),
            "angular_z": 0.0,
            "duration": round(float(end - start), 3)
        })

    while beat_index < len(boundaries) - 1:

        # Movement phase
        move_start = boundaries[beat_index]

        move_end_index = min(
            beat_index + beats_per_move,
            len(boundaries) - 1
        )

        move_end = boundaries[move_end_index]

        add_command(
            move_start,
            move_end,
            direction * speed,
            "forward" if direction == 1 else "backward"
        )

        beat_index = move_end_index

        # Pause phase
        if pause_beats > 0 and beat_index < len(boundaries) - 1:

            pause_start = boundaries[beat_index]

            pause_end_index = min(
                beat_index + pause_beats,
                len(boundaries) - 1
            )

            pause_end = boundaries[pause_end_index]

            add_command(
                pause_start,
                pause_end,
                0.0,
                "stop"
            )

            beat_index = pause_end_index

        direction *= -1

    return {
        "song_name": beat_data["song_name"],
        "tempo": beat_data["tempo"],
        "beats_per_move": beats_per_move,
        "pause_beats": pause_beats,
        "speed": speed,
        "num_commands": len(commands),
        "commands": commands
    }


def simulate_beat_motion(commands_data, output_folder):
    """
    Simulate the robot's position over time.

    The robot moves along one axis:
    - Positive velocity: forward
    - Negative velocity: backward
    - Zero velocity: stationary
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    times = [0.0]
    positions = [0.0]

    x = 0.0

    for command in commands_data["commands"]:

        start_time = command["start_time"]
        duration = command["duration"]
        velocity = command["linear_x"]

        # Hold position before the next command.
        if start_time > times[-1]:
            times.append(start_time)
            positions.append(x)

        steps = max(1, math.ceil(duration / 0.05))
        dt = duration / steps

        for step in range(steps):
            x += velocity * dt

            current_time = start_time + (step + 1) * dt

            times.append(current_time)
            positions.append(x)

    song_stem = Path(commands_data["song_name"]).stem

    output_file = (
        output_folder / f"{song_stem}_basic_path.png"
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        times,
        positions,
        linewidth=2,
        label="Robot position"
    )

    plt.axhline(
        0,
        color="gray",
        linestyle="--",
        alpha=0.5
    )

    # Highlight stationary intervals.
    for command in commands_data["commands"]:
        if command["direction"] == "stop":
            plt.axvspan(
                command["start_time"],
                command["end_time"],
                color="orange",
                alpha=0.18
            )

    plt.title(
        f"{song_stem} - Beat-Synchronized Movement\n"
        f"{commands_data['beats_per_move']} beats per move, "
        f"{commands_data['pause_beats']} beats per pause"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Robot position (meters)")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_file, dpi=150)
    plt.close()

    return output_file


def save_commands(commands_data, output_folder):
    """
    Save generated movement commands to JSON.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    song_stem = Path(commands_data["song_name"]).stem

    output_file = (
        output_folder / f"{song_stem}_basic_commands.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            commands_data,
            f,
            indent=4
        )

    return output_file
