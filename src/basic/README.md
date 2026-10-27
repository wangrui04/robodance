# RoboDance — Basic Beat-Synchronized Movement

## Overview

This module implements a simplified version of RoboDance where a robot moves forward and backward in synchronization with music beats.

Unlike the main RoboDance pipeline, which uses genre classification, music features, and motion planning, this module focuses on a single movement pattern: **alternating forward and backward movement every N beats**.

The goal is to establish a simple baseline for evaluating music synchronization before implementing more complex choreography on the ROSbot.

## How It Works

The pipeline follows these steps:

1. **Beat Detection:** Uses `librosa` to identify beat timestamps in an MP3 file.
2. **Movement Generation:** Groups beats into intervals based on the configured number of beats per movement.
3. **Direction Alternation:** Alternates between forward and backward movement after each interval.
4. **ROSbot Command Generation:** Produces linear and angular velocity commands for each movement.
5. **Simulation:** Plots the robot's simulated position over time.

The robot moves along a single straight line without turning.

## Project Structure

```text
robodance/
├── music/                        # Input MP3 files
├── output/
│   └── basic/                    # Generated outputs
└── src/
    ├── pipeline.py               # Main RoboDance pipeline
    └── basic/
        ├── beat_motion.py        # Beat detection and movement generation
        ├── pipeline.py           # Basic pipeline entry point
        └── README.md
```

## Requirements

- Python 3.10 or newer
- librosa
- numpy
- matplotlib

Install the required packages:

```powershell
python -m pip install librosa numpy matplotlib
```

## How to Run

### 1. Add music files

Place your MP3 files inside the `music/` folder at the root of the repository.

```text
music/
├── river.mp3
├── 16Shots.mp3
└── Tournelle.mp3
```

### 2. Open the project root

In PowerShell, navigate to the RoboDance repository:

```powershell
cd C:\Users\wangr\BezzoLab\robodance
```

### 3. Run the basic pipeline

To alternate between forward and backward movement every **2 beats**:

```powershell
python src/basic/pipeline.py --beats-per-move 2
```

This processes all MP3 files in the `music/` folder.

### 4. Customize movement behavior

The following arguments can be configured:

| Argument | Description | Default |
|---|---|---|
| `--beats-per-move` | Number of beats before reversing direction | `2` |
| `--speed` | Robot's linear speed in meters per second | `0.12` |
| `--start-direction` | Initial direction (`forward` or `backward`) | `forward` |

**Example 1: Reverse every beat**

```powershell
python src/basic/pipeline.py --beats-per-move 1
```

**Example 2: Reverse every 4 beats**

```powershell
python src/basic/pipeline.py --beats-per-move 4
```

**Example 3: Move at a different speed**

```powershell
python src/basic/pipeline.py --beats-per-move 2 --speed 0.15
```

**Example 4: Start by moving backward**

```powershell
python src/basic/pipeline.py --beats-per-move 3 --start-direction backward
```

## Generated Outputs

After running the pipeline, files are saved in:

```text
output/basic/
├── river_basic_commands.json
├── river_basic_path.png
├── 16Shots_basic_commands.json
├── 16Shots_basic_path.png
└── ...
```

### Movement Commands (`*_basic_commands.json`)

Contains the generated velocity commands, including:

- `start_time`: When movement begins, in seconds
- `end_time`: When movement ends, in seconds
- `direction`: Forward or backward
- `linear_x`: Linear velocity in meters per second
- `angular_z`: Angular velocity in radians per second (always zero)
- `duration`: Duration of the movement in seconds

Positive `linear_x` represents forward movement, while negative `linear_x` represents backward movement.

### Movement Simulation (`*_basic_path.png`)

Displays a graph of the robot's simulated position against time.

The graph shows how the robot repeatedly moves forward and backward according to the detected beats.

## Example

Consider a song with a tempo of 120 BPM.

Each beat lasts approximately 0.5 seconds.

With `--beats-per-move 2`, the movement pattern would be:

| Time | Movement |
|---|---|
| 0–1 s | Forward |
| 1–2 s | Backward |
| 2–3 s | Forward |
| 3–4 s | Backward |
| 4–5 s | Forward |

This is an illustrative example assuming the first detected beat occurs at 0 seconds and the beat intervals are regular. Actual movement timing is determined by detected beat timestamps.

## Current Limitations

- Only forward and backward movements are supported.
- Beat detection accuracy depends on the audio.
- Direction reversals are instantaneous in the simulation.
- The robot does not currently account for acceleration, deceleration, or physical constraints.
- This module generates commands and offline simulations but does not control a physical ROSbot.
- Movements begin at the first detected beat rather than necessarily at the beginning of the song.

## Future Improvements

1. Introduce smooth acceleration and deceleration while maintaining beat synchronization.
2. Compare movement patterns using 1, 2, 3, and 4 beats per direction.
3. Add ROS2 integration for publishing velocity commands to `/cmd_vel`.
4. Test synchronization and movement accuracy on the physical ROSbot.
5. Use the basic movement system as a baseline for comparing more advanced music-driven choreography.
