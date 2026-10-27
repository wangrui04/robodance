# RoboDance: Music-Responsive Robotic Dance Improvisation

## Overview

RoboDance is a research project that explores how a mobile robot can generate dance-like movements in response to music.

Rather than following predetermined choreography or moving on every beat, the system analyzes musical features and uses them to make movement decisions. These decisions consider musical energy, beat strength, changes in musical intensity, movement quality, and spatial constraints.

The current implementation uses a simulated ROSbot moving within a square stage. The long-term goal is to develop a robotic system capable of meaningful music-responsive improvisation and eventually transfer the generated movements to physical robot hardware.

## Project Objectives

The main objectives are to:

- Extract musical features such as tempo, beat timing, energy, and spectral brightness.
- Identify meaningful musical accents rather than responding to every detected beat.
- Generate different movement primitives based on musical characteristics.
- Adapt between sharp and smooth movement qualities as the music changes.
- Encourage spatial exploration while maintaining a preference for the centre of the stage.
- Ensure that generated movements remain within a predefined boundary.
- Visualize the resulting robot trajectory and generate an MP4 animation synchronized with the original music.

## Project Structure

```text
robodance/
│
├── music/
│   └── 16Shots.mp3
│
├── src/
│   ├── pipeline.py
│   ├── feature_extractor.py
│   ├── genre_classifier.py
│   ├── music_state_extractor.py
│   ├── motion_planner.py
│   ├── motion_parameter_generator.py
│   ├── rosbot_command_generator.py
│   ├── motion_visualizer.py
│   ├── motion_animator.py
│   │
│   └── basic/
│       └── ...                  # Earlier baseline implementation
│
├── output/
│   ├── features/
│   ├── genres/
│   ├── music_states/
│   ├── motion_timelines/
│   ├── motion_parameters/
│   ├── rosbot_commands/
│   ├── visualizations/
│   └── animations/
│
└── README.md
```

The `src/basic/` directory contains the earlier beat-based simulation and is kept separate from the main implementation.

## Pipeline Architecture

```text
Input MP3
    |
    v
Music Feature Extraction
    |
    v
Genre Classification
    |
    v
Music State and Beat Analysis
    |
    v
Musical Accent Selection
    |
    v
Motion Planning
    |
    v
Motion Parameter Generation
    |
    v
Spatially Aware ROSbot Command Generation
    |
    +--------------------+
    |                    |
    v                    v
Trajectory PNG       MP4 Animation
                     with Music
```

### 1. Music Feature Extraction

**File:** `feature_extractor.py`

The system uses `librosa` to analyze an input audio file and extract:

- **Tempo:** Estimated beats per minute (BPM).
- **Beat timing:** Timestamps of detected musical beats.
- **Energy:** Root mean square (RMS) amplitude, representing musical intensity.
- **Brightness:** Spectral centroid, describing the distribution of higher-frequency content.

These features provide information about the overall musical character.

### 2. Genre Classification

**File:** `genre_classifier.py`

A heuristic classifier uses extracted musical features to estimate the genre or movement style associated with a song.

Supported categories include:

- Contemporary
- EDM
- Pop
- Hip-hop
- Jazz
- Unknown

Genre classification provides general musical context. However, the latest movement selection system does not rely exclusively on genre.

Instead, it considers how the musical characteristics change throughout the song.

### 3. Music State Extraction

**File:** `music_state_extractor.py`

The system analyzes the music over time to identify changing musical characteristics.

The extracted states include:

- Energy
- Brightness
- Energy changes
- Brightness changes
- Movement quality

Movement qualities include:

- Still
- Smooth
- Flowing
- Strong

The system also records detected beats, their timestamps, and normalized beat strengths.

### 4. Musical Accent Selection

**File:** `motion_planner.py`

The robot does not need to change movements on every beat.

Instead, the motion planner identifies meaningful musical accents and selects a subset of them as potential movement transition points.

The current approach:

1. Analyzes the strength of detected beats.
2. Uses a relative threshold based on the song's beat-strength distribution.
3. Identifies local peaks in beat strength.
4. Selects accents within configurable time windows.
5. Allows movements to continue across multiple beats.
6. Introduces short pauses before selected transitions.

This avoids the repetitive behavior of moving forward or backward on every beat.

The selected accents are converted into motion events containing start times, end times, durations, and musical information.

### 5. Motion Parameter Generation

**File:** `motion_parameter_generator.py`

The motion parameter generator converts planned movement events into structured motion parameters.

These include movement durations and parameters associated with the intended motion, such as speed, amplitude, and turning characteristics.

The resulting events are passed to the ROSbot command generator.

### 6. Music-Adaptive Movement Selection

**File:** `rosbot_command_generator.py`

The robot can select from multiple movement primitives.

**Sharp movement primitives:**

- `forward_pulse`
- `backward_pulse`
- `sharp_left`
- `sharp_right`

**Smooth movement primitives:**

- `left_curve`
- `right_curve`
- `s_wave`
- `slow_sway`

The movement selection system uses a sharpness score derived from:

- Current musical energy relative to the song.
- Changes in energy.
- Beat strength.
- Extracted movement quality.

Higher-energy passages generally favor sharper movements, while calmer passages generally favor smoother movements.

However, this is not a fixed classification.

A high-energy song can still produce smooth movements during quieter sections, while a calm song can produce sharp movements in response to strong accents or sudden changes.

This allows the robot to respond to local musical characteristics rather than performing the same type of movement throughout an entire song.

The system also discourages repetitive primitive selection to encourage movement variety.

### 7. Weighted Spatial Exploration

**File:** `rosbot_command_generator.py`

To encourage the robot to explore different parts of the stage, the system uses weighted destination selection.

The stage is divided into five destination regions:

| Region | Selection Weight |
|---|---:|
| Centre | 50% |
| Top left | 12.5% |
| Top right | 12.5% |
| Bottom left | 12.5% |
| Bottom right | 12.5% |

The centre is the preferred destination, but the robot can also travel toward different corners.

The robot periodically selects a new destination according to these weights.

A small random offset is applied to each destination so the robot does not repeatedly target exactly the same coordinates.

When traveling toward a destination, the system can generate a heading correction before executing an expressive movement primitive.

**Important:** These weights represent destination-selection probabilities, not guaranteed stage occupancy. The robot may not reach every selected region, particularly during short previews.

### 8. Boundary-Aware Motion

The robot operates within a predefined square stage.

The motion generator evaluates candidate trajectories and rejects movements that would cross the configured safety limit.

A safety margin is maintained between the robot's simulated centre and the stage boundary.

For example:

```python
SAFETY_MARGIN = 0.40
```

This prevents the robot's simulated centre from approaching closer than 0.40 metres to the boundary.

The current simulation uses a simplified kinematic model. Additional robot footprint, dynamics, and obstacle constraints will be required before physical deployment.

### 9. ROSbot Command Generation

The final movement decisions are converted into velocity commands compatible with a differential-drive-style robot model.

Each command includes:

```json
{
  "start_time": 2.0,
  "end_time": 3.5,
  "duration": 1.5,
  "action": "left_curve",
  "linear_x": 0.24,
  "angular_z": 0.65
}
```

These commands specify:

- Start and end time
- Movement duration
- Selected movement primitive
- Linear velocity
- Angular velocity

The command sequence is saved as JSON for simulation and potential future ROS integration.

### 10. Trajectory Visualization and Animation

**Files:**
- `motion_visualizer.py`
- `motion_animator.py`

The visualizer integrates the generated velocity commands to produce a top-down representation of the robot's trajectory.

The animator uses the same commands to generate an MP4 showing:

- Robot position and orientation
- Movement trajectory
- Stage boundary
- Current movement primitive
- Linear and angular velocities
- Original music synchronized with the animation

The animation duration can be configured independently of the full song duration.

## Installation

### Requirements

- Python 3.13
- NumPy
- librosa
- Matplotlib
- FFmpeg

Install the Python dependencies:

```bash
py -m pip install numpy librosa matplotlib
```

FFmpeg must also be installed and accessible through the system PATH.

Verify FFmpeg:

```bash
ffmpeg -version
```

## Running the Pipeline

Place MP3 files in the `music/` directory.

Open a terminal in the `src/` directory:

```bash
cd src
```

Run:

```bash
py pipeline.py
```

The pipeline will process the selected song, generate motion commands, and produce trajectory and animation outputs.

## Configuration

The main settings are located at the top of `src/pipeline.py`.

### Audio and Preview Settings

```python
SONG = "16Shots.mp3"
PREVIEW = 60
FPS = 12
```

Set `SONG = None` to process all MP3 files in the music directory.

Set `PREVIEW = 0` to animate the full song.

### Musical Movement Timing

```python
STRENGTH = 0.65
MIN_MOVEMENT_GAP = 2.0
STOP = 0.22
```

- `STRENGTH`: Beat-strength threshold setting used by the accent selector.
- `MIN_MOVEMENT_GAP`: Width of the movement-selection time window, in seconds.
- `STOP`: Pause duration before the next selected movement change.

The accent selector uses song-relative beat strengths, so the configured strength value is not necessarily an absolute cutoff.

### Spatial Exploration

```python
REGION_WEIGHTS = {
    "centre": 0.50,
    "top_left": 0.125,
    "top_right": 0.125,
    "bottom_left": 0.125,
    "bottom_right": 0.125,
}

SAFETY_MARGIN = 0.40
TARGET_INTERVAL = 7.0
TARGET_RADIUS = 0.35
RANDOM_SEED = 42
```

- `REGION_WEIGHTS`: Destination-selection probabilities.
- `SAFETY_MARGIN`: Minimum clearance from the stage boundary for the simulated robot centre.
- `TARGET_INTERVAL`: Time between destination selections.
- `TARGET_RADIUS`: Distance within which the robot is considered near its destination.
- `RANDOM_SEED`: Controls reproducibility of randomized movement and destination selections.

### Robot Speed

```python
TRAVEL_SPEED = 0.32
```

This controls the base linear speed used by the music-adaptive motion generator, in metres per second.

## Output Files

The pipeline generates files in the `output/` directory.

| Directory | Contents |
|---|---|
| `features/` | Extracted musical features |
| `genres/` | Predicted genre |
| `music_states/` | Beat information and time-varying musical states |
| `motion_timelines/` | Selected movement events |
| `motion_parameters/` | Parameterized movements |
| `rosbot_commands/` | Generated linear and angular velocity commands |
| `visualizations/` | Top-down robot trajectory PNG |
| `animations/` | MP4 simulation with original music |

For example, a 60-second preview of `16Shots.mp3` produces:

```text
output/animations/16Shots_animation_60s.mp4
```

## Current Development Status

The current prototype supports:

- [x] MP3 audio feature extraction
- [x] Beat detection and beat-strength analysis
- [x] Heuristic genre classification
- [x] Time-varying musical state extraction
- [x] Selective musical accent detection
- [x] Movement generation across multiple beats
- [x] Multiple sharp and smooth movement primitives
- [x] Music-adaptive sharpness scoring
- [x] Weighted spatial destination selection
- [x] Boundary-aware trajectory checking
- [x] ROSbot velocity command generation
- [x] Top-down trajectory visualization
- [x] MP4 animation synchronized with music

These features are implemented in the current simulation prototype; their choreographic quality and effectiveness are still being evaluated.

## Current Limitations

The project is still under active development.

**1. Spatial coverage**

Weighted destination selection does not guarantee that the robot will explore the entire stage. Its ability to reach destinations depends on movement duration, heading, velocity, and target-selection timing.

**2. Movement expressiveness**

The current movement vocabulary is based on combinations of linear and angular velocity. These movements approximate dance qualities but do not reproduce the full expressiveness of human dance.

**3. Musical interpretation**

The sharpness score and movement selection use heuristics rather than a trained model. The relationship between musical characteristics and movement quality requires further evaluation.

**4. Simulation versus physical execution**

The current system uses simplified robot kinematics. Physical deployment will require additional testing for robot dimensions, acceleration limits, control accuracy, and safety.

**5. Genre classification**

The genre classifier is heuristic and may not accurately identify all musical styles.

## Future Work

Planned improvements include:

1. **Spatial exploration evaluation:** Measure how evenly the robot explores the stage and whether it reaches selected destinations.
2. **Improved movement transitions:** Create smoother transitions between primitives and reduce abrupt navigation corrections.
3. **Musical section awareness:** Incorporate changes in musical structure to guide larger choreographic decisions.
4. **Expanded movement vocabulary:** Introduce more expressive motion patterns and combinations.
5. **Dance-style adaptation:** Explore movement choices inspired by contemporary, hip-hop, jazz, and other dance styles.
6. **Human evaluation:** Compare generated robot movement with how dancers interpret selected musical passages.
7. **Physical ROSbot integration:** Translate simulated commands into executable robot control with appropriate safety constraints.
8. **Potential extension to other robots:** Investigate whether the framework can be adapted to platforms with more expressive movement capabilities.

## Research Goal

The broader goal of RoboDance is to investigate how musical features can guide robotic movement decisions that create the appearance of meaningful dance improvisation.

The project focuses on moving beyond simple beat synchronization toward a system that responds to musical dynamics, changes movement quality, and makes spatial decisions while respecting physical constraints.

The ultimate objective is not to replicate human dance exactly, but to explore how a robot can demonstrate musical responsiveness and expressive variation through autonomous movement.