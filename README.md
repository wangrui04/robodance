# RoboDance: Music-Responsive Robotic Dance Improvisation

RoboDance is a research prototype exploring how one or more mobile robots can **improvise movements in response to music**, rather than following a fixed choreography or reacting to every beat. It combines music-feature analysis, movement selection, probabilistic spatial exploration, and 2D ROSbot-style simulation.

The project currently supports **single-robot and coordinated multi-robot simulations** through one entry point. Simulations produce top-down trajectory visualizations and MP4 animations accompanied by the original music. These are **simulated velocity commands and trajectories**, not autonomous execution on physical hardware.

## Features

- **Music analysis:** Beat timing and strength, tempo, RMS energy, spectral brightness, and time-varying musical states.
- **Music-responsive movement:** Musical accents guide motion events; energy, intensity changes, and beat strength influence sharp versus smooth movements.
- **Stochastic improvisation:** A new random seed is generated for each song on each run, so the same input song can yield different performances.
- **Spatial exploration:** A 4×4 grid grouped into five regions, with dynamic destination weights, distance preferences, selection history, and variable time spent exploring each selected region.
- **Single-robot simulation:** Boundary-aware motion generation, command-level output, region-occupancy analysis, trajectory PNG, and music-synchronized MP4.
- **Multi-robot simulation:** A configurable number of robots sharing a musical timeline while making individual movement decisions, with simulated pairwise separation checks and yielding/replanning behavior.
- **Visualization:** Distinct paths in multi-robot trajectory PNGs; animated ROSbot-style bodies with movement-state colors in MP4s.

## Quick start

### Prerequisites

- Python (the project has been developed using Python 3.13)
- FFmpeg installed and available on your system PATH
- Dependencies listed in `requirements.txt`

From the repository root, install dependencies:

```bash
py -m pip install -r requirements.txt
```

Check that FFmpeg is accessible:

```bash
ffmpeg -version
```

On systems where the Windows `py` launcher is unavailable, use `python` instead.

### Choose a song and robot count

Place `.mp3` files in the repository-root `music/` directory. Edit the settings near the top of `src/pipeline.py`:

```python
NUM_ROBOTS = 1               # 1 = solo; 2+ = coordinated multi-robot
SONG = None                  # None = process every *.mp3 in music/
PREVIEW = 0                  # 0 = full-length animation; otherwise seconds
FPS = 12
```

For a quicker test, set `SONG = "river.mp3"` (use the **exact filename and capitalization** of an existing file) and `PREVIEW = 30`.

Run the unified pipeline **from the repository root**:

```bash
py src/pipeline.py
```

There is no separate preprocessing command to run. The pipeline performs music analysis, motion planning, command generation, and rendering in sequence.

> **Output behavior:** Music analysis still processes the full song when `PREVIEW` limits the video. Runs with the same song/output filenames overwrite earlier generated results. Save a copy of any performance you wish to retain.

## Project structure

```text
robodance/
├── README.md
├── LICENSE
├── requirements.txt
├── music/                         # Input MP3 files
├── output/                        # Generated artifacts
└── src/
    ├── pipeline.py                # Unified single-/multi-robot entry point
    ├── music_analysis/
    │   ├── feature_extractor.py
    │   ├── genre_classifier.py
    │   ├── music_state_extractor.py
    │   └── section_detection.py
    ├── motion/
    │   ├── motion_planner.py
    │   ├── motion_parameter_generator.py
    │   ├── rosbot_command_generator.py
    │   ├── region_analysis.py
    │   ├── motion_visualizer.py
    │   ├── motion_animator.py
    │   └── basic/                  # Earlier beat-synchronized baseline
    └── multi_robot/
        ├── planner.py
        └── visualization.py
```

`src/motion/basic/` contains an earlier baseline and its own documentation. The main workflow described here is `src/pipeline.py`.

## Pipeline architecture

```text
Input MP3
   │
   ▼
Feature extraction + heuristic genre classification
   │
   ▼
Musical states, beat strengths, and accent selection
   │
   ▼
Motion timeline + motion parameters
   │
   ├── NUM_ROBOTS = 1 ──► Single-robot command generator
   │                       ├── Region-occupancy analysis
   │                       ├── Trajectory PNG
   │                       └── MP4 with original audio
   │
   └── NUM_ROBOTS >= 2 ─► Coordinated multi-robot planner
                           ├── Shared pairwise separation supervision
                           ├── Combined trajectory PNG
                           └── Multi-robot MP4 with original audio
```

### Music analysis and movement planning

- `src/music_analysis/feature_extractor.py` extracts global audio features.
- `src/music_analysis/genre_classifier.py` uses heuristic rules to classify the song into contemporary, EDM, pop, hip-hop, jazz, or unknown. These are estimates, not learned genre predictions.
- `src/music_analysis/music_state_extractor.py` extracts time-varying musical information and beat strengths.
- `src/motion/motion_planner.py` selects salient musical accents and constructs movement events, rather than changing motion at every beat.
- `src/motion/motion_parameter_generator.py` attaches movement parameters to those events.

The single-robot command generator uses the music to influence the likelihood of choosing **sharp** or **smooth** primitives:

| Sharp primitives | Smooth primitives |
| --- | --- |
| `forward_pulse`, `backward_pulse` | `left_curve`, `right_curve` |
| `sharp_left`, `sharp_right` | `s_wave`, `slow_sway` |

These are simulated combinations of linear and angular velocity, not full-body humanoid dance motions. Multi-robot choreography uses music-responsive movement and navigation controls within a coordinated simulator; its commands need not be identical to the solo primitive sequences.

## Stochastic spatial exploration

Both simulation modes share a **4×4 grid**, organized into a central area and four surrounding logical regions. Each grid cell is individually addressable for target selection and occupancy measurement.

```text
 TL | TL | TR | TR
 TL |  C |  C | TR
 BL |  C |  C | BR
 BL | BL | BR | BR

C = centre; TL/TR/BL/BR = quadrants
```

The baseline region weights are:

| Region | Base weight |
| --- | ---: |
| Centre | 50% |
| Top left | 12.5% |
| Top right | 12.5% |
| Bottom left | 12.5% |
| Bottom right | 12.5% |

**These are starting weights, not fixed visit frequencies or occupancy percentages.** The generator adjusts the effective region-selection probabilities using:

1. **Distance weighting:** Nearby regions are generally favored relative to more distant regions.
2. **Selection history:** Previously chosen destinations receive temporarily reduced weighting, with memory decay allowing them to become attractive again.
3. **Consecutive-selection limits:** A region can be temporarily excluded after repeated consecutive selections.

Once a region is selected, a target point is sampled from its cells. The robot attempts to reach that point subject to a travel timeout, then explores within that region for a randomly sampled dwell interval (default **5–15 seconds**). Actual time spent within a region can differ from the planned dwell duration because of travel, musical event timing, and boundary behavior.

### Randomness and reproducibility

`src/pipeline.py` generates a fresh random seed for each processed song using `secrets.randbits(64)`, and prints it in the terminal. This allows different runs of the same song to produce different simulated choreographies. To reproduce a particular performance in an experiment, retain the seed **and** the code/configuration version, and pass the same seed to the relevant generator.

Note that repeated runs overwrite files using the same names; printing a seed does not automatically create an archival copy.

## Single-robot simulation

With `NUM_ROBOTS = 1`, the pipeline calls `motion.rosbot_command_generator.generate_rosbot_commands()`, then:

- `analyze_region_occupancy()` measures simulated time in each of the five regions and 16 cells, physical region entries, and destination outcomes.
- `visualize_motion()` renders a top-down trajectory PNG.
- `animate_motion()` creates a dark-themed ROSbot MP4 with position, heading, current action, and original audio.

The velocity commands include the action label, start/end times, `linear_x`, and `angular_z`. The single-robot planner checks stage constraints using its simulated kinematic model.

## Multi-robot simulation

With `NUM_ROBOTS >= 2`, the same music analysis and motion parameters feed `multi_robot.planner.generate_multi_robot_commands()`. The generator:

- Creates separate robot states, random choices, and selection histories.
- Places robots at separated initial positions within the available stage area.
- Advances all robots on a shared simulation clock.
- Checks proposed movements against stage constraints and the configured centre-to-centre separation.
- Uses a yielding supervisor and can replan destinations after prolonged blocking.
- Returns each robot's command history for rendering.

The initial illustrative collision geometry uses:

```python
ROBOT_RADIUS = 0.18
EXTRA_CLEARANCE = 0.18
TIME_STEP = 0.05
MAX_YIELD_BEFORE_REPLAN = 3.0
```

This gives a **0.54 m requested centre-to-centre separation** for the simplified circular footprints. The simulator checks multiple robot pairs, but crowded scenarios can cause frequent yielding or prevent a robot from reaching its destination. A higher robot count is configurable, **not a guarantee that arbitrary numbers of robots fit or perform successfully**.

The multi-robot visualizer creates a combined path PNG and a music-synchronized MP4 with ROSbot-style bodies. MP4 body colors indicate movement state: red = stopped, yellow = rotating, blue = backward, and turquoise = moving.

> **Safety limitation:** The collision and boundary checks are simulation-level approximations, not certified collision avoidance or physical-robot safety guarantees. Real robot deployment would require validated footprints, dynamics, stopping distances, sensing/localization uncertainty, and hardware emergency-stop procedures.

## Configuration reference

Change these values at the top of `src/pipeline.py`:

| Setting | Default in the current pipeline | Meaning |
| --- | --- | --- |
| `NUM_ROBOTS` | `1` | `1` uses the solo planner; `2+` uses multi-robot planning |
| `SONG` | `None` | Process all MP3 files, or specify one exact filename |
| `PREVIEW` | `0` | MP4 seconds; `0` means full song |
| `FPS` | `12` | Animation frames per second |
| `STRENGTH` | `0.65` | Accent-selection strength setting |
| `MIN_MOVEMENT_GAP` | `2.0` | Musical movement-event window in seconds |
| `STOP` | `0.22` | Pause before the next selected movement |
| `SAFETY_MARGIN` | `0.40` | Stage boundary margin (metres) |
| `TARGET_RADIUS` | `0.35` | Arrival tolerance (metres) |
| `TRAVEL_SPEED` | `0.32` | Base simulated travel speed (m/s) |
| `DISTANCE_LAMBDA` | `0.35` | Distance penalty in region selection |
| `VISIT_PENALTY` | `0.40` | Selection-history penalty strength |
| `VISIT_MEMORY_DECAY` | `0.85` | Visit-history decay after selections |
| `MAX_CONSECUTIVE_SELECTIONS` | `2` | Maximum consecutive region choices |
| `MAX_TRAVEL_TIME` | `15.0` | Travel timeout threshold (seconds) |
| `DWELL_MIN` / `DWELL_MAX` | `5.0` / `15.0` | Sampled exploration duration (seconds) |
| `ROBOT_RADIUS` | `0.18` | Multi-robot illustrative radius (metres) |
| `EXTRA_CLEARANCE` | `0.18` | Extra inter-robot clearance (metres) |
| `TIME_STEP` | `0.05` | Multi-robot simulation step (seconds) |
| `MAX_YIELD_BEFORE_REPLAN` | `3.0` | Blocked time before replan (seconds) |

These values are prototype parameters, not optimized or hardware-validated settings.

## Output files

Shared music-analysis results are written under `output/` regardless of the robot count:

| Directory | Contents |
| --- | --- |
| `features/` | Extracted global musical features |
| `genres/` | Heuristic genre classifications |
| `music_states/` | Musical state and beat information |
| `motion_timelines/` | Selected movement events |
| `motion_parameters/` | Parameterized movement events |

For **single-robot** runs, additional outputs include:

| Directory | Contents |
| --- | --- |
| `rosbot_commands/` | Velocity-command JSON |
| `region_analysis/` | Per-region/per-cell residence and target histories |
| `visualizations/` | Trajectory PNG |
| `animations/` | MP4 with the original music |

For **multi-robot** runs, the unified pipeline saves the visual results in:

```text
output/multi_robot/
├── visualizations/       # Combined robot trajectories (PNG)
└── animations/           # Multiple robots moving to the music (MP4)
```

The multi-robot planner also generates command data internally, but the unified pipeline uses a temporary directory for its command JSON and discards it after rendering. The printed terminal summary includes robot count, minimum observed simulated separation, yielding events, and target-arrival counts.

## Evaluating results

Useful measures for comparing performances include:

- Whether the same song produces varied trajectories across random seeds.
- Whether strong versus smooth musical passages produce different motion qualities.
- How often each region is selected versus how long the robot physically occupies it.
- Destination arrival, timeout, and replanning behavior.
- With multiple robots, minimum simulated separation, time spent yielding, and how well robots share stage space.

A high destination-selection weight does **not** imply a corresponding proportion of stage occupancy. The project distinguishes region-choice statistics from actual simulated travel and residence.

## Current limitations and future work

RoboDance is a developing 2D simulation. Music-to-motion mappings and genre labels are heuristic; movement expressiveness remains limited by differential-drive kinematics. Multi-robot behavior currently emphasizes independent music-responsive improvisation with spatial coordination, rather than explicit choreographic relationships.

Potential next steps include improving navigation and deadlock handling, introducing deliberate **mirroring, synchronization, canon, and call-and-response**, evaluating choreography with human dancers, exploring musical section-aware decisions, and validating ROS-based execution on physical mobile robots.

## Research goal

The goal is to understand how robots can exhibit musical responsiveness and expressive variation through autonomous, probabilistic movement decisions, individually and as a group. The intention is not to replicate human dance exactly, but to investigate how rhythmic structure, movement quality, space, and inter-robot coordination shape robotic improvisation.
