# RoboDance

A research project exploring how autonomous robots can generate dance-like motion from music.

The long-term goal is to enable robots to improvise movements that adapt to musical structure, energy, rhythm, and genre. The current system processes music files and generates robot motion commands that can later be deployed on platforms such as ROSbot and Boston Dynamics Spot.

---

## Project Pipeline

```text
MP3
 ↓
Feature Extraction
 ↓
Genre Classification
 ↓
Music State Extraction
 ↓
Motion Planning
 ↓
Motion Parameter Generation
 ↓
ROSbot Command Generation
 ↓
Visualization & Animation
```

---

## Current Capabilities

### 1. Feature Extraction

Extracts global musical features:

* Tempo (BPM)
* Beat locations
* Average energy
* Spectral brightness

Output:

```text
output/features/
```

Example:

```json
{
  "tempo_bpm": 123.05,
  "avg_energy": 0.194,
  "brightness": 1979.29
}
```

---

### 2. Genre Classification

Performs rule-based music genre prediction using extracted features.

Current genres:

* Contemporary
* Pop
* Hip-Hop
* EDM
* Jazz
* Unknown

Output:

```text
output/genres/
```

---

### 3. Music State Extraction

Divides each song into 1-second windows and computes local musical characteristics.

For each second:

* Energy
* Brightness
* Energy change
* Brightness change
* Movement quality

Movement quality labels:

```text
still
smooth
flowing
strong
```

Output:

```text
output/music_states/
```

---

### 4. Motion Planning

Converts musical states into high-level dance actions.

Examples:

```text
slow_sway
flowing_arc
sharp_turn
strong_pulse
expand_movement
contract_movement
large_flowing_arc
end_pose
```

Output:

```text
output/motion_timelines/
```

---

### 5. Motion Parameter Generation

Transforms symbolic dance actions into motion parameters.

Parameters include:

```text
speed
amplitude
duration
direction
turn_angle
curvature
motion_shape
body_level
repeat
```

Output:

```text
output/motion_parameters/
```

---

### 6. ROSbot Command Generation

Converts motion parameters into robot navigation commands.

Current command format:

```json
{
  "linear_x": 0.20,
  "angular_z": 0.35,
  "duration": 2.0
}
```

Output:

```text
output/rosbot_commands/
```

Future versions will publish directly to ROS2 `/cmd_vel`.

---

### 7. Motion Visualization

Generates 2D visualizations of robot movement trajectories.

Output:

```text
output/visualizations/
```

---

### 8. Motion Animation

Creates animated robot movement previews from generated motion parameters.

Output:

```text
output/animations/
```

---

## Running the Pipeline

Place audio files into:

```text
music/
```

Supported format:

```text
.mp3
```

Run:

```bash
python pipeline.py
```

The pipeline will automatically process every song inside the music directory.

---

## Current Research Focus

Current work focuses on:

* Genre-aware movement generation
* Music structure analysis
* Dance improvisation
* Motion planning for mobile robots
* Robot choreography generation

---

## Future Work

### Short-Term

* Improved genre classifier
* Beat-level motion generation
* Section-aware choreography
* ROSbot simulation environment

### Mid-Term

* ROS2 integration
* Real-time music processing
* Live motion generation
* Multi-song evaluation dataset

### Long-Term

* Deployment on ROSbot
* Deployment on Boston Dynamics Spot
* Style-specific dance generation
* Robot dance improvisation
* Human-robot dance interaction

```
```
