
from pathlib import Path

from feature_extractor import analyze_song
from genre_classifier import classify_genre, save_genre_result
from music_state_extractor import generate_music_state
from motion_planner import generate_motion_timeline
from motion_parameter_generator import generate_motion_parameters
from motion_visualizer import visualize_motion
from motion_animator import animate_motion
from rosbot_command_generator import generate_rosbot_commands


# ==========================================
# SETTINGS - CHANGE THESE VALUES
# ==========================================

SONG = None      # None = process all songs
PREVIEW = 60               # Seconds (0 = full song)
FPS = 12

# ==========================================
# MUSIC AND MOVEMENT TIMING
# ==========================================

STRENGTH = 0.65            # Beat strength threshold
MIN_MOVEMENT_GAP = 2.0     # Seconds between movement changes
STOP = 0.22                # Pause before next selected beat


# ==========================================
# SPATIAL EXPLORATION
# ==========================================

# Probability of selecting each region as a destination
REGION_WEIGHTS = {
    "centre": 0.50,
    "top_left": 0.125,
    "top_right": 0.125,
    "bottom_left": 0.125,
    "bottom_right": 0.125,
}

SAFETY_MARGIN = 0.40       # Metres from stage boundary
TARGET_INTERVAL = 7.0      # Seconds between destination selections
TARGET_RADIUS = 0.35       # Metres: distance considered near target
RANDOM_SEED = 42           # Reproducible movement choices


# ==========================================
# ROBOT MOVEMENT
# ==========================================

TRAVEL_SPEED = 0.32        # Base linear speed (m/s)


# ==========================================
# FOLDERS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MUSIC_FOLDER = PROJECT_ROOT / "music"
OUTPUT_FOLDER = PROJECT_ROOT / "output"


# ==========================================
# MAIN PIPELINE
# ==========================================

def main():

    if SONG is None:
        audio_files = sorted(MUSIC_FOLDER.glob("*.mp3"))
    else:
        audio_files = [MUSIC_FOLDER / SONG]

    if not audio_files:
        print("No MP3 files found.")
        return

    for audio_file in audio_files:

        if not audio_file.is_file():
            print(f"File not found: {audio_file}")
            continue

        print(f"\nProcessing {audio_file.name}")

        # --------------------------------------
        # 1. Extract musical features
        # --------------------------------------

        features = analyze_song(
            audio_file,
            OUTPUT_FOLDER / "features"
        )

        # --------------------------------------
        # 2. Classify genre
        # --------------------------------------

        genre = classify_genre(features)

        save_genre_result(
            genre,
            OUTPUT_FOLDER / "genres"
        )

        # --------------------------------------
        # 3. Extract beats and musical states
        # --------------------------------------

        music_state = generate_music_state(
            audio_file,
            OUTPUT_FOLDER / "music_states"
        )

        # --------------------------------------
        # 4. Select musical movement changes
        # --------------------------------------

        motion_timeline = generate_motion_timeline(
            music_state,
            genre,
            OUTPUT_FOLDER / "motion_timelines",
            strength_threshold=STRENGTH,
            stop_seconds=STOP,
            min_movement_gap=MIN_MOVEMENT_GAP
        )

        # --------------------------------------
        # 5. Generate movement parameters
        # --------------------------------------

        motion_parameters = generate_motion_parameters(
            motion_timeline,
            OUTPUT_FOLDER / "motion_parameters"
        )

        # --------------------------------------
        # 6. Generate ROSbot commands
        # --------------------------------------

        rosbot_commands = generate_rosbot_commands(
            motion_parameters,
            OUTPUT_FOLDER / "rosbot_commands",
            margin=SAFETY_MARGIN,
            region_weights=REGION_WEIGHTS,
            target_interval=TARGET_INTERVAL,
            target_radius=TARGET_RADIUS,
            seed=RANDOM_SEED,
            travel_speed=TRAVEL_SPEED
        )

        # --------------------------------------
        # 7. Visualize robot trajectory
        # --------------------------------------

        path_file = visualize_motion(
            rosbot_commands,
            OUTPUT_FOLDER / "visualizations"
        )

        # --------------------------------------
        # 8. Generate MP4 with original music
        # --------------------------------------

        video_file = animate_motion(
            rosbot_commands,
            audio_file,
            OUTPUT_FOLDER / "animations",
            preview_seconds=PREVIEW,
            fps=FPS
        )

        # --------------------------------------
        # 9. Print results
        # --------------------------------------

        print(f"\nGenre: {genre['predicted_genre']}")

        print(
            f"Selected accents: "
            f"{motion_timeline['num_strong_beats']}"
        )

        print(
            f"Motion events: "
            f"{motion_timeline['num_motion_events']}"
        )

        print(
            f"ROSbot commands: "
            f"{rosbot_commands['num_commands']}"
        )

        print(
            f"Target selections: "
            f"{len(rosbot_commands['targets'])}"
        )

        print(f"Path: {path_file}")
        print(f"MP4: {video_file}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
