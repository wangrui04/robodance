
import secrets
import tempfile
from pathlib import Path

from music_analysis.feature_extractor import analyze_song
from music_analysis.genre_classifier import classify_genre, save_genre_result
from music_analysis.music_state_extractor import generate_music_state
from motion.motion_planner import generate_motion_timeline
from motion.motion_parameter_generator import generate_motion_parameters
from motion.motion_visualizer import visualize_motion
from motion.motion_animator import animate_motion
from motion.rosbot_command_generator import generate_rosbot_commands
from motion.region_analysis import analyze_region_occupancy

# ==========================================
# MULTI ROBOT
# ==========================================

from multi_robot.planner import generate_multi_robot_commands

from multi_robot.visualization import (
    plot_multi_robot,
    animate_multi_robot,
)


# ==========================================
# SETTINGS
# ==========================================

# Simulation
NUM_ROBOTS = 1             # 1 = solo, 2 = duet, 3 = trio, etc.
SONG = None          # None = process all songs
PREVIEW = 0               # Seconds (0 = full song)
FPS = 12

# Music and movement timing
STRENGTH = 0.65
MIN_MOVEMENT_GAP = 2.0
STOP = 0.22

# ==========================================
# SPATIAL EXPLORATION
# ==========================================

# 4x4 grid grouped into five logical regions
REGION_WEIGHTS = {
    "centre": 0.50,
    "top_left": 0.125,
    "top_right": 0.125,
    "bottom_left": 0.125,
    "bottom_right": 0.125,
}

SAFETY_MARGIN = 0.40
TARGET_RADIUS = 0.35
TRAVEL_SPEED = 0.32

# Dynamic region-selection probabilities
DISTANCE_LAMBDA = 0.35
VISIT_PENALTY = 0.40
VISIT_MEMORY_DECAY = 0.85
MAX_CONSECUTIVE_SELECTIONS = 2

# Hybrid travel + variable in-region exploration
MAX_TRAVEL_TIME = 15.0
DWELL_MIN = 5.0
DWELL_MAX = 15.0

# ==========================================
# MULTI-ROBOT COLLISION SETTINGS
# ==========================================

ROBOT_RADIUS = 0.18        # Approximate circular footprint (m)
EXTRA_CLEARANCE = 0.18     # Additional separation (m)

# Required centre-to-centre distance:
# 2 * ROBOT_RADIUS + EXTRA_CLEARANCE = 0.54m

TIME_STEP = 0.05           # Simulation timestep (seconds)
MAX_YIELD_BEFORE_REPLAN = 3.0

# ==========================================
# FOLDERS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MUSIC_FOLDER = PROJECT_ROOT / "music"
OUTPUT_FOLDER = PROJECT_ROOT / "output"

MULTI_OUTPUT_FOLDER = OUTPUT_FOLDER / "multi_robot"


# ==========================================
# MAIN PIPELINE
# ==========================================

def main():

    if not isinstance(NUM_ROBOTS, int) or NUM_ROBOTS < 1:
        raise ValueError("NUM_ROBOTS must be a positive integer.")

    audio_files = (
        sorted(MUSIC_FOLDER.glob("*.mp3"))
        if SONG is None
        else [MUSIC_FOLDER / SONG]
    )

    if not audio_files:
        print("No MP3 files found.")
        return

    for audio_file in audio_files:

        if not audio_file.is_file():
            print(f"File not found: {audio_file}")
            continue

        print("\n====================================")
        print(f"Processing: {audio_file.name}")
        print(f"Number of robots: {NUM_ROBOTS}")
        print("====================================")

        # New seed for every song and every run
        random_seed = secrets.randbits(64)

        print(f"Random seed: {random_seed}")

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
        # 3. Extract musical states
        # --------------------------------------

        music_state = generate_music_state(
            audio_file,
            OUTPUT_FOLDER / "music_states"
        )

        # --------------------------------------
        # 4. Generate motion timeline
        # --------------------------------------

        motion_timeline = generate_motion_timeline(
            music_state,
            genre,
            OUTPUT_FOLDER / "motion_timelines",
            strength_threshold=STRENGTH,
            stop_seconds=STOP,
            min_movement_gap=MIN_MOVEMENT_GAP,
        )

        # --------------------------------------
        # 5. Generate movement parameters
        # --------------------------------------

        motion_parameters = generate_motion_parameters(
            motion_timeline,
            OUTPUT_FOLDER / "motion_parameters"
        )

        # ======================================
        # 6A. SINGLE-ROBOT SIMULATION
        # ======================================

        if NUM_ROBOTS == 1:

            print("\nRunning single-robot simulation...")

            rosbot_commands = generate_rosbot_commands(
                motion_parameters,
                OUTPUT_FOLDER / "rosbot_commands",
                margin=SAFETY_MARGIN,
                region_weights=REGION_WEIGHTS,
                target_radius=TARGET_RADIUS,
                seed=random_seed,
                travel_speed=TRAVEL_SPEED,
                max_travel_time=MAX_TRAVEL_TIME,
                dwell_min=DWELL_MIN,
                dwell_max=DWELL_MAX,
                distance_lambda=DISTANCE_LAMBDA,
                visit_penalty=VISIT_PENALTY,
                visit_memory_decay=VISIT_MEMORY_DECAY,
                max_consecutive=MAX_CONSECUTIVE_SELECTIONS,
            )

            rosbot_commands["random_seed"] = random_seed

            # Analyze spatial exploration
            analysis, analysis_file = analyze_region_occupancy(
                rosbot_commands,
                OUTPUT_FOLDER / "region_analysis"
            )

            # Generate trajectory visualization
            path_file = visualize_motion(
                rosbot_commands,
                OUTPUT_FOLDER / "visualizations"
            )

            # Generate animation
            video_file = animate_motion(
                rosbot_commands,
                audio_file,
                OUTPUT_FOLDER / "animations",
                preview_seconds=PREVIEW,
                fps=FPS,
            )

            # ----------------------------------
            # Single-robot results
            # ----------------------------------

            print("\nSINGLE-ROBOT SUMMARY")

            print(
                f"ROSbot commands: "
                f"{rosbot_commands['num_commands']}"
            )

            print(
                f"Target selections: "
                f"{len(rosbot_commands['targets'])}"
            )

            print("\nSPATIAL EXPLORATION SUMMARY")

            print(
                f"{'Region':<15} "
                f"{'Selected':>8} "
                f"{'Reached':>8} "
                f"{'Time (s)':>11} "
                f"{'Time %':>8}"
            )

            for region, stats in analysis["region_summary"].items():

                print(
                    f"{region:<15} "
                    f"{stats['selected_count']:>8} "
                    f"{stats['reached_count']:>8} "
                    f"{stats['time_spent_seconds']:>11.2f} "
                    f"{stats['time_spent_percent']:>7.1f}%"
                )

            print(f"\nRegion analysis: {analysis_file}")
            print(f"Path: {path_file}")
            print(f"MP4: {video_file}")

        # ======================================
        # 6B. MULTI-ROBOT SIMULATION
        # ======================================

        else:

            print(
                f"\nRunning {NUM_ROBOTS}-robot simulation..."
            )

            # Temporary command files are discarded
            # after rendering the visualization and MP4.
            with tempfile.TemporaryDirectory() as temp_dir:

                multi_commands, _ = generate_multi_robot_commands(
                    motion_parameters,
                    Path(temp_dir),
                    seed=random_seed,
                    num_robots=NUM_ROBOTS,
                    margin=SAFETY_MARGIN,
                    region_weights=REGION_WEIGHTS,
                    robot_radius=ROBOT_RADIUS,
                    clearance=EXTRA_CLEARANCE,
                    travel_speed=TRAVEL_SPEED,
                    target_radius=TARGET_RADIUS,
                    max_travel_time=MAX_TRAVEL_TIME,
                    dwell_min=DWELL_MIN,
                    dwell_max=DWELL_MAX,
                    distance_lambda=DISTANCE_LAMBDA,
                    visit_penalty=VISIT_PENALTY,
                    visit_memory_decay=VISIT_MEMORY_DECAY,
                    max_consecutive=MAX_CONSECUTIVE_SELECTIONS,
                    timestep=TIME_STEP,
                    max_yield_before_replan=MAX_YIELD_BEFORE_REPLAN,
                )

                # ----------------------------------
                # Visualize multi-robot trajectories
                # ----------------------------------

                path_file = plot_multi_robot(
                    multi_commands,
                    MULTI_OUTPUT_FOLDER / "visualizations"
                )

                # ----------------------------------
                # Generate multi-robot animation
                # ----------------------------------

                video_file = animate_multi_robot(
                    multi_commands,
                    audio_file,
                    MULTI_OUTPUT_FOLDER / "animations",
                    preview=PREVIEW,
                    fps=FPS,
                )

                # ----------------------------------
                # Multi-robot results
                # ----------------------------------

                print("\nMULTI-ROBOT SUMMARY")

                print(
                    f"Number of robots: {NUM_ROBOTS}"
                )

                print(
                    f"Required separation: "
                    f"{multi_commands['required_centre_separation']:.3f}m"
                )

                print(
                    f"Minimum simulated separation: "
                    f"{multi_commands['observed_minimum_separation']:.3f}m"
                )

                print(
                    f"Supervisor yield steps: "
                    f"{multi_commands['yield_steps']}"
                )

                for name, robot in multi_commands["robots"].items():

                    reached = sum(
                        target["outcome"] == "reached"
                        for target in robot["targets"]
                    )

                    print(
                        f"{name}: "
                        f"{len(robot['targets'])} targets, "
                        f"{reached} reached, "
                        f"{robot['yielded_seconds']:.2f}s yielding"
                    )

                print(f"\nPath: {path_file}")
                print(f"MP4: {video_file}")

        # ======================================
        # 7. COMMON RESULTS
        # ======================================

        print(f"\nGenre: {genre['predicted_genre']}")
        print(f"Random seed: {random_seed}")
        print(
            f"Selected accents: "
            f"{motion_timeline['num_strong_beats']}"
        )
        print(
            f"Motion events: "
            f"{motion_timeline['num_motion_events']}"
        )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
