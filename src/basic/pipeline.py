
import argparse
from pathlib import Path

from beat_motion import (
    detect_beats,
    generate_beat_commands,
    simulate_beat_motion,
    save_commands
)

from motion_video import generate_motion_video
from rosbot_simulator import simulate_rosbot


project_root = Path(__file__).resolve().parents[2]

music_folder = project_root / "music"
output_folder = project_root / "output" / "basic"


def main():

    parser = argparse.ArgumentParser(
        description="Beat-synchronized ROSbot movement"
    )

    parser.add_argument(
        "--beats-per-move",
        type=int,
        default=2,
        help="Number of beats to move before stopping"
    )

    parser.add_argument(
        "--pause-beats",
        type=int,
        default=1,
        help="Number of beats to stop between movements"
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=0.12,
        help="Robot linear speed in meters/second"
    )

    parser.add_argument(
        "--start-direction",
        choices=["forward", "backward"],
        default="forward"
    )

    parser.add_argument(
        "--video",
        action="store_true",
        help="Generate an MP4 movement animation with music"
    )

    parser.add_argument(
        "--rosbot",
        action="store_true",
        help="Generate a top-down ROSbot simulation with music"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Video frame rate"
    )

    parser.add_argument(
        "--preview",
        type=float,
        default=30,
        help="Preview duration in seconds (0 for full song)"
    )

    args = parser.parse_args()

    output_folder.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(music_folder.glob("*.mp3"))

    if not audio_files:
        print("No MP3 files found.")
        return

    for audio_file in audio_files:

        print(f"\nProcessing {audio_file.name}")

        # Step 1: Detect beats
        beat_data = detect_beats(audio_file)

        # Step 2: Generate movement commands
        commands_data = generate_beat_commands(
            beat_data,
            beats_per_move=args.beats_per_move,
            pause_beats=args.pause_beats,
            speed=args.speed,
            start_direction=args.start_direction
        )

        # Step 3: Save commands to JSON
        command_file = save_commands(
            commands_data,
            output_folder
        )

        # Step 4: Generate position-versus-time graph
        simulation_file = simulate_beat_motion(
            commands_data,
            output_folder
        )

        print(f"Tempo: {beat_data['tempo']:.2f} BPM")
        print(f"Detected beats: {len(beat_data['beat_times'])}")
        print(f"Beats per movement: {args.beats_per_move}")
        print(f"Pause beats: {args.pause_beats}")
        print(f"Movement commands: {commands_data['num_commands']}")
        print(f"Commands saved: {command_file}")
        print(f"Simulation saved: {simulation_file}")

        # Step 5: Generate basic movement animation
        if args.video:

            print("Generating movement video with music...")

            video_file = generate_motion_video(
                commands_data,
                audio_file,
                output_folder,
                fps=args.fps,
                preview_seconds=(
                    args.preview if args.preview > 0 else None
                )
            )

            print(f"Video saved: {video_file}")

        # Step 6: Generate top-down ROSbot simulation
        if args.rosbot:

            print("Generating ROSbot simulation with music...")

            rosbot_file = simulate_rosbot(
                commands_data,
                audio_file,
                output_folder,
                fps=args.fps,
                preview_seconds=(
                    args.preview if args.preview > 0 else None
                )
            )

            print(f"ROSbot simulation saved: {rosbot_file}")

    print("\nBasic beat movement pipeline complete.")


if __name__ == "__main__":
    main()
