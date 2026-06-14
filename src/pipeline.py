from pathlib import Path

from feature_extractor import analyze_song
from genre_classifier import classify_genre, save_genre_result
from music_state_extractor import generate_music_state
from motion_planner import generate_motion_timeline
from motion_parameter_generator import generate_motion_parameters
from motion_visualizer import visualize_motion
from motion_animator import animate_motion


project_root = Path(__file__).resolve().parent.parent

music_folder = project_root / "music"
output_folder = project_root / "output"

feature_folder = output_folder / "features"
genre_folder = output_folder / "genres"
music_state_folder = output_folder / "music_states"
motion_timeline_folder = output_folder / "motion_timelines"
motion_parameter_folder = output_folder / "motion_parameters"
visualization_folder = output_folder / "visualizations"
animation_folder = output_folder / "animations"

for folder in [
    feature_folder,
    genre_folder,
    music_state_folder,
    motion_timeline_folder,
    motion_parameter_folder,
    visualization_folder,
    animation_folder,
]:
    folder.mkdir(parents=True, exist_ok=True)


for audio_file in sorted(music_folder.glob("*.mp3")):
    print(f"\nProcessing {audio_file.name}")

    features = analyze_song(audio_file, feature_folder)

    genre = classify_genre(features)
    save_genre_result(genre, genre_folder)

    music_state = generate_music_state(audio_file, music_state_folder)

    motion_timeline = generate_motion_timeline(
        music_state,
        genre,
        motion_timeline_folder
    )
    motion_parameters = generate_motion_parameters(
        motion_timeline,
        motion_parameter_folder
    )
    animation_file = animate_motion(
        motion_parameters,
        animation_folder
    )
    visualize_motion(
        motion_parameters,
        visualization_folder
    )

    print(f"Genre: {genre['predicted_genre']}")
    print(f"Motion events: {motion_timeline['num_motion_events']}")
    print(f"Parameterized events: {motion_parameters['num_motion_events']}")
    print(f"Animation: {animation_file}")

print("\nPipeline complete.")