import json


def action_to_parameters(action):
    mapping = {
        "slow_sway": {
            "speed": 0.2,
            "amplitude": 0.4,
            "duration": 4.0,
            "direction": "side_to_side",
            "turn_angle": 0,
            "curvature": 0.8,
            "motion_shape": "s_curve",
            "body_level": "medium",
            "repeat": 2
        },
        "flowing_arc": {
            "speed": 0.4,
            "amplitude": 0.7,
            "duration": 3.0,
            "direction": "arc",
            "turn_angle": 35,
            "curvature": 0.7,
            "motion_shape": "arc",
            "body_level": "medium",
            "repeat": 1
        },
        "large_flowing_arc": {
            "speed": 0.5,
            "amplitude": 1.2,
            "duration": 4.0,
            "direction": "large_arc",
            "turn_angle": 60,
            "curvature": 1.0,
            "motion_shape": "large_arc",
            "body_level": "medium_high",
            "repeat": 1
        },
        "strong_pulse": {
            "speed": 0.8,
            "amplitude": 1.0,
            "duration": 2.0,
            "direction": "forward_rebound",
            "turn_angle": 0,
            "curvature": 0.1,
            "motion_shape": "pulse",
            "body_level": "low",
            "repeat": 1
        },
        "strong_forward_pulse": {
            "speed": 1.0,
            "amplitude": 1.2,
            "duration": 2.0,
            "direction": "forward_rebound",
            "turn_angle": 0,
            "curvature": 0.1,
            "motion_shape": "strong_pulse",
            "body_level": "low",
            "repeat": 1
        },
        "expand_movement": {
            "speed": 0.6,
            "amplitude": 1.5,
            "duration": 3.0,
            "direction": "outward_spiral",
            "turn_angle": 45,
            "curvature": 0.9,
            "motion_shape": "spiral_out",
            "body_level": "high",
            "repeat": 1
        },
        "expand_slowly": {
            "speed": 0.3,
            "amplitude": 1.2,
            "duration": 4.0,
            "direction": "outward_arc",
            "turn_angle": 30,
            "curvature": 0.8,
            "motion_shape": "slow_expand",
            "body_level": "medium_high",
            "repeat": 1
        },
        "contract_movement": {
            "speed": 0.5,
            "amplitude": 0.8,
            "duration": 3.0,
            "direction": "inward_spiral",
            "turn_angle": -45,
            "curvature": 0.8,
            "motion_shape": "spiral_in",
            "body_level": "medium_low",
            "repeat": 1
        },
        "contract_slowly": {
            "speed": 0.25,
            "amplitude": 0.6,
            "duration": 4.0,
            "direction": "inward_arc",
            "turn_angle": -30,
            "curvature": 0.7,
            "motion_shape": "slow_contract",
            "body_level": "medium_low",
            "repeat": 1
        },
        "sharp_turn": {
            "speed": 1.0,
            "amplitude": 0.8,
            "duration": 1.5,
            "direction": "rotate",
            "turn_angle": 90,
            "curvature": 0.0,
            "motion_shape": "sharp_rotation",
            "body_level": "medium",
            "repeat": 1
        },
        "pause": {
            "speed": 0.0,
            "amplitude": 0.0,
            "duration": 2.0,
            "direction": "none",
            "turn_angle": 0,
            "curvature": 0.0,
            "motion_shape": "pause",
            "body_level": "medium",
            "repeat": 1
        },
        "end_pose": {
            "speed": 0.0,
            "amplitude": 0.0,
            "duration": 5.0,
            "direction": "none",
            "turn_angle": 0,
            "curvature": 0.0,
            "motion_shape": "end_pose",
            "body_level": "medium",
            "repeat": 1
        }
    }

    return mapping.get(action, {
        "speed": 0.5,
        "amplitude": 0.5,
        "duration": 2.0,
        "direction": "default",
        "turn_angle": 0,
        "curvature": 0.0,
        "motion_shape": "default",
        "body_level": "medium",
        "repeat": 1
    })


def generate_motion_parameters(motion_timeline, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)

    parameterized_events = []

    for motion in motion_timeline["motion_timeline"]:
        params = action_to_parameters(motion["action"])

        parameterized_event = {
            **motion,
            **params
        }

        parameterized_events.append(parameterized_event)

    output_data = {
        "song_name": motion_timeline["song_name"],
        "duration": motion_timeline["duration"],
        "predicted_genre": motion_timeline["predicted_genre"],
        "num_motion_events": len(parameterized_events),
        "motion_parameters": parameterized_events
    }

    song_stem = motion_timeline["song_name"].replace(".mp3", "")
    output_file = output_folder / f"{song_stem}_motion_parameters.json"

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    return output_data