
import json
import math
import random
from collections import Counter, deque
from pathlib import Path


REGION_WEIGHTS = {
    "centre": 0.50,
    "top_left": 0.125,
    "top_right": 0.125,
    "bottom_left": 0.125,
    "bottom_right": 0.125,
}

SHARP_MOVEMENTS = [
    "forward_pulse",
    "backward_pulse",
    "sharp_left",
    "sharp_right",
]

SMOOTH_MOVEMENTS = [
    "left_curve",
    "right_curve",
    "s_wave",
    "slow_sway",
]


def clamp(value, low, high):
    return max(low, min(high, value))


def angle_difference(target, current):
    return (target - current + math.pi) % (2 * math.pi) - math.pi


def integrate(x, y, heading, v, w, dt):
    if abs(w) < 1e-9:
        return (
            x + v * math.cos(heading) * dt,
            y + v * math.sin(heading) * dt,
            heading,
        )

    new_heading = heading + w * dt

    return (
        x + v / w * (
            math.sin(new_heading) - math.sin(heading)
        ),
        y - v / w * (
            math.cos(new_heading) - math.cos(heading)
        ),
        new_heading,
    )


def get_regions(boundary, margin):
    xmin = boundary["x_min"] + margin
    xmax = boundary["x_max"] - margin
    ymin = boundary["y_min"] + margin
    ymax = boundary["y_max"] - margin

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    # Targets remain inside the safe area.
    dx = (xmax - xmin) * 0.34
    dy = (ymax - ymin) * 0.34

    return {
        "centre": (cx, cy),
        "top_left": (cx - dx, cy + dy),
        "top_right": (cx + dx, cy + dy),
        "bottom_left": (cx - dx, cy - dy),
        "bottom_right": (cx + dx, cy - dy),
    }


def get_sharpness(event, energy_low, energy_high):
    source = event.get("source", {})

    energy = float(source.get("energy", 0.0))
    change = abs(float(source.get("energy_change", 0.0)))
    beat = float(event.get("beat_strength", 0.0))

    # Normalize energy relative to this song.
    energy_norm = clamp(
        (energy - energy_low)
        / max(energy_high - energy_low, 1e-9),
        0.0,
        1.0,
    )

    # Changes and accents can create sharp moments
    # even during otherwise calm passages.
    change_score = clamp(change / 0.08, 0.0, 1.0)
    beat_score = clamp(beat, 0.0, 1.0)

    sharpness = (
        0.55 * energy_norm
        + 0.25 * change_score
        + 0.20 * beat_score
    )

    quality = source.get("movement_quality", "")

    if quality == "strong":
        sharpness += 0.10
    elif quality in ("smooth", "flowing"):
        sharpness -= 0.08

    return clamp(sharpness, 0.10, 0.90)


def movement_profile(action, duration, speed):
    # Each tuple is (linear speed, angular speed, duration).
    # All segments together fill one movement event.

    if action == "forward_pulse":
        return [(speed * 1.2, 0.0, duration)]

    if action == "backward_pulse":
        return [(-speed * 0.8, 0.0, duration)]

    if action == "sharp_left":
        return [
            (0.0, 1.6, duration * 0.30),
            (speed, 0.0, duration * 0.70),
        ]

    if action == "sharp_right":
        return [
            (0.0, -1.6, duration * 0.30),
            (speed, 0.0, duration * 0.70),
        ]

    if action == "left_curve":
        return [(speed, 0.65, duration)]

    if action == "right_curve":
        return [(speed, -0.65, duration)]

    if action == "s_wave":
        return [
            (speed, 0.8, duration * 0.25),
            (speed, -0.8, duration * 0.50),
            (speed, 0.8, duration * 0.25),
        ]

    if action == "slow_sway":
        return [
            (speed * 0.65, 0.55, duration * 0.50),
            (speed * 0.65, -0.55, duration * 0.50),
        ]

    return [(0.0, 0.0, duration)]


def safe_trajectory(pose, segments, boundary, margin):
    x, y, heading = pose

    for v, w, duration in segments:
        steps = max(1, math.ceil(duration / 0.025))
        dt = duration / steps

        for _ in range(steps):
            x, y, heading = integrate(
                x, y, heading, v, w, dt
            )

            if not (
                boundary["x_min"] + margin <= x
                <= boundary["x_max"] - margin
                and boundary["y_min"] + margin <= y
                <= boundary["y_max"] - margin
            ):
                return None

    return (x, y, heading)


def generate_rosbot_commands(
    motion_parameters,
    output_folder,
    margin=0.40,
    region_weights=None,
    target_interval=7.0,
    target_radius=0.35,
    seed=42,
    travel_speed=0.32,
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    boundary = motion_parameters["boundary"]
    events = motion_parameters["motion_parameters"]
    song_duration = float(motion_parameters["duration"])

    if margin < 0:
        raise ValueError("Safety margin must be nonnegative.")

    if target_interval <= 0 or target_radius <= 0:
        raise ValueError("Target timing and radius must be positive.")

    if travel_speed <= 0:
        raise ValueError("Travel speed must be positive.")

    if (
        boundary["x_max"] - boundary["x_min"] <= 2 * margin
        or boundary["y_max"] - boundary["y_min"] <= 2 * margin
    ):
        raise ValueError("Safety margin is too large.")

    weights = (
        dict(REGION_WEIGHTS)
        if region_weights is None
        else dict(region_weights)
    )

    regions = get_regions(boundary, margin)

    if (
        set(weights) != set(regions)
        or any(value < 0 for value in weights.values())
        or sum(weights.values()) <= 0
    ):
        raise ValueError("Invalid region weights.")

    rng = random.Random(seed)

    # Use this song's own energy distribution.
    energies = sorted(
        float(e.get("source", {}).get("energy", 0.0))
        for e in events
    )

    if energies:
        energy_low = energies[max(0, int(len(energies) * 0.10))]
        energy_high = energies[
            min(len(energies) - 1, int(len(energies) * 0.90))
        ]
    else:
        energy_low, energy_high = 0.0, 1.0

    commands = []
    targets = []

    pose = (0.0, 0.0, 0.0)
    current_time = 0.0

    recent_actions = deque(maxlen=3)
    usage = Counter()

    target_name = None
    target_point = None
    next_target_time = 0.0

    def append_command(start, end, action, v, w):
        nonlocal pose

        if end <= start:
            return

        duration = end - start

        commands.append({
            "time": round(start, 4),
            "start_time": round(start, 4),
            "end_time": round(end, 4),
            "duration": round(duration, 4),
            "action": action,
            "linear_x": round(v, 4),
            "angular_z": round(w, 4),
        })

        pose = integrate(*pose, v, w, duration)

    def select_target(time):
        nonlocal target_name, target_point, next_target_time

        names = list(regions)

        # Exact configured weights for each new selection.
        target_name = rng.choices(
            names,
            weights=[weights[name] for name in names],
            k=1,
        )[0]

        anchor = regions[target_name]

        # Small variation within each region.
        jitter = 0.12

        xmin = boundary["x_min"] + margin
        xmax = boundary["x_max"] - margin
        ymin = boundary["y_min"] + margin
        ymax = boundary["y_max"] - margin

        target_point = (
            clamp(anchor[0] + rng.uniform(-jitter, jitter), xmin, xmax),
            clamp(anchor[1] + rng.uniform(-jitter, jitter), ymin, ymax),
        )

        next_target_time = time + target_interval

        targets.append({
            "time": round(time, 4),
            "region": target_name,
            "x": round(target_point[0], 4),
            "y": round(target_point[1], 4),
        })

    def distance_to_target(point):
        return math.hypot(
            point[0] - target_point[0],
            point[1] - target_point[1],
        )

    for event in events:
        start = max(current_time, float(event["time"]))
        end = min(song_duration, float(event["end_time"]))

        if start > current_time:
            append_command(
                current_time, start, "STOP", 0.0, 0.0
            )

        if end <= start:
            current_time = max(current_time, start)
            continue

        if target_point is None or start >= next_target_time:
            select_target(start)

        duration = end - start
        distance = distance_to_target(pose)

        sharpness = get_sharpness(
            event, energy_low, energy_high
        )

        # The music determines the probability of a
        # sharp or smooth movement.
        if rng.random() < sharpness:
            preferred_group = SHARP_MOVEMENTS
            quality = "SHARP"
        else:
            preferred_group = SMOOTH_MOVEMENTS
            quality = "SMOOTH"

        # Select among primitives, discouraging repetition.
        choices = [
            action for action in preferred_group
            if action not in recent_actions
        ]

        if not choices:
            choices = list(preferred_group)

        chosen = rng.choices(
            choices,
            weights=[
                1.0 / (1.0 + usage[action])
                for action in choices
            ],
            k=1,
        )[0]

        # ------------------------------------------------
        # NAVIGATION: turn toward the selected destination
        # ------------------------------------------------

        # Only navigate strongly when not already near target.
        if distance > target_radius:
            desired_heading = math.atan2(
                target_point[1] - pose[1],
                target_point[0] - pose[0],
            )

            error = angle_difference(
                desired_heading, pose[2]
            )

            # A separate heading correction prevents the
            # robot from drifting toward one side forever.
            if abs(error) > 0.25:
                turn_speed = clamp(
                    abs(error) * 1.5, 0.4, 1.6
                )

                turn_time = min(
                    duration * 0.40,
                    abs(error) / turn_speed,
                )

                signed_turn = math.copysign(
                    turn_speed, error
                )

                append_command(
                    start,
                    start + turn_time,
                    "navigate_turn",
                    0.0,
                    signed_turn,
                )

                start += turn_time
                duration = end - start

        if duration <= 1e-6:
            current_time = end
            continue

        # ------------------------------------------------
        # MUSIC: execute expressive movement
        # ------------------------------------------------

        # Choose a speed appropriate to the musical quality.
        if quality == "SHARP":
            speed = travel_speed * 1.15
        else:
            speed = travel_speed * 0.75

        candidates = []

        # Consider the chosen primitive first, then other
        # movements of the same musical quality.
        ordered_actions = [chosen] + [
            action for action in preferred_group
            if action != chosen
        ]

        # Include the other style as a safety fallback.
        other_group = (
            SMOOTH_MOVEMENTS
            if quality == "SHARP"
            else SHARP_MOVEMENTS
        )

        ordered_actions += other_group

        for action in ordered_actions:
            for scale in (1.0, 0.75, 0.50, 0.25):

                segments = movement_profile(
                    action, duration, speed * scale
                )

                final_pose = safe_trajectory(
                    pose, segments, boundary, margin
                )

                if final_pose is None:
                    continue

                progress = (
                    distance_to_target(pose)
                    - distance_to_target(final_pose)
                )

                # Target progress matters, but movement
                # quality still matters too.
                quality_bonus = (
                    2.0 if action in preferred_group else 0.0
                )

                chosen_bonus = (
                    0.8 if action == chosen else 0.0
                )

                repetition_penalty = (
                    1.5 if action in recent_actions else 0.0
                )

                score = (
                    progress * 12.0
                    + quality_bonus
                    + chosen_bonus
                    + scale * 0.5
                    - repetition_penalty
                    - usage[action] * 0.10
                )

                # When near the target, prioritize
                # musical expression over travel.
                if distance < target_radius:
                    score -= progress * 10.0

                candidates.append(
                    (score, action, segments)
                )

        if candidates:
            _, chosen, segments = max(
                candidates,
                key=lambda item: item[0]
            )
        else:
            chosen = "hold"
            segments = [(0.0, 0.0, duration)]

        segment_start = start

        for v, w, segment_duration in segments:
            segment_end = segment_start + segment_duration

            append_command(
                segment_start,
                segment_end,
                chosen,
                v,
                w,
            )

            segment_start = segment_end

        current_time = end

        if chosen != "hold":
            recent_actions.append(chosen)
            usage[chosen] += 1

        print(
            f"{start:6.2f}s | "
            f"Target: {target_name:<12} | "
            f"Sharpness: {sharpness:.2f} | "
            f"Quality: {quality:<6} | "
            f"Motion: {chosen:<16} | "
            f"Pos: ({pose[0]:+.2f}, {pose[1]:+.2f})"
        )

    if current_time < song_duration:
        append_command(
            current_time,
            song_duration,
            "STOP",
            0.0,
            0.0,
        )

    data = {
        "song_name": motion_parameters["song_name"],
        "duration": song_duration,
        "predicted_genre": motion_parameters["predicted_genre"],
        "boundary": boundary,
        "safety_margin": margin,
        "region_weights": weights,
        "region_anchors": {
            name: {"x": p[0], "y": p[1]}
            for name, p in regions.items()
        },
        "targets": targets,
        "num_commands": len(commands),
        "commands": commands,
    }

    output_file = (
        output_folder
        / f"{Path(data['song_name']).stem}_rosbot_commands.json"
    )

    output_file.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    return data
