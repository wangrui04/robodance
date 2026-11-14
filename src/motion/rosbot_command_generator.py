"""Music-driven ROSbot commands with history-aware 4x4 spatial navigation."""

import json
import math
import random
from collections import Counter, deque
from pathlib import Path

from motion.region_analysis import (
    REGION_CELLS,
    region_centres,
    region_for_position,
    sample_target,
    safe_bounds,
)

REGION_WEIGHTS = {
    "centre": 0.50,
    "top_left": 0.125,
    "top_right": 0.125,
    "bottom_left": 0.125,
    "bottom_right": 0.125,
}
SHARP_MOVEMENTS = ["forward_pulse", "backward_pulse", "sharp_left", "sharp_right"]
SMOOTH_MOVEMENTS = ["left_curve", "right_curve", "s_wave", "slow_sway"]


def clamp(value, low, high):
    return max(low, min(high, value))


def angle_difference(target, current):
    return (target - current + math.pi) % (2 * math.pi) - math.pi


def integrate(x, y, heading, v, w, dt):
    if abs(w) < 1e-9:
        return x + v * math.cos(heading) * dt, y + v * math.sin(heading) * dt, heading
    new_heading = heading + w * dt
    return (
        x + v / w * (math.sin(new_heading) - math.sin(heading)),
        y - v / w * (math.cos(new_heading) - math.cos(heading)),
        new_heading,
    )


def get_sharpness(event, energy_low, energy_high):
    source = event.get("source", {})
    energy = float(source.get("energy", 0.0))
    change = abs(float(source.get("energy_change", 0.0)))
    beat = float(event.get("beat_strength", 0.0))
    energy_norm = clamp((energy - energy_low) / max(energy_high - energy_low, 1e-9), 0, 1)
    sharpness = 0.55 * energy_norm + 0.25 * clamp(change / 0.08, 0, 1) + 0.20 * clamp(beat, 0, 1)
    quality = source.get("movement_quality", "")
    if quality == "strong":
        sharpness += 0.10
    elif quality in ("smooth", "flowing"):
        sharpness -= 0.08
    return clamp(sharpness, 0.10, 0.90)


def movement_profile(action, duration, speed):
    if action == "forward_pulse":
        return [(speed * 1.2, 0.0, duration)]
    if action == "backward_pulse":
        return [(-speed * 0.8, 0.0, duration)]
    if action == "sharp_left":
        return [(0.0, 1.6, duration * 0.30), (speed, 0.0, duration * 0.70)]
    if action == "sharp_right":
        return [(0.0, -1.6, duration * 0.30), (speed, 0.0, duration * 0.70)]
    if action == "left_curve":
        return [(speed, 0.65, duration)]
    if action == "right_curve":
        return [(speed, -0.65, duration)]
    if action == "s_wave":
        return [(speed, 0.8, duration * 0.25), (speed, -0.8, duration * 0.50), (speed, 0.8, duration * 0.25)]
    if action == "slow_sway":
        return [(speed * 0.65, 0.55, duration * 0.50), (speed * 0.65, -0.55, duration * 0.50)]
    return [(0.0, 0.0, duration)]


def safe_trajectory(pose, segments, boundary, margin, stay_in_region=None):
    x, y, heading = pose
    xmin, xmax, ymin, ymax = safe_bounds(boundary, margin)
    for v, w, duration in segments:
        steps = max(1, math.ceil(duration / 0.025))
        dt = duration / steps
        for _ in range(steps):
            x, y, heading = integrate(x, y, heading, v, w, dt)
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                return None
            if stay_in_region and region_for_position(x, y, boundary, margin) != stay_in_region:
                return None
    return x, y, heading


def calculate_region_probabilities(
    pose, boundary, margin, weights, visit_memory,
    last_selected, consecutive_selections, distance_lambda,
    visit_penalty, max_consecutive,
):
    centres = region_centres(boundary, margin)
    xmin, xmax, ymin, ymax = safe_bounds(boundary, margin)
    cell_size = ((xmax - xmin) / 4 + (ymax - ymin) / 4) / 2
    scores = {}
    for region, weight in weights.items():
        # Measure distance from the robot's physical pose, not a previous goal.
        cx, cy = centres[region]
        distance_in_cells = math.hypot(pose[0] - cx, pose[1] - cy) / cell_size
        score = weight * math.exp(-distance_lambda * distance_in_cells)
        score *= math.exp(-visit_penalty * visit_memory[region])
        if region == last_selected and consecutive_selections >= max_consecutive:
            score = 0.0
        scores[region] = score
    total = sum(scores.values())
    if total <= 0:
        raise ValueError("No eligible regions: check base weights and repetition constraint.")
    return {name: value / total for name, value in scores.items()}


def generate_rosbot_commands(
    motion_parameters,
    output_folder,
    margin=0.40,
    region_weights=None,
    target_radius=0.35,
    seed=42,
    travel_speed=0.32,
    max_travel_time=15.0,
    dwell_min=5.0,
    dwell_max=15.0,
    distance_lambda=0.35,
    visit_penalty=0.40,
    visit_memory_decay=0.85,
    max_consecutive=2,
):
    """Choose a region, travel to its target, then explore there for a variable duration.

    All phase transitions take place at eligible musical-event boundaries.
    Dwell begins after the exact point target is reached, not at selection time.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    boundary = motion_parameters["boundary"]
    events = sorted(motion_parameters["motion_parameters"], key=lambda event: event["time"])
    song_duration = float(motion_parameters["duration"])
    weights = dict(REGION_WEIGHTS if region_weights is None else region_weights)

    xmin, xmax, ymin, ymax = safe_bounds(boundary, margin)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Safety margin is too large for the stage.")
    if set(weights) != set(REGION_CELLS) or any(not math.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Provide nonnegative weights for all five regions, with a positive sum.")
    if target_radius <= 0 or travel_speed <= 0 or max_travel_time <= 0:
        raise ValueError("Target radius, travel speed, and maximum travel time must be positive.")
    if dwell_min <= 0 or dwell_max < dwell_min:
        raise ValueError("Require 0 < dwell_min <= dwell_max.")
    if distance_lambda < 0 or visit_penalty < 0 or not 0 <= visit_memory_decay <= 1 or max_consecutive < 1:
        raise ValueError("Invalid transition or history settings.")

    rng = random.Random(seed)
    energies = sorted(float(e.get("source", {}).get("energy", 0)) for e in events)
    if energies:
        low = energies[int((len(energies) - 1) * 0.10)]
        high = energies[int((len(energies) - 1) * 0.90)]
    else:
        low, high = 0.0, 1.0

    commands, targets = [], []
    pose = (0.0, 0.0, 0.0)
    current_time = 0.0
    recent_actions = deque(maxlen=3)
    usage = Counter()
    memory = {name: 0.0 for name in weights}
    previous_selected = None
    consecutive_selections = 0
    target = None
    stage = "choose"  # choose -> travel -> dwell -> choose
    dwell_end = None

    def append_command(start, end, action, v, w):
        nonlocal pose
        if end <= start:
            return
        commands.append({
            "time": round(start, 4),
            "start_time": round(start, 4),
            "end_time": round(end, 4),
            "duration": round(end - start, 4),
            "action": action,
            "linear_x": round(v, 4),
            "angular_z": round(w, 4),
        })
        pose = integrate(*pose, v, w, end - start)

    def choose_target(now):
        nonlocal target, previous_selected, consecutive_selections, stage
        probabilities = calculate_region_probabilities(
            pose, boundary, margin, weights, memory,
            previous_selected, consecutive_selections,
            distance_lambda, visit_penalty, max_consecutive,
        )
        region = rng.choices(list(probabilities), weights=list(probabilities.values()), k=1)[0]
        x, y, cell = sample_target(region, boundary, margin, rng)
        target = {
            "time": round(now, 4),
            "region": region,
            "x": round(x, 4),
            "y": round(y, 4),
            "cell": f"r{cell[0]+1}c{cell[1]+1}",
            "selection_probability": round(probabilities[region], 6),
            "probabilities": {name: round(p, 6) for name, p in probabilities.items()},
            "outcome": "pending",
            "arrival_time": None,
            "travel_time": None,
            "planned_dwell_seconds": None,
            "dwell_started_at": None,
            "dwell_ended_at": None,
        }
        targets.append(target)
        consecutive_selections = consecutive_selections + 1 if region == previous_selected else 1
        previous_selected = region
        for name in memory:
            memory[name] *= visit_memory_decay
        memory[region] += 1.0
        stage = "travel"

    def target_distance():
        return math.hypot(pose[0] - target["x"], pose[1] - target["y"])

    def arrived():
        return (target_distance() <= target_radius and
                region_for_position(pose[0], pose[1], boundary, margin) == target["region"])

    def begin_dwell(now):
        nonlocal dwell_end, stage
        target["outcome"] = "reached"
        target["arrival_time"] = round(now, 4)
        target["travel_time"] = round(now - target["time"], 4)
        target["planned_dwell_seconds"] = round(rng.uniform(dwell_min, dwell_max), 4)
        target["dwell_started_at"] = round(now, 4)
        dwell_end = now + target["planned_dwell_seconds"]
        stage = "dwell"

    for event in events:
        start = max(current_time, float(event["time"]))
        end = min(song_duration, float(event["end_time"]))
        if start > current_time:
            append_command(current_time, start, "STOP", 0.0, 0.0)
        if end <= start:
            current_time = max(current_time, start)
            continue

        # Decisions occur only at musical movement event boundaries.
        if stage == "travel":
            if arrived():
                begin_dwell(start)
            elif start - target["time"] >= max_travel_time:
                target["outcome"] = "timeout"
                target["travel_time"] = round(start - target["time"], 4)
                stage = "choose"
        if stage == "dwell" and start >= dwell_end:
            target["dwell_ended_at"] = round(start, 4)
            stage = "choose"
        if stage == "choose":
            choose_target(start)
            if arrived():
                begin_dwell(start)

        sharpness = get_sharpness(event, low, high)
        is_sharp = rng.random() < sharpness
        group = SHARP_MOVEMENTS if is_sharp else SMOOTH_MOVEMENTS
        other = SMOOTH_MOVEMENTS if is_sharp else SHARP_MOVEMENTS
        choices = [a for a in group if a not in recent_actions] or list(group)
        preferred = rng.choices(choices, weights=[1.0 / (1 + usage[a]) for a in choices], k=1)[0]
        duration = end - start

        if stage == "travel" and target_distance() > target_radius:
            angle = math.atan2(target["y"] - pose[1], target["x"] - pose[0])
            error = angle_difference(angle, pose[2])
            if abs(error) > 0.20:
                turn_speed = clamp(abs(error) * 1.5, 0.4, 1.6)
                turn_time = min(duration * 0.40, abs(error) / turn_speed)
                append_command(start, start + turn_time, "navigate_turn", 0.0, math.copysign(turn_speed, error))
                start += turn_time
                duration = end - start

        if duration <= 1e-8:
            current_time = end
            continue

        speed = travel_speed * (1.15 if is_sharp else 0.75)
        candidates = []
        ordered = [preferred] + [a for a in group if a != preferred] + list(other)
        # A directed forward movement is a reliable fallback for travelling.
        if stage == "travel":
            ordered.append("navigate_forward")

        for action in ordered:
            for scale in (1.0, 0.75, 0.5, 0.25):
                if action == "navigate_forward":
                    segments = [(travel_speed * scale, 0.0, duration)]
                else:
                    segments = movement_profile(action, duration, speed * scale)
                final_pose = safe_trajectory(
                    pose, segments, boundary, margin,
                    stay_in_region=target["region"] if stage == "dwell" else None,
                )
                if final_pose is None:
                    continue
                repetition = 1.5 if action in recent_actions else 0.0
                quality_bonus = 2.0 if action in group else 0.0
                chosen_bonus = 0.8 if action == preferred else 0.0
                score = quality_bonus + chosen_bonus + scale * 0.5 - repetition - usage[action] * 0.10
                if stage == "travel":
                    before = target_distance()
                    after = math.hypot(final_pose[0] - target["x"], final_pose[1] - target["y"])
                    score += (before - after) * 20.0
                    if action == "navigate_forward":
                        score -= 0.6  # Prefer expressive primitives when progress is similar.
                candidates.append((score, action, segments))

        if candidates:
            _, chosen, segments = max(candidates, key=lambda item: item[0])
        else:
            chosen, segments = "hold", [(0.0, 0.0, duration)]

        segment_start = start
        for v, w, segment_duration in segments:
            segment_end = segment_start + segment_duration
            append_command(segment_start, segment_end, chosen, v, w)
            segment_start = segment_end

        current_time = end
        if chosen != "hold":
            recent_actions.append(chosen)
            usage[chosen] += 1
        # Arrival can be detected at event end; dwell can start on the next event.
        if stage == "travel" and arrived():
            begin_dwell(end)

        print(
            f"{start:6.2f}s | Phase: {stage:<6} | Target: {target['region']:<12} | "
            f"Sharpness: {sharpness:.2f} | Motion: {chosen:<16} | "
            f"Pos: ({pose[0]:+.2f}, {pose[1]:+.2f})"
        )

    if current_time < song_duration:
        append_command(current_time, song_duration, "STOP", 0.0, 0.0)
    if target is not None:
        if target["outcome"] == "pending":
            target["outcome"] = "unfinished"
            target["travel_time"] = round(song_duration - target["time"], 4)
        if target["outcome"] == "reached" and target["dwell_ended_at"] is None:
            target["dwell_ended_at"] = round(song_duration, 4)

    data = {
        "song_name": motion_parameters["song_name"],
        "duration": song_duration,
        "predicted_genre": motion_parameters["predicted_genre"],
        "boundary": boundary,
        "safety_margin": margin,
        "region_weights": weights,
        "region_anchors": {name: {"x": x, "y": y} for name, (x, y) in region_centres(boundary, margin).items()},
        "targets": targets,
        "num_commands": len(commands),
        "commands": commands,
    }
    output_file = output_folder / f"{Path(data['song_name']).stem}_rosbot_commands.json"
    output_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
