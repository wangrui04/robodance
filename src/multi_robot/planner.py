"""Music-responsive, coordinated N-robot choreography in a 2D simulation.

This module reuses the single-robot project's five-region model, musical
sharpness model, and unicycle integration. It is NOT a certified controller
for physical robots.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from motion.region_analysis import (
    REGION_CELLS,
    region_for_position,
    safe_bounds,
    sample_target,
)
from motion.rosbot_command_generator import (
    angle_difference,
    calculate_region_probabilities,
    get_sharpness,
    integrate,
)

DEFAULT_WEIGHTS = {
    "centre": 0.50,
    "top_left": 0.125,
    "top_right": 0.125,
    "bottom_left": 0.125,
    "bottom_right": 0.125,
}


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(value, low, high):
    return max(low, min(high, value))


def initial_poses(count, boundary, margin, radius, separation):
    """Place robots on a ring, facing the centre. Reject crowded stages."""
    xmin, xmax, ymin, ymax = safe_bounds(boundary, margin + radius)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    ring_radius = 0.76 * min((xmax - xmin) / 2, (ymax - ymin) / 2)
    if count < 2 or ring_radius <= 0:
        raise ValueError("Multi-robot simulation requires at least two robots and a nonempty stage.")
    if 2 * ring_radius * math.sin(math.pi / count) < separation:
        raise ValueError(
            f"Cannot fit {count} robots on the stage with {separation:.2f}m "
            "required separation. Reduce the count or adjust the stage dimensions."
        )
    poses = []
    for index in range(count):
        angle = math.pi + index * 2 * math.pi / count
        x = cx + ring_radius * math.cos(angle)
        y = cy + ring_radius * math.sin(angle)
        heading = math.atan2(cy - y, cx - x)
        poses.append((x, y, heading))
    return poses


def make_robot(index, start, rng, weights):
    return {
        "name": f"robot_{index + 1}",
        "pose": start,
        "start_pose": start,
        "rng": rng,
        "phase": "choose",
        "target": None,
        "targets": [],
        "commands": [],
        "memory": {region: 0.0 for region in weights},
        "last_region": None,
        "consecutive": 0,
        "dwell_until": 0.0,
        "event_id": None,
        "event_sharp": False,
        "event_turn": 1,
        "yielded_seconds": 0.0,
        "blocked_seconds": 0.0,
        "target_replans": 0,
    }


def choose_target(robot, robots, now, boundary, settings):
    probabilities = calculate_region_probabilities(
        robot["pose"], boundary, settings["margin"], settings["weights"],
        robot["memory"], robot["last_region"], robot["consecutive"],
        settings["distance_lambda"], settings["visit_penalty"],
        settings["max_consecutive"],
    )
    name = robot["rng"].choices(
        list(probabilities), weights=list(probabilities.values()), k=1
    )[0]
    # Prefer targets that are not presently occupied by another robot.
    # Collision supervision is still applied at every simulation step.
    xmin, xmax, ymin, ymax = safe_bounds(
        boundary, settings["margin"] + settings["robot_radius"]
    )
    candidates = []
    for _ in range(80):
        candidate = sample_target(name, boundary, settings["margin"], robot["rng"])
        x0, y0, _ = candidate
        if xmin <= x0 <= xmax and ymin <= y0 <= ymax:
            candidates.append(candidate)
    if not candidates:
        raise ValueError("No footprint-safe target point available in selected region.")
    x, y, cell = max(
        candidates,
        key=lambda p: min(
            distance(p, other["pose"])
            for other in robots if other is not robot
        ),
    )
    target = {
        "time": round(now, 4),
        "region": name,
        "x": round(x, 4),
        "y": round(y, 4),
        "cell": f"r{cell[0]+1}c{cell[1]+1}",
        "selection_probability": round(probabilities[name], 6),
        "probabilities": {k: round(v, 6) for k, v in probabilities.items()},
        "outcome": "pending",
        "arrival_time": None,
        "travel_time": None,
        "planned_dwell_seconds": None,
        "dwell_started_at": None,
        "dwell_ended_at": None,
    }
    robot["targets"].append(target)
    robot["target"] = target
    robot["consecutive"] = (
        robot["consecutive"] + 1 if robot["last_region"] == name else 1
    )
    robot["last_region"] = name
    for region in robot["memory"]:
        robot["memory"][region] *= settings["visit_memory_decay"]
    robot["memory"][name] += 1.0
    robot["phase"] = "travel"
    robot["blocked_seconds"] = 0.0


def target_reached(robot, boundary, settings):
    t = robot["target"]
    return bool(
        t and distance(robot["pose"], (t["x"], t["y"])) <= settings["target_radius"]
        and region_for_position(*robot["pose"][:2], boundary, settings["margin"]) == t["region"]
    )


def start_dwell(robot, now, settings):
    target = robot["target"]
    seconds = robot["rng"].uniform(settings["dwell_min"], settings["dwell_max"])
    target.update(
        outcome="reached", arrival_time=round(now, 4),
        travel_time=round(now - target["time"], 4),
        planned_dwell_seconds=round(seconds, 4),
        dwell_started_at=round(now, 4),
    )
    robot["dwell_until"] = now + seconds
    robot["phase"] = "dwell"


def update_phase(robot, robots, now, boundary, settings):
    target = robot["target"]
    if robot["phase"] == "travel" and target:
        if target_reached(robot, boundary, settings):
            start_dwell(robot, now, settings)
        elif now - target["time"] >= settings["max_travel_time"]:
            target["outcome"] = "timeout"
            target["travel_time"] = round(now - target["time"], 4)
            robot["phase"] = "choose"
    if robot["phase"] == "dwell" and now >= robot["dwell_until"]:
        target["dwell_ended_at"] = round(now, 4)
        robot["phase"] = "choose"
    if robot["phase"] == "choose":
        choose_target(robot, robots, now, boundary, settings)
        if target_reached(robot, boundary, settings):
            start_dwell(robot, now, settings)


def desired_velocity(robot, event, now, energy_low, energy_high, settings):
    if event is None:
        return (0.0, 0.0, "music_pause")
    if robot["event_id"] != float(event["time"]):
        robot["event_id"] = float(event["time"])
        robot["event_sharp"] = robot["rng"].random() < get_sharpness(
            event, energy_low, energy_high
        )
        robot["event_turn"] = robot["rng"].choice((-1, 1))

    sharp = robot["event_sharp"]
    pose = robot["pose"]
    target = robot["target"]
    speed = settings["travel_speed"] * (1.10 if sharp else 0.75)
    angle = math.atan2(target["y"] - pose[1], target["x"] - pose[0])
    error = angle_difference(angle, pose[2])

    if robot["phase"] == "travel":
        if abs(error) > 0.38:
            return (0.0, clamp(error * 2.0, -1.4, 1.4), "navigate_turn")
        sway = 0.22 * math.sin(2.0 * now + robot["event_turn"])
        return (
            speed * max(0.35, 1.0 - abs(error)),
            clamp(error * 1.8 + sway, -0.9, 0.9),
            "sharp_travel" if sharp else "smooth_travel",
        )

    if distance(pose, (target["x"], target["y"])) > 0.42:
        if abs(error) > 0.55:
            return (0.0, clamp(error * 1.6, -1.1, 1.1), "dwell_reorient")
        return (speed * 0.42, clamp(error * 1.4, -1.1, 1.1), "dwell_return")
    if sharp:
        return (speed * 0.30, robot["event_turn"] * 0.90, "sharp_arc")
    return (
        speed * 0.42,
        robot["event_turn"] * 0.60 * math.sin(now * 1.5),
        "smooth_sway",
    )


def movement_safe(robot, velocity, angular, dt, boundary, settings):
    """Validate complete (sampled) arc against stage and dwell region."""
    pose = robot["pose"]
    xmin, xmax, ymin, ymax = safe_bounds(
        boundary, settings["margin"] + settings["robot_radius"]
    )
    # Include intermediate poses, rather than checking only end positions.
    for step in range(1, 9):
        candidate = integrate(*pose, velocity, angular, dt * step / 8)
        if not (xmin <= candidate[0] <= xmax and ymin <= candidate[1] <= ymax):
            return False
        if robot["phase"] == "dwell" and region_for_position(
            candidate[0], candidate[1], boundary, settings["margin"]
        ) != robot["target"]["region"]:
            return False
    return True


def pair_safe(pose_a, control_a, pose_b, control_b, dt, separation):
    """Conservative separation bound using speed-limited temporal samples.

    For samples spaced dt/16, robot centre distance can decrease by at most
    (|v_a|+|v_b|)*dt/32 between a sample and its nearest sample. Thus passing
    this inequality certifies clearance under our unicycle model over the step.
    """
    v_a, w_a, _ = control_a
    v_b, w_b, _ = control_b
    samples = 16
    allowance = (abs(v_a) + abs(v_b)) * dt / (2 * samples)
    for k in range(samples + 1):
        t = dt * k / samples
        a = integrate(*pose_a, v_a, w_a, t)
        b = integrate(*pose_b, v_b, w_b, t)
        if distance(a, b) < separation + allowance - 1e-9:
            return False
    return True


def all_pairs_safe(robots, controls, dt, separation):
    for i in range(len(robots)):
        for j in range(i + 1, len(robots)):
            if not pair_safe(
                robots[i]["pose"], controls[i],
                robots[j]["pose"], controls[j], dt, separation,
            ):
                return False
    return True


def supervise(robots, requests, now, dt, settings):
    """Rotate priority and yield lower-priority robots until all pairs are safe.

    Stopping in place is safe only when robots begin the step separated.
    A sustained yield triggers a different destination on a later step.
    """
    accepted = list(requests)
    count = len(robots)
    # Rotate priority every two seconds, not every small timestep.
    first = int(now / 2.0) % count
    rank = {idx: (idx - first) % count for idx in range(count)}
    separation = settings["separation"]

    for _ in range(count + 1):
        conflicts = []
        for i in range(count):
            for j in range(i + 1, count):
                if not pair_safe(
                    robots[i]["pose"], accepted[i],
                    robots[j]["pose"], accepted[j], dt, separation
                ):
                    conflicts.append((i, j))
        if not conflicts:
            return accepted
        changed = False
        for i, j in conflicts:
            lower = i if rank[i] > rank[j] else j
            if accepted[lower] != (0.0, 0.0, "yield_stop"):
                accepted[lower] = (0.0, 0.0, "yield_stop")
                changed = True
        if not changed:
            break

    # Final fallback: stop everyone rather than accept an unsafe movement.
    stopped = [(0.0, 0.0, "yield_stop") for _ in robots]
    if not all_pairs_safe(robots, stopped, dt, separation):
        raise RuntimeError("Starting poses violate clearance; cannot resolve safely.")
    return stopped


def add_command(robot, start, end, velocity, angular, action):
    v, w = round(velocity, 4), round(angular, 4)
    cmds = robot["commands"]
    if (
        cmds and cmds[-1]["action"] == action
        and cmds[-1]["linear_x"] == v and cmds[-1]["angular_z"] == w
        and abs(cmds[-1]["end_time"] - start) < 1e-5
    ):
        cmds[-1]["end_time"] = round(end, 4)
        cmds[-1]["duration"] = round(end - cmds[-1]["start_time"], 4)
    else:
        cmds.append({
            "time": round(start, 4), "start_time": round(start, 4),
            "end_time": round(end, 4), "duration": round(end - start, 4),
            "action": action, "linear_x": v, "angular_z": w,
        })


def generate_multi_robot_commands(
    motion_parameters,
    output_folder,
    seed=None,
    num_robots=2,
    *,
    margin=0.40,
    region_weights=None,
    robot_radius=0.18,
    clearance=0.18,
    travel_speed=0.32,
    target_radius=0.35,
    max_travel_time=15.0,
    dwell_min=5.0,
    dwell_max=15.0,
    distance_lambda=0.35,
    visit_penalty=0.40,
    visit_memory_decay=0.85,
    max_consecutive=2,
    timestep=0.05,
    max_yield_before_replan=3.0,
):
    """Generate a simulated performance for N >= 2 robots.

    Returns (data, JSON_path). Caller should use the existing solo generator
    when num_robots == 1. All robots use the same music timeline, but have
    independently sampled movement, targets and dwell durations.
    """
    if not isinstance(num_robots, int) or isinstance(num_robots, bool) or num_robots < 2:
        raise ValueError("Use num_robots >= 2; use the solo planner for one robot.")
    weights = dict(DEFAULT_WEIGHTS if region_weights is None else region_weights)
    if (
        set(weights) != set(REGION_CELLS)
        or any(not math.isfinite(w) or w < 0 for w in weights.values())
        or sum(weights.values()) <= 0
    ):
        raise ValueError("Specify five nonnegative region weights with positive total.")
    if not (
        0 < timestep <= 0.05 and margin >= 0 and robot_radius > 0
        and clearance >= 0 and travel_speed > 0 and target_radius > 0
        and max_travel_time > 0 and 0 < dwell_min <= dwell_max
        and distance_lambda >= 0 and visit_penalty >= 0
        and 0 <= visit_memory_decay <= 1 and max_consecutive >= 1
        and max_yield_before_replan > 0
    ):
        raise ValueError("Invalid multi-robot movement parameters.")

    boundary = motion_parameters["boundary"]
    duration = float(motion_parameters["duration"])
    events = sorted(motion_parameters["motion_parameters"], key=lambda e: e["time"])
    separation = robot_radius * 2 + clearance
    starts = initial_poses(num_robots, boundary, margin, robot_radius, separation)
    settings = dict(
        margin=margin, robot_radius=robot_radius, separation=separation,
        travel_speed=travel_speed, target_radius=target_radius,
        max_travel_time=max_travel_time, dwell_min=dwell_min, dwell_max=dwell_max,
        distance_lambda=distance_lambda, visit_penalty=visit_penalty,
        visit_memory_decay=visit_memory_decay, max_consecutive=max_consecutive,
        max_yield_before_replan=max_yield_before_replan, weights=weights,
    )

    master = random.Random(seed)
    robots = [
        make_robot(i, pose, random.Random(master.getrandbits(64)), weights)
        for i, pose in enumerate(starts)
    ]
    energies = sorted(float(e.get("source", {}).get("energy", 0)) for e in events)
    low = energies[int(0.10 * (len(energies) - 1))] if energies else 0.0
    high = energies[int(0.90 * (len(energies) - 1))] if energies else 1.0

    observed_min = min(
        distance(starts[i], starts[j])
        for i in range(num_robots) for j in range(i + 1, num_robots)
    )
    if observed_min < separation:
        raise ValueError("Starting positions overlap.")
    yield_steps = 0
    event_index = 0
    steps = math.ceil(duration / timestep)

    for step in range(steps):
        now = min(step * timestep, duration)
        end = min((step + 1) * timestep, duration)
        dt = end - now
        if dt <= 0:
            continue
        while event_index + 1 < len(events) and events[event_index + 1]["time"] <= now:
            event_index += 1
        event = (
            events[event_index] if events and
            float(events[event_index]["time"]) <= now < float(events[event_index]["end_time"])
            else None
        )

        for robot in robots:
            update_phase(robot, robots, now, boundary, settings)

        proposals = [
            desired_velocity(robot, event, now, low, high, settings)
            for robot in robots
        ]
        for i, (robot, proposed) in enumerate(zip(robots, proposals)):
            if not movement_safe(robot, proposed[0], proposed[1], dt, boundary, settings):
                # Rotate toward target in place where possible; otherwise stop.
                target = robot["target"]
                angle = math.atan2(target["y"] - robot["pose"][1], target["x"] - robot["pose"][0])
                error = angle_difference(angle, robot["pose"][2])
                proposals[i] = (0.0, clamp(error * 1.5 if abs(error) > .1 else .6, -1.1, 1.1), "boundary_or_region_turn")
                if not movement_safe(robot, proposals[i][0], proposals[i][1], dt, boundary, settings):
                    proposals[i] = (0.0, 0.0, "boundary_stop")

        accepted = supervise(robots, proposals, now, dt, settings)
        if not all_pairs_safe(robots, accepted, dt, separation):
            raise RuntimeError(f"Unsafe pair at t={now:.3f}s")
        if any(a[2] == "yield_stop" for a in accepted):
            yield_steps += 1

        for robot, control in zip(robots, accepted):
            v, w, label = control
            if not movement_safe(robot, v, w, dt, boundary, settings):
                raise RuntimeError(f"Unsafe stage/region movement at t={now:.3f}s")
            new_pose = integrate(*robot["pose"], v, w, dt)
            add_command(robot, now, end, v, w, label)
            robot["pose"] = new_pose
            if label == "yield_stop":
                robot["yielded_seconds"] += dt
                robot["blocked_seconds"] += dt
            else:
                robot["blocked_seconds"] = 0.0
            if (
                robot["phase"] == "travel"
                and robot["blocked_seconds"] >= max_yield_before_replan
            ):
                robot["target"]["outcome"] = "blocked"
                robot["target"]["travel_time"] = round(end - robot["target"]["time"], 4)
                robot["phase"] = "choose"
                robot["blocked_seconds"] = 0.0
                robot["target_replans"] += 1

        observed_min = min(
            observed_min,
            *(distance(robots[i]["pose"], robots[j]["pose"])
              for i in range(num_robots) for j in range(i + 1, num_robots))
        )

    for robot in robots:
        target = robot["target"]
        if target is not None:
            if target["outcome"] == "pending":
                target["outcome"] = "unfinished"
                target["travel_time"] = round(duration - target["time"], 4)
            elif target["outcome"] == "reached" and target["dwell_ended_at"] is None:
                target["dwell_ended_at"] = round(duration, 4)

    data = {
        "song_name": motion_parameters["song_name"],
        "predicted_genre": motion_parameters.get("predicted_genre", "unknown"),
        "duration": duration,
        "boundary": boundary,
        "safety_margin": margin,
        "num_robots": num_robots,
        "random_seed": seed,
        "robot_radius": robot_radius,
        "additional_clearance": clearance,
        "required_centre_separation": separation,
        "observed_minimum_separation": round(observed_min, 4),
        "collision_check_timestep": timestep,
        "yield_steps": yield_steps,
        "note": "Simulation-only geometry and collision supervision; not a real-world safety guarantee.",
        "robots": {
            robot["name"]: {
                "start_pose": [round(v, 6) for v in robot["start_pose"]],
                "targets": robot["targets"],
                "commands": robot["commands"],
                "num_commands": len(robot["commands"]),
                "yielded_seconds": round(robot["yielded_seconds"], 3),
                "target_replans": robot["target_replans"],
            }
            for robot in robots
        },
    }
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    path = output_folder / f"{Path(data['song_name']).stem}_{num_robots}robots_commands.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data, path


# Backward-compatible name for the former two-robot entry point.
def generate_duet(motion_parameters, output_folder, seed=None, **kwargs):
    return generate_multi_robot_commands(
        motion_parameters, output_folder, seed=seed, num_robots=2, **kwargs
    )
