
"""Music-responsive, coordinated N-robot choreography in a 2D simulation.

Reuses the single-robot five-region model, music sharpness model,
and unicycle integration.

Simulation only: not a certified controller for physical robots.
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


# ==========================================
# BASIC HELPERS
# ==========================================

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(value, low, high):
    return max(low, min(high, value))


def initial_poses(count, boundary, margin, radius, separation):
    """Place robots on a ring, facing the centre."""

    xmin, xmax, ymin, ymax = safe_bounds(
        boundary, margin + radius
    )

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    ring_radius = 0.76 * min(
        (xmax - xmin) / 2,
        (ymax - ymin) / 2
    )

    if count < 2 or ring_radius <= 0:
        raise ValueError(
            "Multi-robot simulation requires at least two "
            "robots and a nonempty stage."
        )

    if 2 * ring_radius * math.sin(math.pi / count) < separation:
        raise ValueError(
            f"Cannot fit {count} robots with "
            f"{separation:.2f}m required separation."
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
    """Each robot keeps an independent improvisation state."""

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

        # Independent musical phrasing
        "event_sharp": False,
        "event_turn": 1,
        "next_phrase_change": index * 0.65,
        "rest_until": 0.0,
        "rest_count": 0,

        # Collision avoidance statistics
        "avoid_steps": 0,
        "yielded_seconds": 0.0,
        "blocked_seconds": 0.0,
        "target_replans": 0,
    }


# ==========================================
# DYNAMIC REGION SELECTION
# ==========================================

def choose_target(robot, robots, now, boundary, settings):

    probabilities = calculate_region_probabilities(
        robot["pose"],
        boundary,
        settings["margin"],
        settings["weights"],
        robot["memory"],
        robot["last_region"],
        robot["consecutive"],
        settings["distance_lambda"],
        settings["visit_penalty"],
        settings["max_consecutive"],
    )

    name = robot["rng"].choices(
        list(probabilities),
        weights=list(probabilities.values()),
        k=1
    )[0]

    # Prefer target points away from other robots.
    xmin, xmax, ymin, ymax = safe_bounds(
        boundary,
        settings["margin"] + settings["robot_radius"]
    )

    candidates = []

    for _ in range(80):
        candidate = sample_target(
            name,
            boundary,
            settings["margin"],
            robot["rng"]
        )

        x, y, cell = candidate

        if xmin <= x <= xmax and ymin <= y <= ymax:
            candidates.append(candidate)

    if not candidates:
        raise ValueError(
            "No footprint-safe target in selected region."
        )

    x, y, cell = max(
        candidates,
        key=lambda p: min(
            distance(p, other["pose"])
            for other in robots
            if other is not robot
        ),
    )

    target = {
        "time": round(now, 4),
        "region": name,
        "x": round(x, 4),
        "y": round(y, 4),
        "cell": f"r{cell[0]+1}c{cell[1]+1}",
        "selection_probability": round(probabilities[name], 6),
        "probabilities": {
            k: round(v, 6)
            for k, v in probabilities.items()
        },
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
        robot["consecutive"] + 1
        if robot["last_region"] == name
        else 1
    )

    robot["last_region"] = name

    for region in robot["memory"]:
        robot["memory"][region] *= settings["visit_memory_decay"]

    robot["memory"][name] += 1.0

    robot["phase"] = "travel"
    robot["blocked_seconds"] = 0.0


# ==========================================
# ARRIVAL AND DWELL
# ==========================================

def target_reached(robot, boundary, settings):

    target = robot["target"]

    return bool(
        target
        and distance(
            robot["pose"],
            (target["x"], target["y"])
        ) <= settings["target_radius"]
        and region_for_position(
            *robot["pose"][:2],
            boundary,
            settings["margin"]
        ) == target["region"]
    )


def start_dwell(robot, now, settings):

    target = robot["target"]

    seconds = robot["rng"].uniform(
        settings["dwell_min"],
        settings["dwell_max"]
    )

    target.update(
        outcome="reached",
        arrival_time=round(now, 4),
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
            target["travel_time"] = round(
                now - target["time"], 4
            )

            robot["phase"] = "choose"

    if robot["phase"] == "dwell":

        if now >= robot["dwell_until"]:
            target["dwell_ended_at"] = round(now, 4)
            robot["phase"] = "choose"

    if robot["phase"] == "choose":

        choose_target(
            robot,
            robots,
            now,
            boundary,
            settings
        )

        if target_reached(robot, boundary, settings):
            start_dwell(robot, now, settings)


# ==========================================
# INDEPENDENT MUSIC-RESPONSIVE IMPROVISATION
# ==========================================

def desired_velocity(
    robot,
    event,
    now,
    energy_low,
    energy_high,
    settings
):
    """
    Each robot independently chooses:
    - Phrase duration
    - Sharp or smooth movement
    - Turning direction
    - Whether and when to pause

    Gaps between selected musical accents do not
    automatically stop every robot.
    """

    rng = robot["rng"]

    # --------------------------------------
    # Select a new independent phrase
    # --------------------------------------

    if now >= robot["next_phrase_change"]:

        sharpness = (
            get_sharpness(event, energy_low, energy_high)
            if event
            else 0.35
        )

        robot["event_sharp"] = (
            rng.random() < sharpness
        )

        robot["event_turn"] = rng.choice((-1, 1))

        robot["next_phrase_change"] = (
            now + rng.uniform(1.4, 3.8)
        )

        # Independent optional expressive rest
        if rng.random() < settings["rest_probability"]:

            robot["rest_until"] = (
                now + rng.uniform(0.35, 1.2)
            )

            robot["rest_count"] += 1

    if now < robot["rest_until"]:
        return (0.0, 0.0, "expressive_rest")

    sharp = robot["event_sharp"]

    pose = robot["pose"]
    target = robot["target"]

    speed = settings["travel_speed"] * (
        1.10 if sharp else 0.75
    )

    angle = math.atan2(
        target["y"] - pose[1],
        target["x"] - pose[0]
    )

    error = angle_difference(
        angle,
        pose[2]
    )

    # --------------------------------------
    # TRAVEL PHASE
    # --------------------------------------

    if robot["phase"] == "travel":

        if abs(error) > 0.38:

            return (
                0.0,
                clamp(error * 2.0, -1.4, 1.4),
                "navigate_turn"
            )

        sway = 0.22 * math.sin(
            2.0 * now + robot["event_turn"]
        )

        return (
            speed * max(0.35, 1.0 - abs(error)),
            clamp(error * 1.8 + sway, -0.9, 0.9),
            "sharp_travel" if sharp else "smooth_travel",
        )

    # --------------------------------------
    # DWELL PHASE
    # --------------------------------------

    if distance(
        pose,
        (target["x"], target["y"])
    ) > 0.42:

        if abs(error) > 0.55:

            return (
                0.0,
                clamp(error * 1.6, -1.1, 1.1),
                "dwell_reorient"
            )

        return (
            speed * 0.42,
            clamp(error * 1.4, -1.1, 1.1),
            "dwell_return"
        )

    if sharp:

        return (
            speed * 0.30,
            robot["event_turn"] * 0.90,
            "sharp_arc"
        )

    return (
        speed * 0.42,
        robot["event_turn"] * 0.60 * math.sin(now * 1.5),
        "smooth_sway",
    )


# ==========================================
# STAGE AND REGION SAFETY
# ==========================================

def movement_safe(
    robot,
    velocity,
    angular,
    dt,
    boundary,
    settings
):
    """Check intermediate positions along the simulated arc."""

    pose = robot["pose"]

    xmin, xmax, ymin, ymax = safe_bounds(
        boundary,
        settings["margin"] + settings["robot_radius"]
    )

    for step in range(1, 9):

        candidate = integrate(
            *pose,
            velocity,
            angular,
            dt * step / 8
        )

        if not (
            xmin <= candidate[0] <= xmax
            and ymin <= candidate[1] <= ymax
        ):
            return False

        if (
            robot["phase"] == "dwell"
            and region_for_position(
                candidate[0],
                candidate[1],
                boundary,
                settings["margin"]
            ) != robot["target"]["region"]
        ):
            return False

    return True


# ==========================================
# PAIRWISE COLLISION CHECKING
# ==========================================

def pair_safe(
    pose_a,
    control_a,
    pose_b,
    control_b,
    dt,
    separation
):
    """
    Check simulated separation throughout the movement step.

    Adds a conservative speed-based allowance between samples
    under the idealized unicycle model.
    """

    v_a, w_a, _ = control_a
    v_b, w_b, _ = control_b

    samples = 16

    allowance = (
        (abs(v_a) + abs(v_b)) * dt / (2 * samples)
    )

    for k in range(samples + 1):

        t = dt * k / samples

        a = integrate(
            *pose_a,
            v_a,
            w_a,
            t
        )

        b = integrate(
            *pose_b,
            v_b,
            w_b,
            t
        )

        if distance(a, b) < separation + allowance - 1e-9:
            return False

    return True


def all_pairs_safe(robots, controls, dt, separation):

    for i in range(len(robots)):

        for j in range(i + 1, len(robots)):

            if not pair_safe(
                robots[i]["pose"],
                controls[i],
                robots[j]["pose"],
                controls[j],
                dt,
                separation
            ):
                return False

    return True


# ==========================================
# PREDICTIVE COLLISION AVOIDANCE
# ==========================================

def projected_min_distance(
    pose_a,
    control_a,
    pose_b,
    control_b,
    horizon
):
    """Estimate separation over the prediction horizon."""

    va, wa, _ = control_a
    vb, wb, _ = control_b

    minimum = float("inf")

    for index in range(13):

        t = horizon * index / 12

        a = integrate(
            *pose_a,
            va,
            wa,
            t
        )

        b = integrate(
            *pose_b,
            vb,
            wb,
            t
        )

        minimum = min(
            minimum,
            distance(a, b)
        )

    return minimum


def avoidance_candidates(request, turn_preference):
    """Generate steering and speed alternatives before stopping."""

    v, w, label = request

    choices = [request]

    # Try curved movement at several speeds.
    for scale in (1.0, 0.7, 0.4):

        for offset in (0.6, 1.2, -0.6, -1.2):

            choices.append((
                v * scale,
                clamp(
                    w + turn_preference * offset,
                    -1.5,
                    1.5
                ),
                "avoid_steer"
            ))

    if abs(v) > 1e-6:

        choices.extend([
            (
                v * 0.3,
                w,
                "avoid_slow"
            ),
            (
                0.0,
                turn_preference * 1.0,
                "avoid_turn"
            ),
            (
                0.0,
                -turn_preference * 1.0,
                "avoid_turn"
            ),
        ])

    # Last resort
    choices.append(
        (0.0, 0.0, "yield_stop")
    )

    return list(dict.fromkeys(choices))


def supervise(robots, requests, now, dt, settings):
    """
    Predict nearby conflicts and prefer steering around them.

    If no safe movement is found, retain a stopping fallback.
    """

    count = len(robots)
    accepted = list(requests)

    separation = settings["separation"]
    horizon = settings["avoid_horizon"]
    buffer = settings["avoid_buffer"]

    # Rotate priority every two seconds.
    first = int(now / 2.0) % count

    order = [
        (first + i) % count
        for i in range(count)
    ]

    for idx in order:

        robot = robots[idx]
        requested = requests[idx]

        processed = order[:order.index(idx)]

        others = [
            (
                j,
                accepted[j] if j in processed else requests[j]
            )
            for j in range(count)
            if j != idx
        ]

        nearest = min(
            projected_min_distance(
                robot["pose"],
                requested,
                robots[j]["pose"],
                other,
                horizon
            )
            for j, other in others
        )

        # Keep the original movement when sufficiently clear.
        if (
            nearest >= separation + buffer
            and all(
                pair_safe(
                    robot["pose"],
                    requested,
                    robots[j]["pose"],
                    other,
                    dt,
                    separation
                )
                for j, other in others
            )
        ):
            accepted[idx] = requested
            continue

        # --------------------------------------
        # Choose an avoidance direction
        # --------------------------------------

        closest_j = min(
            (
                j for j in range(count)
                if j != idx
            ),
            key=lambda j: distance(
                robot["pose"],
                robots[j]["pose"]
            )
        )

        other_pose = robots[closest_j]["pose"]

        bearing = angle_difference(
            math.atan2(
                other_pose[1] - robot["pose"][1],
                other_pose[0] - robot["pose"][0]
            ),
            robot["pose"][2]
        )

        preference = -1 if bearing >= 0 else 1

        # Break symmetric reactions.
        if abs(bearing) < 0.03:
            preference = 1 if idx % 2 == 0 else -1

        candidates = avoidance_candidates(
            requested,
            preference
        )

        scored = []

        # --------------------------------------
        # Evaluate alternative movements
        # --------------------------------------

        for control in candidates:

            v, w, label = control

            if not movement_safe(
                robot,
                v,
                w,
                dt,
                settings["boundary"],
                settings
            ):
                continue

            if any(
                not pair_safe(
                    robot["pose"],
                    control,
                    robots[j]["pose"],
                    other,
                    dt,
                    separation
                )
                for j, other in others
            ):
                continue

            clearance = min(
                projected_min_distance(
                    robot["pose"],
                    control,
                    robots[j]["pose"],
                    other,
                    horizon
                )
                for j, other in others
            )

            # Prefer clearance without sacrificing too much
            # of the original musical movement.
            maneuver_cost = (
                1.6 * abs(v - requested[0])
                + 0.15 * abs(w - requested[1])
            )

            stopped = (
                0.8 if label == "yield_stop" else 0
            )

            score = (
                4.0 * min(clearance - separation, 0.6)
                - maneuver_cost
                - stopped
            )

            if clearance < separation + 0.04:
                score -= 3.0

            scored.append((score, control))

        if scored:

            accepted[idx] = max(
                scored,
                key=lambda item: item[0]
            )[1]

        else:

            accepted[idx] = (
                0.0,
                0.0,
                "yield_stop"
            )

    # ======================================
    # FINAL IMMEDIATE SAFETY CHECK
    # ======================================

    for _ in range(count + 1):

        if all_pairs_safe(
            robots,
            accepted,
            dt,
            separation
        ):
            return accepted

        # Suppress the lowest-priority active movement.
        for i in reversed(order):

            if accepted[i] != (
                0.0,
                0.0,
                "yield_stop"
            ):

                accepted[i] = (
                    0.0,
                    0.0,
                    "yield_stop"
                )

                break

    # Final fallback: everyone stops.
    stopped = [
        (0.0, 0.0, "yield_stop")
        for _ in robots
    ]

    if not all_pairs_safe(
        robots,
        stopped,
        dt,
        separation
    ):
        raise RuntimeError(
            "Robot footprints already violate separation."
        )

    return stopped


# ==========================================
# RECORD VELOCITY COMMANDS
# ==========================================

def add_command(
    robot,
    start,
    end,
    velocity,
    angular,
    action
):

    v = round(velocity, 4)
    w = round(angular, 4)

    commands = robot["commands"]

    if (
        commands
        and commands[-1]["action"] == action
        and commands[-1]["linear_x"] == v
        and commands[-1]["angular_z"] == w
        and abs(commands[-1]["end_time"] - start) < 1e-5
    ):

        commands[-1]["end_time"] = round(end, 4)

        commands[-1]["duration"] = round(
            end - commands[-1]["start_time"],
            4
        )

    else:

        commands.append({
            "time": round(start, 4),
            "start_time": round(start, 4),
            "end_time": round(end, 4),
            "duration": round(end - start, 4),
            "action": action,
            "linear_x": v,
            "angular_z": w,
        })


# ==========================================
# MAIN MULTI-ROBOT PLANNER
# ==========================================

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
    max_yield_before_replan=1.5,
    rest_probability=0.12,
    avoid_horizon=1.4,
    avoid_buffer=0.25,
):
    """
    Generate an independent but collision-coordinated performance
    for N >= 2 robots.

    Uses a shared musical timeline and independent random states.

    Returns:
        (data, output_json_path)
    """

    # ======================================
    # VALIDATION
    # ======================================

    if (
        not isinstance(num_robots, int)
        or isinstance(num_robots, bool)
        or num_robots < 2
    ):
        raise ValueError(
            "Use num_robots >= 2."
        )

    weights = dict(
        DEFAULT_WEIGHTS
        if region_weights is None
        else region_weights
    )

    if (
        set(weights) != set(REGION_CELLS)
        or any(
            not math.isfinite(w) or w < 0
            for w in weights.values()
        )
        or sum(weights.values()) <= 0
    ):
        raise ValueError(
            "Specify five nonnegative region weights."
        )

    if not (
        0 < timestep <= 0.05
        and margin >= 0
        and robot_radius > 0
        and clearance >= 0
        and travel_speed > 0
        and target_radius > 0
        and max_travel_time > 0
        and 0 < dwell_min <= dwell_max
        and distance_lambda >= 0
        and visit_penalty >= 0
        and 0 <= visit_memory_decay <= 1
        and max_consecutive >= 1
        and max_yield_before_replan > 0
        and 0 <= rest_probability <= 1
        and avoid_horizon > 0
        and avoid_buffer >= 0
    ):
        raise ValueError(
            "Invalid multi-robot movement parameters."
        )

    # ======================================
    # INITIALIZATION
    # ======================================

    boundary = motion_parameters["boundary"]
    duration = float(motion_parameters["duration"])

    events = sorted(
        motion_parameters["motion_parameters"],
        key=lambda e: e["time"]
    )

    separation = 2 * robot_radius + clearance

    starts = initial_poses(
        num_robots,
        boundary,
        margin,
        robot_radius,
        separation
    )

    settings = {
        "margin": margin,
        "robot_radius": robot_radius,
        "separation": separation,
        "travel_speed": travel_speed,
        "target_radius": target_radius,
        "max_travel_time": max_travel_time,
        "dwell_min": dwell_min,
        "dwell_max": dwell_max,
        "distance_lambda": distance_lambda,
        "visit_penalty": visit_penalty,
        "visit_memory_decay": visit_memory_decay,
        "max_consecutive": max_consecutive,
        "max_yield_before_replan": max_yield_before_replan,
        "weights": weights,
        "rest_probability": rest_probability,
        "avoid_horizon": avoid_horizon,
        "avoid_buffer": avoid_buffer,
        "boundary": boundary,
    }

    # Create independent random generators for each robot.
    master = random.Random(seed)

    robots = [
        make_robot(
            i,
            pose,
            random.Random(master.getrandbits(64)),
            weights
        )
        for i, pose in enumerate(starts)
    ]

    # Determine this song's energy range.
    energies = sorted(
        float(
            e.get("source", {}).get("energy", 0)
        )
        for e in events
    )

    low = (
        energies[int(0.10 * (len(energies) - 1))]
        if energies else 0.0
    )

    high = (
        energies[int(0.90 * (len(energies) - 1))]
        if energies else 1.0
    )

    observed_min = min(
        distance(starts[i], starts[j])
        for i in range(num_robots)
        for j in range(i + 1, num_robots)
    )

    if observed_min < separation:
        raise ValueError(
            "Starting positions overlap."
        )

    yield_steps = 0
    event_index = 0
    steps = math.ceil(duration / timestep)

    # ======================================
    # SHARED SIMULATION CLOCK
    # ======================================

    for step in range(steps):

        now = min(step * timestep, duration)
        end = min((step + 1) * timestep, duration)
        dt = end - now

        if dt <= 0:
            continue

        while (
            event_index + 1 < len(events)
            and events[event_index + 1]["time"] <= now
        ):
            event_index += 1

        # Keep the most recent musical state.
        # Robots can still stop independently.
        event = (
            events[event_index]
            if events
            and float(events[event_index]["time"]) <= now
            else None
        )

        # ----------------------------------
        # 1. Update destination phases
        # ----------------------------------

        for robot in robots:

            update_phase(
                robot,
                robots,
                now,
                boundary,
                settings
            )

        # ----------------------------------
        # 2. Independent movement proposals
        # ----------------------------------

        proposals = [
            desired_velocity(
                robot,
                event,
                now,
                low,
                high,
                settings
            )
            for robot in robots
        ]

        # ----------------------------------
        # 3. Boundary checks
        # ----------------------------------

        for i, (robot, proposed) in enumerate(
            zip(robots, proposals)
        ):

            if not movement_safe(
                robot,
                proposed[0],
                proposed[1],
                dt,
                boundary,
                settings
            ):

                target = robot["target"]

                angle = math.atan2(
                    target["y"] - robot["pose"][1],
                    target["x"] - robot["pose"][0]
                )

                error = angle_difference(
                    angle,
                    robot["pose"][2]
                )

                proposals[i] = (
                    0.0,
                    clamp(
                        error * 1.5
                        if abs(error) > 0.1
                        else 0.6,
                        -1.1,
                        1.1
                    ),
                    "boundary_or_region_turn"
                )

                if not movement_safe(
                    robot,
                    proposals[i][0],
                    proposals[i][1],
                    dt,
                    boundary,
                    settings
                ):

                    proposals[i] = (
                        0.0,
                        0.0,
                        "boundary_stop"
                    )

        # ----------------------------------
        # 4. Predictive collision avoidance
        # ----------------------------------

        accepted = supervise(
            robots,
            proposals,
            now,
            dt,
            settings
        )

        if not all_pairs_safe(
            robots,
            accepted,
            dt,
            separation
        ):
            raise RuntimeError(
                f"Unsafe pair at t={now:.3f}s"
            )

        if any(
            action == "yield_stop"
            for _, _, action in accepted
        ):
            yield_steps += 1

        # ----------------------------------
        # 5. Execute accepted movements
        # ----------------------------------

        for robot, control in zip(robots, accepted):

            v, w, label = control

            if not movement_safe(
                robot,
                v,
                w,
                dt,
                boundary,
                settings
            ):
                raise RuntimeError(
                    f"Unsafe movement at t={now:.3f}s"
                )

            new_pose = integrate(
                *robot["pose"],
                v,
                w,
                dt
            )

            add_command(
                robot,
                now,
                end,
                v,
                w,
                label
            )

            robot["pose"] = new_pose

            if label == "yield_stop":

                robot["yielded_seconds"] += dt
                robot["blocked_seconds"] += dt

            elif label.startswith("avoid_"):

                robot["avoid_steps"] += 1
                robot["blocked_seconds"] = 0.0

            else:

                robot["blocked_seconds"] = 0.0

            # Replan when blocked too long.
            if (
                robot["phase"] == "travel"
                and robot["blocked_seconds"]
                >= max_yield_before_replan
            ):

                robot["target"]["outcome"] = "blocked"

                robot["target"]["travel_time"] = round(
                    end - robot["target"]["time"],
                    4
                )

                robot["phase"] = "choose"
                robot["blocked_seconds"] = 0.0
                robot["target_replans"] += 1

        # ----------------------------------
        # 6. Record minimum separation
        # ----------------------------------

        observed_min = min(
            observed_min,
            *(
                distance(
                    robots[i]["pose"],
                    robots[j]["pose"]
                )
                for i in range(num_robots)
                for j in range(i + 1, num_robots)
            )
        )

    # ======================================
    # FINALIZE TARGET HISTORIES
    # ======================================

    for robot in robots:

        target = robot["target"]

        if target is not None:

            if target["outcome"] == "pending":

                target["outcome"] = "unfinished"

                target["travel_time"] = round(
                    duration - target["time"],
                    4
                )

            elif (
                target["outcome"] == "reached"
                and target["dwell_ended_at"] is None
            ):

                target["dwell_ended_at"] = round(
                    duration,
                    4
                )

    # ======================================
    # BUILD OUTPUT
    # ======================================

    data = {
        "song_name": motion_parameters["song_name"],
        "predicted_genre": motion_parameters.get(
            "predicted_genre",
            "unknown"
        ),
        "duration": duration,
        "boundary": boundary,
        "safety_margin": margin,
        "num_robots": num_robots,
        "random_seed": seed,
        "robot_radius": robot_radius,
        "additional_clearance": clearance,
        "required_centre_separation": separation,
        "observed_minimum_separation": round(
            observed_min, 4
        ),
        "collision_check_timestep": timestep,
        "yield_steps": yield_steps,
        "note": (
            "Simulation-only collision supervision; "
            "not a real-world safety guarantee."
        ),
        "robots": {
            robot["name"]: {
                "start_pose": [
                    round(v, 6)
                    for v in robot["start_pose"]
                ],
                "targets": robot["targets"],
                "commands": robot["commands"],
                "num_commands": len(robot["commands"]),
                "yielded_seconds": round(
                    robot["yielded_seconds"], 3
                ),
                "target_replans": robot["target_replans"],
                "independent_rests": robot["rest_count"],
                "avoidance_steps": robot["avoid_steps"],
            }
            for robot in robots
        },
    }

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    path = (
        output_folder
        / f"{Path(data['song_name']).stem}_{num_robots}robots_commands.json"
    )

    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )

    return data, path


# ==========================================
# BACKWARD COMPATIBILITY
# ==========================================

def generate_duet(
    motion_parameters,
    output_folder,
    seed=None,
    **kwargs
):
    return generate_multi_robot_commands(
        motion_parameters,
        output_folder,
        seed=seed,
        num_robots=2,
        **kwargs
    )
