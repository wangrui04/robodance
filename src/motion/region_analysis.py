"""4x4 spatial grid, five logical regions, and observed occupancy analysis."""

import json
import math
from pathlib import Path

REGION_CELLS = {
    "top_left": [(0, 0), (0, 1), (1, 0)],
    "top_right": [(0, 2), (0, 3), (1, 3)],
    "centre": [(1, 1), (1, 2), (2, 1), (2, 2)],
    "bottom_left": [(2, 0), (3, 0), (3, 1)],
    "bottom_right": [(2, 3), (3, 2), (3, 3)],
}
CELL_REGION = {cell: region for region, cells in REGION_CELLS.items() for cell in cells}


def safe_bounds(boundary, margin):
    return (
        boundary["x_min"] + margin,
        boundary["x_max"] - margin,
        boundary["y_min"] + margin,
        boundary["y_max"] - margin,
    )


def cell_for_position(x, y, boundary, margin):
    """Return (row, col), with row 0 at the top; edge points are clamped."""
    xmin, xmax, ymin, ymax = safe_bounds(boundary, margin)
    col = min(3, max(0, int(4 * (x - xmin) / (xmax - xmin))))
    row = min(3, max(0, int(4 * (ymax - y) / (ymax - ymin))))
    return row, col


def region_for_position(x, y, boundary, margin):
    return CELL_REGION[cell_for_position(x, y, boundary, margin)]


def cell_bounds(cell, boundary, margin):
    row, col = cell
    xmin, xmax, ymin, ymax = safe_bounds(boundary, margin)
    width = (xmax - xmin) / 4
    height = (ymax - ymin) / 4
    x0 = xmin + col * width
    y1 = ymax - row * height
    return x0, x0 + width, y1 - height, y1


def region_centres(boundary, margin):
    centres = {}
    for region, cells in REGION_CELLS.items():
        points = []
        for cell in cells:
            x0, x1, y0, y1 = cell_bounds(cell, boundary, margin)
            points.append(((x0 + x1) / 2, (y0 + y1) / 2))
        centres[region] = (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        )
    return centres


def sample_target(region, boundary, margin, rng):
    """Uniformly select one of a region's equally sized cells, then a safe interior point."""
    cell = rng.choice(REGION_CELLS[region])
    x0, x1, y0, y1 = cell_bounds(cell, boundary, margin)
    inset = min(0.12, (x1 - x0) * 0.2, (y1 - y0) * 0.2)
    return rng.uniform(x0 + inset, x1 - inset), rng.uniform(y0 + inset, y1 - inset), cell


def analyze_region_occupancy(commands_data, output_folder, step=0.025):
    """Integrate actual commands; dwell accounting includes turning and STOP periods."""
    # Local import avoids circular dependency with the command generator.
    from motion.rosbot_command_generator import integrate

    if step <= 0:
        raise ValueError("Occupancy sampling step must be positive.")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    boundary = commands_data["boundary"]
    margin = commands_data["safety_margin"]
    duration = float(commands_data["duration"])
    region_time = {name: 0.0 for name in REGION_CELLS}
    cell_time = {f"r{r + 1}c{c + 1}": 0.0 for r in range(4) for c in range(4)}
    entries = {name: 0 for name in REGION_CELLS}
    transitions = []
    x = y = heading = 0.0
    previous_region = None
    previous_end = 0.0

    def observe(start, end, v, w):
        nonlocal x, y, heading, previous_region
        if end <= start:
            return
        steps = max(1, math.ceil((end - start) / step))
        dt = (end - start) / steps
        for i in range(steps):
            # The midpoint approximates occupancy during the timestep.
            mx, my, _ = integrate(x, y, heading, v, w, dt / 2)
            cell = cell_for_position(mx, my, boundary, margin)
            region = CELL_REGION[cell]
            region_time[region] += dt
            cell_time[f"r{cell[0] + 1}c{cell[1] + 1}"] += dt
            if region != previous_region:
                entries[region] += 1
                transitions.append({
                    "time": round(start + i * dt, 4),
                    "from": previous_region,
                    "to": region,
                })
                previous_region = region
            x, y, heading = integrate(x, y, heading, v, w, dt)

    for command in sorted(commands_data["commands"], key=lambda c: c["start_time"]):
        start = float(command["start_time"])
        end = float(command["end_time"])
        if start > previous_end:
            observe(previous_end, start, 0.0, 0.0)
        observe(start, end, float(command["linear_x"]), float(command["angular_z"]))
        previous_end = end
    if previous_end < duration:
        observe(previous_end, duration, 0.0, 0.0)

    target_history = commands_data["targets"]
    summary = {}
    for name in REGION_CELLS:
        selected = [t for t in target_history if t["region"] == name]
        summary[name] = {
            "selected_count": len(selected),
            "reached_count": sum(t["outcome"] == "reached" for t in selected),
            "timeout_count": sum(t["outcome"] == "timeout" for t in selected),
            "physical_entries": entries[name],
            "time_spent_seconds": round(region_time[name], 3),
            "time_spent_percent": round(100 * region_time[name] / duration, 2) if duration else 0.0,
        }

    result = {
        "song_name": commands_data["song_name"],
        "duration": duration,
        "note": "Occupancy is estimated from commanded 2D motion at 0.025-second intervals, not measured hardware telemetry.",
        "grid_rows": 4,
        "grid_cols": 4,
        "region_cells": {name: [f"r{r+1}c{c+1}" for r, c in cells] for name, cells in REGION_CELLS.items()},
        "region_summary": summary,
        "cell_occupancy_seconds": {k: round(v, 3) for k, v in cell_time.items()},
        "target_history": target_history,
        "region_transitions": transitions,
    }
    path = output_folder / f"{Path(result['song_name']).stem}_region_analysis.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, path
