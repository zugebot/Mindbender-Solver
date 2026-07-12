from __future__ import annotations
# pyth/viewer/chuzzle_mouse_export.py

import json
from pathlib import Path

from chuzzle_mouse_domain import RouteProfile, route_profile_to_dict
from chuzzle_mouse_routes import RouteProfileManager


def export_profile_json(profile: RouteProfile) -> dict:
    world_totals = RouteProfileManager.world_summary(profile)
    world_counts = RouteProfileManager.world_step_counts(profile)

    return {
        "name": profile.name,
        "description": profile.description,
        "category": profile.category,
        "route_totals": {
            "total_cost": RouteProfileManager.route_total_cost(profile),
            "total_move": RouteProfileManager.route_total_move(profile),
            "total_drag": RouteProfileManager.route_total_drag(profile),
            "step_count": len(profile.steps),
        },
        "world_totals": {
            str(world): {
                "total_cost": world_totals[world],
                "step_count": world_counts.get(world, 0),
            }
            for world in sorted(world_totals)
        },
        "steps": route_profile_to_dict(profile)["steps"],
    }


def write_profile_json(profile: RouteProfile, out_path: str | Path) -> Path:
    path = Path(out_path)
    data = export_profile_json(profile)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path