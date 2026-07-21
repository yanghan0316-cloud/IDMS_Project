import os
import sys
from dataclasses import dataclass


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.mrm_planner import (  # noqa: E402
    ACTION_BRAKE,
    ACTION_EMERGENCY_BRAKE,
    ACTION_KEEP,
    ACTION_SLOW_DOWN,
    MRMPlanner,
)


@dataclass
class FakeFusion:
    ext_score: float = 0.0
    int_score: float = 0.0
    cross_score: float = 0.0
    fused_score: float = 0.0
    fused_level: int = 0
    fused_text: str = "SAFE"
    ext_min_ttc: float = 99.0
    ext_min_dist: float = 99.0
    ext_max_level: int = 0
    int_fatigue_score: float = 0.0
    int_attention_score: float = 0.0
    int_drowsy: bool = False
    int_yawning: bool = False
    int_distracted: bool = False
    int_nodding: bool = False


def check(name, condition, details):
    icon = "[OK]" if condition else "[X]"
    print(f"{icon} {name}: {details}")
    if not condition:
        raise AssertionError(name)


def main():
    planner = MRMPlanner({"log_enable": False})

    safe = planner.plan(FakeFusion())
    check(
        "safe scene keeps current speed",
        safe.action == ACTION_KEEP,
        f"action={safe.action}, risk={safe.predicted_risk:.2f}",
    )

    driver_high = planner.plan(
        FakeFusion(
            int_score=0.78,
            fused_score=0.58,
            fused_level=2,
            fused_text="HIGH",
            int_attention_score=0.8,
            int_distracted=True,
        ),
        face_data={"has_face": True, "is_distracted": True},
    )
    check(
        "driver-only high risk slows down",
        driver_high.action == ACTION_SLOW_DOWN,
        f"action={driver_high.action}, reasons={driver_high.reasons}",
    )

    lead_close = [{
        "box": [260, 180, 420, 420],
        "class_name": "car",
        "distance": 8.0,
        "ttc": 1.2,
        "rel_speed": 6.0,
        "warning_level": 2,
        "lane_relevance": 1.0,
    }]
    forward_risk = planner.plan(
        FakeFusion(
            ext_score=0.92,
            int_score=0.10,
            cross_score=0.092,
            fused_score=0.45,
            fused_level=1,
            fused_text="LOW",
            ext_min_ttc=1.2,
            ext_min_dist=8.0,
            ext_max_level=2,
        ),
        vehicle_data=lead_close,
    )
    check(
        "forward short-term risk brakes",
        forward_risk.action in (ACTION_BRAKE, ACTION_EMERGENCY_BRAKE),
        f"action={forward_risk.action}, risk={forward_risk.predicted_risk:.2f}",
    )

    print("All MRM planner tests passed.")


if __name__ == "__main__":
    main()
