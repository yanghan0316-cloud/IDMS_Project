"""External monitoring package with lazy imports for optional YOLO runtime."""

__all__ = ["YoloDetector", "DistanceEstimator", "CollisionWarner"]


def __getattr__(name):
    if name == "YoloDetector":
        from .yolo_detector import YoloDetector
        return YoloDetector
    if name == "DistanceEstimator":
        from .distance_est import DistanceEstimator
        return DistanceEstimator
    if name == "CollisionWarner":
        from .collision_warn import CollisionWarner
        return CollisionWarner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
