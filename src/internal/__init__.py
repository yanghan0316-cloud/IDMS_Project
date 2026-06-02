"""Internal monitoring package.

Heavy dependencies such as OpenCV and MediaPipe are imported lazily so pure
logic modules can be tested without a camera/vision environment.
"""

__all__ = [
    "FaceMeshDetector",
    "FatigueAnalyzer",
    "AttentionAnalyzer",
    "DriverStateAssessor",
]


def __getattr__(name):
    if name == "FaceMeshDetector":
        from .face_mesh import FaceMeshDetector
        return FaceMeshDetector
    if name == "FatigueAnalyzer":
        from .fatigue_logic import FatigueAnalyzer
        return FatigueAnalyzer
    if name == "AttentionAnalyzer":
        from .attention_logic import AttentionAnalyzer
        return AttentionAnalyzer
    if name == "DriverStateAssessor":
        from .driver_state import DriverStateAssessor
        return DriverStateAssessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
