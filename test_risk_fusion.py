"""
test_risk_fusion.py

验证多模态融合引擎的核心行为:
    1. 单路舱外高危 → HIGH（不是 CRITICAL）
    2. 单路舱内高危 → HIGH（不是 CRITICAL）
    3. 双路同时高危 → CRITICAL（交叉项放大！）
    4. 双路低危 → LOW（不过度报警）
    5. 空输入 → SAFE

运行: python test_risk_fusion.py
"""

import sys
import os

# 确保能找到 src 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.risk_fusion import RiskFusionEngine, LEVEL_SAFE, LEVEL_LOW, LEVEL_HIGH, LEVEL_CRITICAL


def test_fusion():
    config = {}  # 使用默认参数
    engine = RiskFusionEngine(config)

    passed = 0
    failed = 0

    def check(name, actual_level, expected_levels, score=None):
        nonlocal passed, failed
        ok = actual_level in expected_levels
        icon = "[OK]" if ok else "[X] "
        score_str = f" (score={score:.3f})" if score is not None else ""
        expected_names = [
            {0: "SAFE", 1: "LOW", 2: "HIGH", 3: "CRITICAL"}[l]
            for l in expected_levels
        ]
        actual_name = {0: "SAFE", 1: "LOW", 2: "HIGH", 3: "CRITICAL"}[actual_level]
        print(f"  {icon} {name}: got {actual_name}{score_str}, "
              f"expected {'/'.join(expected_names)}")
        if ok:
            passed += 1
        else:
            failed += 1

    print("=" * 60)
    print("  多模态融合引擎 单元测试")
    print("=" * 60)

    # ====== 场景 1: 空输入 → SAFE ======
    print("\n[Test 1] 空输入")
    engine.reset()
    r = engine.evaluate(vehicle_data=None, face_data=None)
    check("无数据", r.fused_level, [LEVEL_SAFE], r.fused_score)

    # ====== 场景 2: 仅舱外高危 (TTC=1.0, 距离=2m) ======
    print("\n[Test 2] 仅舱外高危")
    engine.reset()
    vehicle_high = [{
        "box": [300, 200, 500, 400],
        "distance": 2.0,
        "ttc": 1.0,
        "warning_level": 2,
        "lane_relevance": 1.0,
        "rel_speed": 5.0,
    }]
    face_normal = {
        "has_face": True,
        "ear": 0.28, "mar": 0.3,
        "is_drowsy": False, "is_yawning": False,
        "is_distracted": False, "is_nodding": False,
        "yaw": 2.0, "pitch": -5.0,
    }
    r = engine.evaluate(vehicle_data=vehicle_high, face_data=face_normal)
    check("舱外高危 + 驾驶员正常",
          r.fused_level, [LEVEL_LOW, LEVEL_HIGH], r.fused_score)
    print(f"       ext={r.ext_score:.3f} int={r.int_score:.3f} "
          f"cross={r.cross_score:.3f}")

    # ====== 场景 3: 仅舱内高危 (疲劳 + 分心) ======
    print("\n[Test 3] 仅舱内高危")
    engine.reset()
    face_danger = {
        "has_face": True,
        "ear": 0.15, "mar": 0.7,
        "is_drowsy": True, "is_yawning": True,
        "is_distracted": True, "is_nodding": False,
        "yaw": 35.0, "pitch": -10.0,
    }
    r = engine.evaluate(vehicle_data=[], face_data=face_danger)
    check("驾驶员高危 + 前方空旷",
          r.fused_level, [LEVEL_LOW, LEVEL_HIGH], r.fused_score)
    print(f"       ext={r.ext_score:.3f} int={r.int_score:.3f} "
          f"cross={r.cross_score:.3f}")

    # ====== 场景 4: 双路同时高危 → 应该是 CRITICAL ======
    print("\n[Test 4] 双路叠加高危 (核心场景!)")
    engine.reset()
    r = engine.evaluate(vehicle_data=vehicle_high, face_data=face_danger)
    check("舱外高危 + 驾驶员高危 → CRITICAL",
          r.fused_level, [LEVEL_CRITICAL], r.fused_score)
    print(f"       ext={r.ext_score:.3f} int={r.int_score:.3f} "
          f"cross={r.cross_score:.3f}")
    print(f"       ↑ 交叉项使分值从单路水平跃升到 CRITICAL")

    # ====== 场景 5: 双路轻微 → 不应过度报警 ======
    print("\n[Test 5] 双路轻微异常")
    engine.reset()
    vehicle_mild = [{
        "box": [300, 200, 400, 350],
        "distance": 20.0,
        "ttc": 5.0,
        "warning_level": 1,
        "lane_relevance": 1.0,
        "rel_speed": 1.0,
    }]
    face_mild = {
        "has_face": True,
        "ear": 0.24, "mar": 0.4,
        "is_drowsy": False, "is_yawning": False,
        "is_distracted": False, "is_nodding": False,
        "yaw": 12.0, "pitch": -8.0,
    }
    r = engine.evaluate(vehicle_data=vehicle_mild, face_data=face_mild)
    check("双路轻微 → 不应 CRITICAL",
          r.fused_level, [LEVEL_SAFE, LEVEL_LOW], r.fused_score)
    print(f"       ext={r.ext_score:.3f} int={r.int_score:.3f} "
          f"cross={r.cross_score:.3f}")

    # ====== 场景 6: 验证无人脸时舱内归零 ======
    print("\n[Test 6] 无人脸检测")
    engine.reset()
    r = engine.evaluate(vehicle_data=vehicle_high,
                        face_data={"has_face": False})
    check("舱外高危 + 无人脸 → 仅舱外贡献",
          r.fused_level, [LEVEL_LOW, LEVEL_HIGH], r.fused_score)
    assert r.int_score == 0.0, "无人脸时舱内分值应为 0"
    print(f"       int_score={r.int_score} (正确归零)")

    # ====== 总结 ======
    print(f"\n{'=' * 60}")
    total = passed + failed
    print(f"  测试结果: {passed}/{total} 通过")
    if failed == 0:
        print("  All tests passed!")
    else:
        print(f"  {failed} tests FAILED.")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = test_fusion()
    sys.exit(0 if success else 1)