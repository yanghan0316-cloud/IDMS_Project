"""
test_driver_state.py

验证 DriverStateAssessor 的信号互相印证逻辑:

    1. 所有信号正常 → 风险≈0
    2. PERCLOS 单独高 → 中等风险（封顶，单信号不可靠）
    3. PERCLOS + 点头 → 风险明显高于任一单独（印证放大）
    4. PERCLOS + 点头 + 高频眨眼 → 三信号印证，风险最高
    5. PERCLOS高 + 分心（矛盾）→ 风险被削减
    6. 仅点头（无PERCLOS佐证）→ 封顶，不过度报警
    7. 分心 + 哈欠 → 注意力维度被印证放大

运行: python test_driver_state.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.internal.driver_state import DriverStateAssessor


def make_face(perclos=0.0, ear=0.28, is_drowsy=False, is_nodding=False,
              is_yawning=False, is_distracted=False, blink_freq=15.0,
              yaw=2.0, pitch=-5.0, **kw):
    return {
        "has_face": True, "perclos": perclos, "ear": ear,
        "is_drowsy": is_drowsy, "is_nodding": is_nodding,
        "is_yawning": is_yawning, "is_distracted": is_distracted,
        "blink_freq": blink_freq, "yaw": yaw, "pitch": pitch,
        "is_perclos_fatigued": perclos > 0.15,
        "is_blink_freq_high": blink_freq > 25,
        "mar": 0.3, "blink": False,
        **kw,
    }


def test():
    p, f = 0, 0

    def check(name, cond, detail=""):
        nonlocal p, f
        icon = "[OK]" if cond else "[X] "
        extra = f" ({detail})" if detail else ""
        print(f"  {icon} {name}{extra}")
        (p if cond else f).__class__  # just to count
        if cond: p += 1
        else: f += 1

    cfg = {}
    A = DriverStateAssessor(cfg)

    print("=" * 62)
    print("  DriverStateAssessor 协同评估 单元测试")
    print("=" * 62)

    # 1. 全正常
    print("\n[Test 1] 所有信号正常")
    r = A.evaluate(make_face())
    check("fatigue ≈ 0", r.fatigue_score < 0.10, f"f={r.fatigue_score:.3f}")
    check("attention ≈ 0", r.attention_score < 0.05, f"a={r.attention_score:.3f}")
    check("confidence=none", r.confidence_label == "none")

    # 2. PERCLOS 单独高
    print("\n[Test 2] PERCLOS=0.25 单独高，无其他疲劳信号")
    r2 = A.evaluate(make_face(perclos=0.25))
    check("fatigue 中等", 0.1 < r2.fatigue_score <= 0.6,
          f"f={r2.fatigue_score:.3f}")
    check("单信号封顶", r2.confidence_label == "single")

    # 3. PERCLOS + 点头 (两信号印证)
    print("\n[Test 3] PERCLOS=0.25 + 点头 (两信号印证)")
    r3 = A.evaluate(make_face(perclos=0.25, is_nodding=True))
    check("fatigue > 单PERCLOS", r3.fatigue_score > r2.fatigue_score,
          f"f={r3.fatigue_score:.3f} > {r2.fatigue_score:.3f}")
    check("印证标签", r3.confidence_label == "corroborated")

    # 4. PERCLOS + 点头 + 高频眨眼 (三信号印证)
    print("\n[Test 4] PERCLOS + 点头 + 高频眨眼 (三信号强印证)")
    r4 = A.evaluate(make_face(perclos=0.25, is_nodding=True, blink_freq=30))
    check("fatigue > 双信号", r4.fatigue_score > r3.fatigue_score,
          f"f={r4.fatigue_score:.3f} > {r3.fatigue_score:.3f}")
    check("3+ 信号印证", r4.fatigue_signals >= 3, f"signals={r4.fatigue_signals}")

    # 5. PERCLOS高 + 分心 (矛盾信号)
    print("\n[Test 5] PERCLOS=0.25 + 分心 (矛盾)")
    r5 = A.evaluate(make_face(perclos=0.25, is_distracted=True, yaw=35))
    check("fatigue < 单PERCLOS", r5.fatigue_score < r2.fatigue_score,
          f"contradicted={r5.fatigue_score:.3f} < single={r2.fatigue_score:.3f}")
    check("检测到矛盾", r5.has_contradiction)
    check("矛盾标签", r5.confidence_label == "contradicted")

    # 6. 仅点头 (无PERCLOS佐证)
    print("\n[Test 6] 仅点头，PERCLOS正常")
    r6 = A.evaluate(make_face(is_nodding=True))
    check("fatigue 适中", r6.fatigue_score <= 0.6,
          f"f={r6.fatigue_score:.3f}")
    check("单信号", r6.confidence_label == "single")

    # 7. 分心 + 哈欠 (注意力维度印证)
    print("\n[Test 7] 分心 + 哈欠 (注意力维度)")
    r7 = A.evaluate(make_face(is_distracted=True, is_yawning=True, yaw=30))
    r7_single = A.evaluate(make_face(is_distracted=True, yaw=30))
    check("attention > 单分心", r7.attention_score > r7_single.attention_score,
          f"dual={r7.attention_score:.3f} > single={r7_single.attention_score:.3f}")

    # 8. 综合风险: 全面疲劳 > 部分疲劳 > 单一信号 > 正常
    print("\n[Test 8] 风险排序验证")
    r_normal = A.evaluate(make_face())
    r_single = r2  # PERCLOS alone
    r_double = r3  # PERCLOS + nod
    r_triple = r4  # PERCLOS + nod + blink
    check("triple > double > single > normal",
          r_triple.driver_risk > r_double.driver_risk > r_single.driver_risk > r_normal.driver_risk,
          f"{r_triple.driver_risk:.3f} > {r_double.driver_risk:.3f} > "
          f"{r_single.driver_risk:.3f} > {r_normal.driver_risk:.3f}")

    # 9. 无人脸 → 全零
    print("\n[Test 9] 无人脸")
    r9 = A.evaluate({"has_face": False})
    check("全零", r9.driver_risk == 0.0 and r9.fatigue_score == 0.0)

    print(f"\n{'=' * 62}")
    total = p + f
    print(f"  测试结果: {p}/{total} 通过")
    if f == 0:
        print("  All tests passed!")
    else:
        print(f"  {f} tests FAILED.")
    print("=" * 62)
    return f == 0


if __name__ == "__main__":
    ok = test()
    sys.exit(0 if ok else 1)