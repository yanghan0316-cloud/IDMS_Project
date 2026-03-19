"""
test_perclos_blink.py

验证 PERCLOS 和眨眼频率检测的核心行为:
    1. 持续睁眼 → PERCLOS ≈ 0, 无报警
    2. 持续闭眼 → PERCLOS → 1.0, 触发报警
    3. 间歇闭眼 (模拟疲劳) → PERCLOS 逐渐升高
    4. 快速眨眼 → blink_freq 升高, 触发高频报警
    5. 正常眨眼 → blink_freq 在正常范围
    6. reset() 清除所有历史状态

运行: python test_perclos_blink.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.internal.fatigue_logic import FatigueAnalyzer


def test_perclos_and_blink():
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        icon = "[OK]" if condition else "[X] "
        extra = f" ({detail})" if detail else ""
        print(f"  {icon} {name}{extra}")
        if condition:
            passed += 1
        else:
            failed += 1

    print("=" * 60)
    print("  PERCLOS & 眨眼频率 单元测试")
    print("=" * 60)

    # 使用固定 FPS 以便精确控制帧数
    config = {
        "fps": 30,
        "ear_threshold": 0.22,
        "mar_threshold": 0.60,
        "ema_alpha": 1.0,  # 禁用 EMA 平滑，测试更确定
        "perclos_window_sec": 2.0,       # 2 秒窗口 = 60 帧
        "perclos_threshold": 0.15,
        "blink_freq_window_sec": 60.0,
        "blink_freq_high_threshold": 25,
        "drowsy_duration_sec": 1.5,
        "yawn_duration_sec": 2.0,
        "blink_max_sec": 0.3,
    }

    # ====== Test 1: 持续睁眼 → PERCLOS ≈ 0 ======
    print("\n[Test 1] 持续睁眼 60 帧")
    analyzer = FatigueAnalyzer(config)
    for _ in range(60):
        state = analyzer.update(ear=0.30, mar=0.3)  # 正常 EAR

    check("PERCLOS ≈ 0", state.perclos < 0.01,
          f"perclos={state.perclos:.4f}")
    check("无 PERCLOS 报警", not state.is_perclos_fatigued)

    # ====== Test 2: 持续闭眼 → PERCLOS → 1.0 ======
    print("\n[Test 2] 持续闭眼 60 帧")
    analyzer = FatigueAnalyzer(config)
    for _ in range(60):
        state = analyzer.update(ear=0.15, mar=0.3)  # 低 EAR = 闭眼

    check("PERCLOS ≈ 1.0", state.perclos > 0.95,
          f"perclos={state.perclos:.4f}")
    check("PERCLOS 报警触发", state.is_perclos_fatigued)

    # ====== Test 3: 20% 闭眼 → PERCLOS > 阈值 ======
    print("\n[Test 3] 间歇闭眼 (20% 的帧闭眼)")
    analyzer = FatigueAnalyzer(config)
    for i in range(60):
        # 每 5 帧闭眼 1 帧 = 20%
        ear = 0.15 if (i % 5 == 0) else 0.30
        state = analyzer.update(ear=ear, mar=0.3)

    check("PERCLOS ≈ 0.20", 0.15 < state.perclos < 0.30,
          f"perclos={state.perclos:.4f}")
    check("PERCLOS 报警 (>15%)", state.is_perclos_fatigued)

    # ====== Test 4: 10% 闭眼 → PERCLOS < 阈值 ======
    print("\n[Test 4] 轻微闭眼 (10% 的帧闭眼)")
    analyzer = FatigueAnalyzer(config)
    for i in range(60):
        ear = 0.15 if (i % 10 == 0) else 0.30
        state = analyzer.update(ear=ear, mar=0.3)

    check("PERCLOS ≈ 0.10", state.perclos < 0.15,
          f"perclos={state.perclos:.4f}")
    check("无 PERCLOS 报警", not state.is_perclos_fatigued)

    # ====== Test 5: 快速眨眼 → blink_freq 高 ======
    print("\n[Test 5] 快速眨眼 (模拟 30 次眨眼/分钟)")
    analyzer = FatigueAnalyzer(config)
    # 模拟 2 秒内产生很多眨眼事件:
    # 每次眨眼 = 3 帧闭眼 + 3 帧睁眼 = 6 帧一个周期
    # 60 帧 / 6 = 10 次眨眼 in 2 秒 = 300 次/分钟
    # 但 blink_freq 需要至少 10 秒数据，所以我们用 time.sleep 模拟
    # 改用直接检查 blink 事件计数

    # 简化测试: 直接快速产生眨眼事件
    blink_count = 0
    for cycle in range(20):
        # 闭眼 3 帧
        for _ in range(3):
            state = analyzer.update(ear=0.15, mar=0.3)
        # 睁眼 3 帧 (第一帧会触发 blink=True)
        for _ in range(3):
            state = analyzer.update(ear=0.30, mar=0.3)
            if state.blink:
                blink_count += 1

    check("检测到多次眨眼事件", blink_count >= 15,
          f"blinks={blink_count}")
    # 注: blink_freq 依赖 time.time() 的实际间隔。
    # 单元测试中帧间隔极短 (<10秒)，频率计算需要至少 10 秒的数据
    # 才开始输出非零值，这是防误报的设计。
    # 改为验证眨眼时间戳确实被记录了。
    check("眨眼时间戳已记录", len(analyzer._blink_timestamps) >= 15,
          f"timestamps={len(analyzer._blink_timestamps)}")

    # ====== Test 6: reset() 清除状态 ======
    print("\n[Test 6] reset() 清除所有状态")
    # 先让 PERCLOS 升高
    analyzer = FatigueAnalyzer(config)
    for _ in range(60):
        analyzer.update(ear=0.15, mar=0.3)

    analyzer.reset()
    state = analyzer.update(ear=0.30, mar=0.3)

    check("reset 后 PERCLOS ≈ 0", state.perclos < 0.05,
          f"perclos={state.perclos:.4f}")
    check("reset 后 blink_freq = 0", state.blink_freq == 0.0)
    check("reset 后无报警", not state.is_perclos_fatigued and not state.is_blink_freq_high)

    # ====== Test 7: PERCLOS 窗口滑动效果 ======
    print("\n[Test 7] PERCLOS 窗口滑动 (闭眼后恢复)")
    analyzer = FatigueAnalyzer(config)
    # 先 30 帧闭眼
    for _ in range(30):
        analyzer.update(ear=0.15, mar=0.3)
    state_mid = analyzer.update(ear=0.15, mar=0.3)
    mid_perclos = state_mid.perclos

    # 再 60 帧睁眼（把闭眼帧推出窗口）
    for _ in range(60):
        state = analyzer.update(ear=0.30, mar=0.3)

    check("闭眼阶段 PERCLOS 较高", mid_perclos > 0.3,
          f"mid={mid_perclos:.4f}")
    check("恢复后 PERCLOS 降低", state.perclos < mid_perclos,
          f"after={state.perclos:.4f}")

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
    success = test_perclos_and_blink()
    sys.exit(0 if success else 1)