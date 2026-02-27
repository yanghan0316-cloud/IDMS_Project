
"""
src.ui.alert_system

声音报警模块：当连续多帧出现 DANGER 级别警告时，播放警报音。

功能：
- 连续 N 帧 DANGER → 触发声音警报
- 冷却期机制：报警后一段时间内不会重复播放（防止轰炸）
- 支持舱外（碰撞）和舱内（疲劳/分心）两路独立报警
- 如果没有音频文件，自动生成合成蜂鸣音

用法：
    alerter = AudioAlerter(config)

    # 每帧调用：
    alerter.update(
        ext_danger=True,   # 舱外是否有 DANGER
        int_danger=False,   # 舱内是否有 DANGER（疲劳/分心等）
    )

    # 程序退出时：
    alerter.close()

配置项（config.yaml -> ui -> alert）：
    enable: true
    consecutive_frames: 5        # 连续多少帧 DANGER 才触发
    cooldown_sec: 3.0            # 报警后冷却时间（秒）
    volume: 0.7                  # 音量 0.0 ~ 1.0
    ext_sound: "assets/alert_collision.wav"   # 舱外警报音（可选）
    int_sound: "assets/alert_fatigue.wav"     # 舱内警报音（可选）
"""

from __future__ import annotations

import os
import time
import struct
import math
import wave
import tempfile
from typing import Dict, Optional


class AudioAlerter:
    """
    双通道声音报警器（舱外碰撞 + 舱内疲劳）
    """

    def __init__(self, config: Dict):
        alert_cfg = config.get("alert", {}) if config else {}

        self.enabled = bool(alert_cfg.get("enable", True))
        self.consecutive_threshold = int(alert_cfg.get("consecutive_frames", 5))
        self.cooldown_sec = float(alert_cfg.get("cooldown_sec", 3.0))
        self.volume = float(alert_cfg.get("volume", 0.7))

        # 连续帧计数器
        self._ext_danger_count = 0
        self._int_danger_count = 0

        # 冷却时间戳
        self._ext_last_alert_time = 0.0
        self._int_last_alert_time = 0.0

        # pygame 初始化
        self._mixer_ready = False
        self._ext_sound = None
        self._int_sound = None

        if not self.enabled:
            return

        try:
            import pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self._mixer_ready = True
        except Exception as e:
            print(f"[AudioAlerter] pygame.mixer 初始化失败: {e}")
            print("[AudioAlerter] 声音报警将被禁用。")
            return

        # 加载或生成警报音
        ext_path = alert_cfg.get("ext_sound", "")
        int_path = alert_cfg.get("int_sound", "")

        self._ext_sound = self._load_or_generate(ext_path, freq=880, duration=0.5)
        self._int_sound = self._load_or_generate(int_path, freq=660, duration=0.8)

        if self._ext_sound:
            self._ext_sound.set_volume(self.volume)
        if self._int_sound:
            self._int_sound.set_volume(self.volume)

        print(f"[AudioAlerter] 已启用 | 连续帧阈值: {self.consecutive_threshold} | "
              f"冷却: {self.cooldown_sec}s | 音量: {self.volume}")

    def _load_or_generate(self, path: str, freq: float, duration: float):
        """
        尝试加载音频文件，如果不存在则自动生成一段合成蜂鸣音。
        """
        import pygame

        # 1) 尝试加载用户提供的音频文件
        if path and os.path.isfile(path):
            try:
                sound = pygame.mixer.Sound(path)
                print(f"[AudioAlerter] 已加载音频: {path}")
                return sound
            except Exception as e:
                print(f"[AudioAlerter] 加载 {path} 失败: {e}，将使用合成音")

        # 2) 自动生成合成蜂鸣音（纯正弦波）
        try:
            wav_path = self._generate_beep_wav(freq, duration)
            sound = pygame.mixer.Sound(wav_path)
            # 生成后可以删掉临时文件
            os.unlink(wav_path)
            return sound
        except Exception as e:
            print(f"[AudioAlerter] 合成蜂鸣音失败: {e}")
            return None

    @staticmethod
    def _generate_beep_wav(freq: float, duration: float,
                           sample_rate: int = 22050, amplitude: float = 0.6) -> str:
        """
        生成一段纯正弦波 WAV 文件（带淡入淡出避免爆音），返回临时文件路径。
        """
        n_samples = int(sample_rate * duration)
        fade_samples = min(int(sample_rate * 0.02), n_samples // 4)  # 20ms 淡入淡出

        samples = []
        for i in range(n_samples):
            # 正弦波
            value = amplitude * math.sin(2.0 * math.pi * freq * i / sample_rate)

            # 淡入
            if i < fade_samples:
                value *= i / fade_samples
            # 淡出
            elif i > n_samples - fade_samples:
                value *= (n_samples - i) / fade_samples

            # 转为 16-bit PCM
            sample_int = max(-32767, min(32767, int(value * 32767)))
            samples.append(struct.pack('<h', sample_int))

        # 写入临时 WAV 文件
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp.name
        tmp.close()

        with wave.open(tmp_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(samples))

        return tmp_path

    def update(self, ext_danger: bool = False, int_danger: bool = False) -> Dict[str, bool]:
        """
        每帧调用一次，根据当前帧的危险状态决定是否播放声音。

        Args:
            ext_danger: 舱外是否存在 DANGER 级别目标
            int_danger: 舱内是否存在危险状态（疲劳/分心/点头等）

        Returns:
            dict: {"ext_alert_fired": bool, "int_alert_fired": bool}
        """
        result = {"ext_alert_fired": False, "int_alert_fired": False}

        if not self.enabled or not self._mixer_ready:
            return result

        now = time.time()

        # --- 舱外碰撞警报 ---
        if ext_danger:
            self._ext_danger_count += 1
        else:
            self._ext_danger_count = 0

        if (self._ext_danger_count >= self.consecutive_threshold
                and (now - self._ext_last_alert_time) > self.cooldown_sec):
            if self._ext_sound:
                self._ext_sound.play()
                self._ext_last_alert_time = now
                result["ext_alert_fired"] = True

        # --- 舱内疲劳/分心警报 ---
        if int_danger:
            self._int_danger_count += 1
        else:
            self._int_danger_count = 0

        if (self._int_danger_count >= self.consecutive_threshold
                and (now - self._int_last_alert_time) > self.cooldown_sec):
            if self._int_sound:
                self._int_sound.play()
                self._int_last_alert_time = now
                result["int_alert_fired"] = True

        return result

    def close(self):
        """释放 pygame mixer 资源"""
        if self._mixer_ready:
            try:
                import pygame
                pygame.mixer.quit()
            except Exception:
                pass



