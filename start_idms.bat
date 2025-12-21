@echo off
echo ========================================
echo   正在启动 IDMS 竞赛环境...
echo ========================================
call conda activate idms_env
echo 环境已激活: idms_env
echo 正在验证环境...
python test-environment.py
if %errorlevel% equ 0 (
    echo.
    echo ✅ 环境验证成功！可以开始编程。
    echo 输入要运行的Python文件名（如: python main.py）
) else (
    echo.
    echo ❌ 环境验证失败！请检查。
    pause
    exit /b 1
)
echo ========================================
cmd /k