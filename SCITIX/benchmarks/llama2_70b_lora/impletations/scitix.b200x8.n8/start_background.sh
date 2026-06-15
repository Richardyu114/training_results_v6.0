#!/usr/bin/env bash
set -u

mkdir -p logs

# 后台运行主脚本，把主脚本自身的输出写到 driver.log
nohup bash ./run_10_times_bg.sh > logs/driver.log 2>&1 &

echo "Started in background."
echo "PID: $!"
echo "Progress log: logs/driver.log"

