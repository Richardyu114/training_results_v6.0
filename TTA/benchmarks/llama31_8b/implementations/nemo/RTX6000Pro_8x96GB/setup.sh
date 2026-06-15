#!/bin/bash
# MLPerf Training v6.0 - LLaMA 3.1 8B - Setup Script

set -e

echo "=== MLPerf Training v6.0 Setup ==="

echo "Building Docker image..."
cd "$(dirname "$0")/../code"
docker build -t mlperf-nvidia:llama31_8b-pyt -f Dockerfile .

echo "Applying system optimizations..."

# GPU: persistence mode
sudo nvidia-smi -pm 1

# CPU: performance governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Memory: disable transparent hugepages (reduces jitter)
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/defrag

echo "Setup complete."
echo ""
echo "Verify GPU clocks:"
nvidia-smi --query-gpu=index,clocks.current.graphics,clocks.max.graphics --format=csv
