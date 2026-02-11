# Hardware Setup for Reproducible Benchmarks

All benchmark results must be collected under controlled, reproducible conditions.
This document specifies the required hardware configuration for each target platform.

## General Requirements

- Disable dynamic frequency scaling (set CPU governor to `performance`)
- Disable turbo/boost modes for stable measurements
- Ensure adequate thermal management (active cooling recommended)
- Monitor CPU temperature during benchmarks; abort if exceeding thresholds
- Minimize background processes (disable unused services, networking if possible)
- Use the same OS kernel version and driver versions across measurement runs

## NVIDIA DGX Spark

- **Power mode**: Max performance (`sudo nvpmodel -m 0`)
- **GPU clocks**: Lock to sustained frequency (`sudo nvidia-smi -lgc 1500,1500`)
- **CPU governor**: `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
- **ECC memory**: Enabled (default)
- **Thermal limit**: 100W GPU power cap, monitor via `nvidia-smi dmon`

## Intel Xeon Server

- **BIOS settings**:
  - Disable Hyper-Threading (or ensure benchmark pins to physical cores only)
  - Disable Intel Turbo Boost Technology
  - Set CPU frequency to fixed max base frequency
  - Disable C-states deeper than C1 (`idle=poll` kernel parameter or BIOS)
  - Enable Intel AMX if available
- **OS-level**:
  - `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
  - `echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo`
  - Pin benchmark to specific NUMA node: `numactl --cpunodebind=0 --membind=0`
- **Thermal**: Monitor with `sensors` (lm-sensors), threshold 80C

## NVIDIA Jetson Orin Nano

- **Power mode**: `sudo nvpmodel -m 0` (MAXN)
- **Clock lock**: `sudo jetson_clocks` (locks CPU, GPU, EMC to max)
- **Fan**: Ensure active fan at full speed (`sudo jetson_clocks --fan`)
- **Thermal**: Monitor via `/sys/devices/virtual/thermal/thermal_zone*/temp`
- **Power**: Read from INA3221 sensor at `/sys/bus/i2c/drivers/ina3221x/*/iio:device*/`

## Raspberry Pi 5

- **/boot/firmware/config.txt**:
  ```
  arm_freq=2400
  force_turbo=1
  over_voltage=0
  ```
- **CPU governor**: `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
- **Thermal**: Active cooler or heatsink case required
  - Monitor: `vcgencmd measure_temp`
  - Throttle check: `vcgencmd get_throttled` (should return `0x0`)
- **Disable unused interfaces** (reduce thermal load and jitter):
  ```
  dtoverlay=disable-bt
  dtoverlay=disable-wifi
  ```
- **Storage**: Use NVMe via HAT+ for model loading; microSD is too slow for large models
