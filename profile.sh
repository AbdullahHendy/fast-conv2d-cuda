#!/usr/bin/env bash
set -e

# Ensure output directory exists
mkdir -p profiles

TOOL=""
OUTPUT=""
GUI_CMD=""

case "$1" in
  --ncu)
    OUTPUT="profiles/profile.ncu-repz"
    # Detailed NCU profiling;
    TOOL="ncu --set detailed --import-source yes -o profiles/profile --force-overwrite --profile-from-start 0 --kernel-name regex:(implicitUnrollWmmaTC|convTiled)"
    GUI_CMD="ncu-ui"
    ;;
  --nsys)
    OUTPUT="profiles/profile.nsys-rep"
    # System-level trace (CUDA API, kernel execution, and CPU/NVTX ranges)
    TOOL="nsys profile -t cuda,nvtx,osrt -o profiles/profile --force-overwrite true"
    GUI_CMD="nsys-ui"
    ;;
  --memcheck)
    TOOL="compute-sanitizer --tool memcheck"
    ;;
  --racecheck)
    TOOL="compute-sanitizer --tool racecheck"
    ;;
  --synccheck)
    TOOL="compute-sanitizer --tool synccheck"
    ;;
  *)
    echo "Usage: ./profile.sh [--ncu | --nsys | --memcheck | --racecheck | --synccheck] [--gui]"
    exit 1
    ;;
esac

echo "==> Running: $TOOL python3 benchmark.py"
$TOOL .venv/bin/python3 benchmark.py

# Launch GUI if selected with --gui and GUI_CMD is set
if [[ "$2" == "--gui" && -n "$GUI_CMD" ]]; then
  echo "==> Launching $GUI_CMD..."
  $GUI_CMD "$OUTPUT" &
fi