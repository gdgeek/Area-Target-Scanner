#!/bin/bash
# Build script for macOS universal binary (x86_64 + arm64)
# Prerequisites: brew install opencv cmake
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Building macOS universal binary ==="

# Configure with CMake
cmake -B "$SCRIPT_DIR/build" \
    -S "$SCRIPT_DIR" \
    -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" \
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build "$SCRIPT_DIR/build" --config Release

# Verify the output
DYLIB="$SCRIPT_DIR/build/libvisual_localizer.dylib"
if [ ! -f "$DYLIB" ]; then
    echo "ERROR: $DYLIB not found"
    exit 1
fi

echo "=== Verifying architectures ==="
file "$DYLIB"
lipo -info "$DYLIB"

echo "=== Checking binary size ==="
ls -lh "$DYLIB"

# Copy to Unity plugin directory
DEST="$SCRIPT_DIR/../unity_project/Assets/Plugins/macOS/libvisual_localizer.dylib"
cp "$DYLIB" "$DEST"
echo "=== Copied to $DEST ==="

echo "=== Done ==="
