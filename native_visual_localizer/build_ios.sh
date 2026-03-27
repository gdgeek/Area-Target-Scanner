#!/bin/bash
# Build script for iOS arm64 static library
# Downloads OpenCV iOS framework if not present, then cross-compiles
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build_ios"
OPENCV_DIR="$SCRIPT_DIR/opencv_ios"
OPENCV_VER="4.10.0"
OPENCV_URL="https://github.com/opencv/opencv/releases/download/${OPENCV_VER}/opencv-${OPENCV_VER}-ios-framework.zip"
DEST="$SCRIPT_DIR/../unity_project/Assets/Plugins/iOS/libvisual_localizer.a"

echo "=== Building iOS arm64 static library ==="

# Step 1: Download OpenCV iOS framework if needed
if [ ! -d "$OPENCV_DIR/opencv2.framework" ]; then
    echo "--- Downloading OpenCV ${OPENCV_VER} iOS framework ---"
    mkdir -p "$OPENCV_DIR"
    curl -L "$OPENCV_URL" -o "$OPENCV_DIR/opencv_ios.zip"
    unzip -q "$OPENCV_DIR/opencv_ios.zip" -d "$OPENCV_DIR"
    rm "$OPENCV_DIR/opencv_ios.zip"
    echo "--- OpenCV downloaded ---"
fi

OPENCV_FW="$OPENCV_DIR/opencv2.framework"
if [ ! -d "$OPENCV_FW" ]; then
    echo "ERROR: opencv2.framework not found at $OPENCV_FW"
    echo "Contents of $OPENCV_DIR:"
    ls -la "$OPENCV_DIR"
    exit 1
fi

echo "--- OpenCV framework: $OPENCV_FW ---"

# Step 2: Compile C++ sources for iOS arm64
mkdir -p "$BUILD_DIR"

SYSROOT=$(xcrun --sdk iphoneos --show-sdk-path)
CC=$(xcrun --sdk iphoneos --find clang)
CXX=$(xcrun --sdk iphoneos --find clang++)

echo "--- Compiling visual_localizer.cpp ---"
$CXX -std=c++17 -O2 -DNDEBUG -arch arm64 -isysroot "$SYSROOT" \
    -miphoneos-version-min=14.0 \
    -I"$SCRIPT_DIR/include" \
    -I"$SCRIPT_DIR/src" \
    -F"$OPENCV_DIR" \
    -fvisibility=hidden -fvisibility-inlines-hidden \
    -fPIC \
    -c "$SCRIPT_DIR/src/visual_localizer.cpp" -o "$BUILD_DIR/visual_localizer.o"

echo "--- Compiling visual_localizer_impl.cpp ---"
$CXX -std=c++17 -O2 -DNDEBUG -arch arm64 -isysroot "$SYSROOT" \
    -miphoneos-version-min=14.0 \
    -I"$SCRIPT_DIR/include" \
    -I"$SCRIPT_DIR/src" \
    -F"$OPENCV_DIR" \
    -fvisibility=hidden -fvisibility-inlines-hidden \
    -fPIC \
    -c "$SCRIPT_DIR/src/visual_localizer_impl.cpp" -o "$BUILD_DIR/visual_localizer_impl.o"

# Step 3: Create static library
echo "--- Creating static library ---"
ar rcs "$BUILD_DIR/libvisual_localizer.a" \
    "$BUILD_DIR/visual_localizer.o" \
    "$BUILD_DIR/visual_localizer_impl.o"

# Step 4: Verify
echo "=== Verifying ==="
file "$BUILD_DIR/libvisual_localizer.a"
lipo -info "$BUILD_DIR/libvisual_localizer.a"
nm "$BUILD_DIR/libvisual_localizer.a" | grep " T _vl_" | head -20
echo ""
echo "--- Checking all 11 exported symbols ---"
nm "$BUILD_DIR/libvisual_localizer.a" | grep "vl_create" || echo "WARNING: vl_create NOT found!"
nm "$BUILD_DIR/libvisual_localizer.a" | grep "vl_destroy" || echo "WARNING: vl_destroy NOT found!"
nm "$BUILD_DIR/libvisual_localizer.a" | grep "vl_add_vocabulary_word" || echo "WARNING: vl_add_vocabulary_word NOT found!"
nm "$BUILD_DIR/libvisual_localizer.a" | grep "vl_add_keyframe" || echo "WARNING: vl_add_keyframe NOT found!"
nm "$BUILD_DIR/libvisual_localizer.a" | grep "vl_add_keyframe_akaze" || echo "WARNING: vl_add_keyframe_akaze NOT found!"
nm "$BUILD_DIR/libvisual_localizer.a" | grep "vl_build_index" || echo "WARNING: vl_build_index NOT found!"
nm "$BUILD_DIR/libvisual_localizer.a" | grep "vl_process_frame" || echo "WARNING: vl_process_frame NOT found!"
nm "$BUILD_DIR/libvisual_localizer.a" | grep "vl_process_frame_out" || echo "WARNING: vl_process_frame_out NOT found!"
nm "$BUILD_DIR/libvisual_localizer.a" | grep "vl_reset" || echo "WARNING: vl_reset NOT found!"
nm "$BUILD_DIR/libvisual_localizer.a" | grep "vl_set_alignment_transform" || echo "WARNING: vl_set_alignment_transform NOT found!"
nm "$BUILD_DIR/libvisual_localizer.a" | grep "vl_get_debug_info" || echo "WARNING: vl_get_debug_info NOT found!"

# Step 5: Copy to Unity
echo ""
echo "--- Backing up old library ---"
if [ -f "$DEST" ]; then
    cp "$DEST" "${DEST}.bak"
fi
cp "$BUILD_DIR/libvisual_localizer.a" "$DEST"
echo "=== Deployed to $DEST ==="

# Step 6: Size comparison
echo ""
echo "--- Size comparison ---"
ls -lh "${DEST}.bak" 2>/dev/null || true
ls -lh "$DEST"

echo ""
echo "=== Done! Rebuild Unity iOS project to use the new library. ==="
