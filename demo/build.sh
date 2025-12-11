#!/bin/bash
# Build script for PoIntInt intersection volume demo
#
# Usage:
#   ./build.sh        # Build everything
#   ./build.sh clean  # Clean build directories

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$SCRIPT_DIR/../cpp"
DEMO_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Clean option
if [ "$1" == "clean" ]; then
    echo_info "Cleaning build directories..."
    rm -rf "$CPP_DIR/build"
    rm -rf "$DEMO_DIR/build"
    rm -f "$DEMO_DIR"/pointint_cpp*.so
    rm -f "$DEMO_DIR"/pointint_cpp*.pyd
    echo_info "Clean complete"
    exit 0
fi

# Step 1: Build cpp project (poIntInt_core library)
echo_info "Building poIntInt_core library..."
cd "$CPP_DIR"
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DBUILD_EXE=OFF \
    -DBUILD_TESTS=OFF

cmake --build . --target poIntInt_core --config Release -j$(nproc)

if [ ! -f "libpoIntInt_core.a" ] && [ ! -f "Release/poIntInt_core.lib" ]; then
    echo_error "Failed to build poIntInt_core library"
    exit 1
fi
echo_info "poIntInt_core built successfully"

# Step 2: Build Python bindings
echo_info "Building Python bindings..."
cd "$DEMO_DIR"
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build . --config Release -j$(nproc)

# Check if module was built
cd "$DEMO_DIR"
if ls pointint_cpp*.so 1> /dev/null 2>&1 || ls pointint_cpp*.pyd 1> /dev/null 2>&1; then
    echo_info "Python module built successfully!"
    echo_info "Module location: $(ls pointint_cpp*)"
else
    echo_error "Failed to build Python module"
    exit 1
fi

# Step 3: Test import
echo_info "Testing Python import..."
python3 -c "import pointint_cpp; print('Import successful!')" 2>/dev/null || {
    echo_warn "Python import test failed. You may need to:"
    echo_warn "  1. Ensure pybind11 is installed: pip install pybind11"
    echo_warn "  2. Run from the demo directory: cd $DEMO_DIR && python demo_intersection.py"
}

echo ""
echo_info "Build complete!"
echo_info "To run the demo:"
echo_info "  cd $DEMO_DIR"
echo_info "  python demo_intersection.py"
echo_info "Then open http://localhost:8080 in your browser"
