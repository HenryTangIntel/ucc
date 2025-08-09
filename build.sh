#!/bin/bash

# UCC Build Script with HCCL Support
# This script configures and builds UCC with HCCL (Habana Communication Library) support

set -e  # Exit on any error
set -x  # Print commands as they execute

echo "=== UCC Build Script with HCCL Support ==="

# Check if we're in the right directory
if [ ! -f "configure.ac" ]; then
    echo "Error: Must run from UCC root directory"
    exit 1
fi

# Clean previous build artifacts
echo "Cleaning previous build artifacts..."
make clean || true
make distclean || true

# Remove autoconf cache to ensure fresh detection
echo "Removing autoconf cache..."
rm -rf autom4te.cache config.cache

# Regenerate build system
echo "Regenerating build system..."
./autogen.sh

# Set UCX path (required dependency)
export UCX_DIR="/workspace/ucx-gaudi/install"

# Configure with HCCL support
echo "Configuring UCC with HCCL support..."
./configure \
    --prefix=/workspace/ucc/install \
    --with-ucx="${UCX_DIR}" \
    --with-hccl=/usr \
    --enable-debug \
    --enable-assertions

# Verify HCCL was detected
echo "Checking HCCL configuration..."
grep -i "HCCL support" config.log || echo "Warning: HCCL support not clearly indicated"

# Build UCC
echo "Building UCC..."
make -j$(nproc)

# Verify HCCL component was built
echo "Verifying HCCL component build..."
if [ -f "src/components/tl/hccl/.libs/libucc_tl_hccl.so" ]; then
    echo "✅ HCCL component built successfully!"
    ls -la src/components/tl/hccl/.libs/libucc_tl_hccl.*
else
    echo "❌ HCCL component build failed!"
    exit 1
fi

# Optional: Install UCC
echo "Installing UCC..."
make install

echo "=== Build completed successfully! ==="
echo "UCC with HCCL support has been built and installed to /workspace/ucc/install"
echo ""
echo "Key components built:"
echo "- UCC core library: $(find install/lib -name 'libucc.so*' | head -1)"
echo "- HCCL transport: src/components/tl/hccl/.libs/libucc_tl_hccl.so"
echo ""
echo "To use: export LD_LIBRARY_PATH=/workspace/ucc/install/lib:\$LD_LIBRARY_PATH"
