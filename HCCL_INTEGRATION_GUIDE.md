# HCCL Integration into UCC - Implementation Guide

## Overview
This document outlines the integration of HCCL (Habana Communication Library) as a Transport Layer (TL) component in UCC (Unified Collective Communication).

## Integration Architecture

### 1. Component Structure
```
src/components/tl/hccl/
├── tl_hccl.h              # Main header with data structures and interfaces
├── tl_hccl.c              # Main component implementation and interface
├── tl_hccl_lib.c          # Library initialization/cleanup
├── tl_hccl_context.c      # Context management
├── tl_hccl_team.c         # Team/communicator management
├── tl_hccl_coll.c         # Collective operations implementation
├── tl_hccl_coll.h         # Collective operations header
├── configure.m4           # Build configuration
└── Makefile.am            # Build rules
```

### 2. Configuration Files
```
config/m4/hccl.m4          # HCCL library detection and configuration
configure.ac               # Updated to include HCCL support
```

## Key Features Implemented

### 1. Core TL Components
- **Library Management**: Initialization and cleanup of HCCL library context
- **Context Management**: Per-context configuration and resource management
- **Team Management**: HCCL communicator initialization and team handling
- **Task Management**: Asynchronous collective operation tasks

### 2. Collective Operations Support
- **AllReduce**: Reduction operations across all ranks
- **AllGather**: Gather operations from all ranks to all ranks
- **Broadcast**: One-to-all data distribution
- **Barrier**: Synchronization primitive
- **Extensible**: Framework for additional HCCL collectives

### 3. Configuration Options
- **Completion Sync**: Event-based or memory operations completion detection
- **Lazy Initialization**: Optional delayed HCCL communicator setup
- **Blocking Mode**: Synchronous vs asynchronous operation modes

## Build Integration

### 1. Autotools Integration
- HCCL detection via `config/m4/hccl.m4`
- Automatic inclusion via `autogen.sh` script
- Conditional building based on HCCL availability

### 2. Build Options
```bash
# Enable HCCL support
./configure --with-hccl=/path/to/hccl

# Include HCCL in TL selection
./configure --with-tls=all  # or --with-tls=hccl,ucp,nccl
```

## Usage Example

### 1. Library Initialization
```c
ucc_lib_config_h lib_config;
ucc_lib_params_t lib_params;
ucc_lib_h        lib;

// Configure to use HCCL TL
lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
lib_params.thread_mode = UCC_THREAD_SINGLE;

ucc_lib_config_read(NULL, NULL, &lib_config);
ucc_init(&lib_params, lib_config, &lib);
```

### 2. Context and Team Creation
```c
ucc_context_h context;
ucc_team_h    team;

// Create context
ucc_context_create(lib, &context_params, &context);

// Create team with HCCL communicator
ucc_team_create_post(&context, &team_params, &team);
```

### 3. Collective Operations
```c
ucc_coll_args_t coll_args = {
    .mask      = UCC_COLL_ARGS_FIELD_FLAGS,
    .coll_type = UCC_COLL_TYPE_ALLREDUCE,
    .src.info  = {src_buf, count, UCC_DT_FLOAT32, UCC_MEMORY_TYPE_HOST},
    .dst.info  = {dst_buf, count, UCC_DT_FLOAT32, UCC_MEMORY_TYPE_HOST},
    .op        = UCC_OP_SUM
};

ucc_collective_init(&coll_args, &request, team);
ucc_collective_post(request);
```

## Key Design Decisions

### 1. HCCL API Mapping
- Direct mapping of UCC collective types to HCCL operations
- Transparent data type and reduction operation conversion
- Stream-based asynchronous execution model

### 2. Error Handling
- Comprehensive HCCL error code translation
- Asynchronous error detection support
- Graceful fallback and error reporting

### 3. Memory Management
- Pool-based task allocation for performance
- Proper cleanup of HCCL resources
- Support for different memory types

## Dependencies and Requirements

### 1. HCCL Library Requirements
- HCCL development headers (`hccl.h`, `hccl_types.h`)
- HCCL runtime library (`libhcl.so`)
- Habana device drivers and runtime

### 2. Build Requirements
- Autotools (autoconf, automake, libtool)
- C compiler with C11 support
- Standard UCC build dependencies

## Future Enhancements

### 1. Additional Collectives
- AllToAll, AllToAllV operations
- Reduce, ReduceScatter operations
- Gather, Scatter operations
- Custom collective patterns

### 2. Advanced Features
- Multi-stream support
- Memory type optimizations
- Performance tuning and scoring
- Advanced error recovery

### 3. Integration Improvements
- Better device detection
- Dynamic configuration
- Profiling and debugging support

## Testing and Validation

### 1. Build Testing
First, ensure the HCCL integration builds correctly:

```bash
# Clean and regenerate build files
./autogen.sh

# Configure with HCCL support
./configure --with-hccl=/path/to/hcl/installation \
            --with-tls=hccl,ucp,self \
            --enable-debug

# Build the project
make -j$(nproc)

# Verify HCCL TL was built
ls src/components/tl/hccl/.libs/libucc_tl_hccl.so
```

### 2. Component Verification
Check if HCCL TL is properly loaded:

```bash
# Check available TLs
./tools/info/ucc_info -t

# Check HCCL TL specifically
./tools/info/ucc_info -t hccl

# Verify configuration
UCC_LOG_LEVEL=DEBUG UCC_TL=hccl ./tools/info/ucc_info -c
```

### 3. Unit Tests with GTest
```bash
cd test/gtest

# Run HCCL-specific unit tests
./gtest --gtest_filter="*hccl*"

# Run core collective tests
./gtest --gtest_filter="*allreduce*:*allgather*:*barrier*:*bcast*"

# Run with debug logging
UCC_LOG_LEVEL=DEBUG UCC_TL_HCCL_LOG_LEVEL=TRACE ./gtest
```

### 4. MPI Integration Tests
```bash
cd test/mpi

# Build MPI tests
make

# Test AllReduce
mpirun -n 4 --mca coll_ucc_enable 1 \
             --mca coll_ucc_priority 100 \
             --mca pml ucx \
             ./test_allreduce

# Test AllGather
mpirun -n 4 --mca coll_ucc_enable 1 \
             --mca coll_ucc_priority 100 \
             ./test_allgather

# Test Broadcast
mpirun -n 4 --mca coll_ucc_enable 1 \
             --mca coll_ucc_priority 100 \
             ./test_bcast

# Test Barrier
mpirun -n 4 --mca coll_ucc_enable 1 \
             --mca coll_ucc_priority 100 \
             ./test_barrier
```

### 5. Performance Benchmarks
```bash
cd tools/perf

# Build performance tests
make

# AllReduce performance
mpirun -n 4 ./ucc_perftest -c allreduce -b 8:1M -i 100

# AllGather performance  
mpirun -n 4 ./ucc_perftest -c allgather -b 8:1M -i 100

# Barrier latency
mpirun -n 4 ./ucc_perftest -c barrier -i 1000

# Multiple collective comparison
mpirun -n 4 ./ucc_perftest -c allreduce,allgather,bcast,barrier -b 1K,1M
```

### 6. Device-Specific Testing
For systems with Habana devices:

```bash
# Check device availability
ls /dev/hl*

# Set device for testing
export HABANA_VISIBLE_DEVICES=0,1,2,3

# Run with device affinity
mpirun -n 4 --bind-to numa ./test_allreduce

# Multi-device testing
mpirun -n 8 --map-by ppr:2:node ./ucc_perftest -c allreduce
```

### 7. Error and Edge Case Testing
```bash
# Test with different message sizes
for size in 1 1K 1M 16M; do
    echo "Testing size: $size"
    mpirun -n 4 ./ucc_perftest -c allreduce -b $size -i 10
done

# Test with different team sizes
for np in 2 4 8 16; do
    echo "Testing with $np processes"
    mpirun -n $np ./test_allreduce
done

# Test error conditions
UCC_LOG_LEVEL=ERROR ./test_allreduce  # Should handle errors gracefully
```

### 8. Memory Type Testing
If supporting multiple memory types:

```bash
# Host memory (default)
mpirun -n 4 ./ucc_perftest -c allreduce -m host

# Test with different data types
mpirun -n 4 ./ucc_perftest -c allreduce -d float32,float64,int32,int64
```

### 9. Stress Testing
```bash
# Long-running test
mpirun -n 4 ./ucc_perftest -c allreduce -i 10000 -b 1M

# Mixed collective workload
mpirun -n 4 ./ucc_perftest -c allreduce,allgather,bcast -i 1000 -b 1K:1M

# Memory stress test
mpirun -n 4 ./ucc_perftest -c allreduce -b 100M -i 100
```

### 10. Debugging Failed Tests
```bash
# Enable comprehensive logging
export UCC_LOG_LEVEL=TRACE
export UCC_TL_HCCL_LOG_LEVEL=TRACE
export UCC_PROFILE_MODE=log
export UCC_PROFILE_FILE=/tmp/ucc_profile.log

# Run failing test with full debug
mpirun -n 4 ./test_allreduce

# Check logs
cat /tmp/ucc_profile.log

# Use GDB for crashes
mpirun -n 1 gdb --args ./test_allreduce
```

### 11. Validation Test Script
Create a comprehensive test script:

```bash
#!/bin/bash
# hccl_test.sh - HCCL Integration Test Suite

set -e

echo "=== HCCL Integration Test Suite ==="

# 1. Build verification
echo "1. Checking build..."
if [ ! -f "src/components/tl/hccl/.libs/libucc_tl_hccl.so" ]; then
    echo "ERROR: HCCL TL library not found"
    exit 1
fi

# 2. Component verification
echo "2. Checking TL availability..."
./tools/info/ucc_info -t | grep -q hccl || {
    echo "ERROR: HCCL TL not available"
    exit 1
}

# 3. Basic functionality tests
echo "3. Running basic collective tests..."
cd test/mpi
for coll in allreduce allgather bcast barrier; do
    echo "  Testing $coll..."
    mpirun -n 4 --mca coll_ucc_enable 1 ./test_$coll || {
        echo "ERROR: $coll test failed"
        exit 1
    }
done

# 4. Performance sanity check
echo "4. Running performance sanity check..."
cd ../../tools/perf
mpirun -n 4 ./ucc_perftest -c allreduce -b 1K -i 10 || {
    echo "ERROR: Performance test failed"
    exit 1
}

echo "=== All tests passed! ==="
```

### 12. Continuous Integration Testing
For automated testing environments:

```bash
# CI test script
#!/bin/bash
# ci_test_hccl.sh

export UCC_LOG_LEVEL=WARN
export MPIRUN_OPTIONS="--oversubscribe --allow-run-as-root"

# Quick smoke test
mpirun $MPIRUN_OPTIONS -n 2 ./test_allreduce
mpirun $MPIRUN_OPTIONS -n 2 ./test_barrier

# Medium test suite
for np in 2 4; do
    for coll in allreduce allgather bcast; do
        mpirun $MPIRUN_OPTIONS -n $np ./test_$coll
    done
done
```

## Troubleshooting

### 1. Common Issues
- **HCCL not found**: Ensure `--with-hccl` points to correct installation
- **Device initialization**: Verify Habana device drivers are loaded
- **Runtime errors**: Check HCCL library version compatibility

### 2. Debug Options
```bash
# Enable debug logging
export UCC_LOG_LEVEL=DEBUG
export UCC_TL_HCCL_LOG_LEVEL=TRACE
```

This integration provides a solid foundation for using HCCL within the UCC framework, enabling Habana devices to participate in the broader UCC ecosystem while maintaining performance and feature compatibility.
