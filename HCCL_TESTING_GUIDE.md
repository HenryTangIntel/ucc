# HCCL Integration Testing Guide

## Overview
This document provides a comprehensive testing strategy for the HCCL Transport Layer integration into UCC (Unified Collective Communication). The testing approach covers build verification, functional testing, performance benchmarking, and stress testing.

## Quick Start

### 1. Basic Testing
```bash
# Run the complete test suite
./test_hccl_integration.sh

# Or use the makefile
make -f Makefile.hccl-test test-hccl
```

### 2. Quick Smoke Test
```bash
# Fast validation (2 processes, basic tests)
make -f Makefile.hccl-test test-smoke
```

### 3. Specific Test Categories
```bash
# Build verification only
./test_hccl_integration.sh -t build

# MPI integration tests only  
./test_hccl_integration.sh -t mpi

# Performance benchmarks only
./test_hccl_integration.sh -t performance
```

## Test Architecture

### Test Categories

1. **Build Tests** (`test-build`)
   - Verify HCCL TL library compilation
   - Check library file existence
   - Validate build configuration

2. **Component Tests** (`test-component`)
   - Check TL registration and availability
   - Verify component initialization
   - Test configuration parsing

3. **Unit Tests** (`test-gtest`)
   - Run GTest framework tests
   - Test individual functions
   - Validate error conditions

4. **MPI Integration Tests** (`test-mpi`)
   - AllReduce collective operations
   - AllGather collective operations
   - Broadcast operations
   - Barrier synchronization

5. **Performance Tests** (`test-perf`)
   - Latency measurements
   - Bandwidth benchmarks
   - Scalability testing
   - Comparison with other TLs

6. **Stress Tests** (`test-stress`)
   - Large message sizes
   - Long-running operations
   - High iteration counts
   - Memory pressure testing

7. **Error Handling Tests** (`test-error`)
   - Invalid parameter handling
   - Resource exhaustion scenarios
   - Graceful degradation

## Testing Tools

### 1. Main Test Script
**File**: `test_hccl_integration.sh`
- Comprehensive test runner
- Colored output and logging
- Configurable parameters
- Detailed reporting

**Usage**:
```bash
./test_hccl_integration.sh [OPTIONS]

Options:
  -n, --processes N       MPI processes (default: 4)
  -i, --iterations N      Test iterations (default: 10)
  -t, --test TEST         Specific test category
  -v, --verbose           Enable verbose logging
  -c, --clean            Clean previous results
```

### 2. Test Makefile  
**File**: `Makefile.hccl-test`
- Convenient test targets
- Parameterized testing
- CI/CD integration
- Environment setup

**Key Targets**:
```bash
make -f Makefile.hccl-test test-hccl      # Full suite
make -f Makefile.hccl-test test-smoke     # Quick test
make -f Makefile.hccl-test test-mpi       # MPI tests
make -f Makefile.hccl-test check-hccl     # Availability check
```

### 3. UCC Testing Framework
- **GTest**: C++ unit testing framework
- **MPI Tests**: Integration with MPI collectives
- **Performance Tools**: ucc_perftest utility
- **Info Tools**: Component introspection

## Test Environment Setup

### Prerequisites
```bash
# 1. Build UCC with HCCL support
./autogen.sh
./configure --with-hccl=/path/to/hcl --with-tls=hccl,ucp,self
make -j$(nproc)

# 2. Ensure MPI is available
which mpirun

# 3. Set environment variables
export UCC_TL=hccl,ucp,self
export UCC_LOG_LEVEL=INFO
```

### Hardware Requirements
- Habana devices (for device-specific testing)
- Multiple nodes/processes for scalability testing
- Sufficient memory for large message testing

## Test Execution Workflow

### 1. Pre-test Validation
```bash
# Check build status
make -f Makefile.hccl-test check-hccl

# Verify environment
make -f Makefile.hccl-test debug-info
```

### 2. Progressive Testing
```bash
# Step 1: Build verification
./test_hccl_integration.sh -t build

# Step 2: Component availability
./test_hccl_integration.sh -t component

# Step 3: Basic functionality
./test_hccl_integration.sh -t mpi

# Step 4: Performance validation
./test_hccl_integration.sh -t performance
```

### 3. Full Test Suite
```bash
# Run all tests with custom parameters
./test_hccl_integration.sh -n 8 -i 50 -v
```

## Expected Test Results

### Build Tests
- ✅ HCCL TL library exists: `src/components/tl/hccl/.libs/libucc_tl_hccl.so`
- ✅ Component registration successful
- ✅ Configuration files processed

### Functional Tests
- ✅ AllReduce: Correct result computation
- ✅ AllGather: Proper data gathering
- ✅ Broadcast: Data distribution
- ✅ Barrier: Synchronization

### Performance Tests
- ✅ Latency: < 10µs for small messages
- ✅ Bandwidth: > 1GB/s for large messages
- ✅ Scalability: Linear with process count

## Troubleshooting

### Common Issues

1. **Library Not Found**
```bash
ERROR: HCCL TL library not found
Solution: Rebuild with --with-hccl=<path>
```

2. **Component Not Available**
```bash
ERROR: HCCL TL not available
Solution: Check LD_LIBRARY_PATH and UCC_TL settings
```

3. **MPI Test Failures**
```bash
ERROR: MPI test failed
Solution: Check MPI installation and process limits
```

4. **Performance Issues**
```bash
WARNING: Poor performance detected
Solution: Check device availability and configuration
```

### Debug Commands
```bash
# Enable debug logging
export UCC_LOG_LEVEL=TRACE
export UCC_TL_HCCL_LOG_LEVEL=TRACE

# Run with GDB
mpirun -n 1 gdb --args ./test_allreduce

# Check device status
ls -la /dev/hl*

# Verify library linking
ldd src/components/tl/hccl/.libs/libucc_tl_hccl.so
```

## Continuous Integration

### CI Script Template
```bash
#!/bin/bash
# ci_hccl_test.sh

set -e

# Build with HCCL
./autogen.sh
./configure --with-hccl=$HCCL_PATH --enable-debug
make -j$(nproc)

# Run CI-friendly tests
./test_hccl_integration.sh -n 2 -i 5 --clean

# Generate test report
echo "HCCL Integration: PASSED" > test_results.txt
```

### GitHub Actions Integration
```yaml
- name: Test HCCL Integration
  run: |
    ./test_hccl_integration.sh --clean
    make -f Makefile.hccl-test test-ci
```

## Performance Benchmarking

### Standard Benchmarks
```bash
# Latency test
mpirun -n 2 ./ucc_perftest -c barrier -i 1000

# Bandwidth test
mpirun -n 4 ./ucc_perftest -c allreduce -b 1M -i 100

# Scalability test
for np in 2 4 8 16; do
  mpirun -n $np ./ucc_perftest -c allreduce -b 1K
done
```

### Performance Targets
- **Barrier Latency**: < 5µs
- **AllReduce 1KB**: < 20µs
- **AllReduce 1MB**: > 500MB/s effective bandwidth
- **Scalability**: 90% efficiency up to 16 processes

## Test Reporting

### Automated Reports
The test script generates detailed reports in `/tmp/ucc_hccl_tests/`:
- `test_report.txt`: Summary of all test results
- Individual logs for each test category
- Performance statistics and timing data

### Manual Verification
```bash
# Check test results
cat /tmp/ucc_hccl_tests/test_report.txt

# Review detailed logs
ls /tmp/ucc_hccl_tests/*.log

# Analyze performance data
grep "avg" /tmp/ucc_hccl_tests/perf_*.log
```

## Integration with Existing Workflows

### Development Testing
```bash
# After code changes
make clean && make
./test_hccl_integration.sh -t build -t mpi
```

### Release Testing
```bash
# Full validation before release
./test_hccl_integration.sh -v
make -f Makefile.hccl-test test-stress
```

### Regression Testing
```bash
# Regular validation
./test_hccl_integration.sh > nightly_test.log 2>&1
```

This comprehensive testing strategy ensures robust validation of the HCCL integration, covering all aspects from basic functionality to performance characteristics and error handling.
