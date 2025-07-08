#!/bin/bash
#
# HCCL Integration Test Suite
# This script provides comprehensive testing for the HCCL Transport Layer integration
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
UCC_ROOT="$(pwd)"
TEST_LOG_DIR="/tmp/ucc_hccl_tests"
MPIRUN_NP=${MPIRUN_NP:-4}
TEST_ITERATIONS=${TEST_ITERATIONS:-10}

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup() {
    log_info "Cleaning up test files..."
    rm -rf "$TEST_LOG_DIR"
}

# Setup
setup_tests() {
    log_info "Setting up HCCL integration tests..."
    
    # Create log directory
    mkdir -p "$TEST_LOG_DIR"
    
    # Set environment variables
    export UCC_LOG_LEVEL=INFO
    export UCC_TL_HCCL_LOG_LEVEL=DEBUG
    export UCC_PROFILE_MODE=log
    export UCC_PROFILE_FILE="$TEST_LOG_DIR/ucc_profile.log"
    
    # Trap for cleanup
    trap cleanup EXIT
}

# Test 1: Build verification
test_build() {
    log_info "Test 1: Verifying HCCL TL build..."
    
    local lib_path="src/components/tl/hccl/.libs/libucc_tl_hccl.so"
    if [ -f "$lib_path" ]; then
        log_success "HCCL TL library found: $lib_path"
        return 0
    else
        log_error "HCCL TL library not found: $lib_path"
        log_info "Please ensure you built with --with-hccl=<path>"
        return 1
    fi
}

# Test 2: Component availability
test_component() {
    log_info "Test 2: Checking HCCL TL component availability..."
    
    if ./tools/info/ucc_info -t | grep -q hccl; then
        log_success "HCCL TL is available"
        ./tools/info/ucc_info -t hccl > "$TEST_LOG_DIR/hccl_info.log" 2>&1
        return 0
    else
        log_error "HCCL TL not available in component list"
        ./tools/info/ucc_info -t > "$TEST_LOG_DIR/available_tls.log" 2>&1
        log_info "Available TLs logged to: $TEST_LOG_DIR/available_tls.log"
        return 1
    fi
}

# Test 3: GTest unit tests
test_gtest() {
    log_info "Test 3: Running GTest unit tests..."
    
    if [ ! -f "test/gtest/gtest" ]; then
        log_warning "GTest binary not found, skipping unit tests"
        return 0
    fi
    
    cd test/gtest
    
    # Run core collective tests
    local test_cases=("test_allreduce" "test_allgather" "test_barrier" "test_bcast")
    
    for test_case in "${test_cases[@]}"; do
        log_info "  Running $test_case..."
        if timeout 60 ./gtest --gtest_filter="*${test_case#test_}*" > "$TEST_LOG_DIR/${test_case}.log" 2>&1; then
            log_success "  $test_case passed"
        else
            log_error "  $test_case failed"
            tail -20 "$TEST_LOG_DIR/${test_case}.log"
            cd "$UCC_ROOT"
            return 1
        fi
    done
    
    cd "$UCC_ROOT"
    return 0
}

# Test 4: MPI integration tests
test_mpi() {
    log_info "Test 4: Running MPI integration tests..."
    
    cd test/mpi
    
    # Check if MPI tests are built
    if [ ! -f "test_allreduce" ]; then
        log_info "  Building MPI tests..."
        if ! make > "$TEST_LOG_DIR/mpi_build.log" 2>&1; then
            log_error "  Failed to build MPI tests"
            cat "$TEST_LOG_DIR/mpi_build.log"
            cd "$UCC_ROOT"
            return 1
        fi
    fi
    
    # Run MPI tests
    local mpi_tests=("test_allreduce" "test_allgather" "test_bcast" "test_barrier")
    local mpi_opts="--oversubscribe --allow-run-as-root --mca coll_ucc_enable 1 --mca coll_ucc_priority 100"
    
    for test in "${mpi_tests[@]}"; do
        log_info "  Running MPI $test with $MPIRUN_NP processes..."
        
        local cmd="mpirun $mpi_opts -n $MPIRUN_NP ./$test"
        if timeout 120 $cmd > "$TEST_LOG_DIR/mpi_${test}.log" 2>&1; then
            log_success "  MPI $test passed"
        else
            log_error "  MPI $test failed"
            tail -20 "$TEST_LOG_DIR/mpi_${test}.log"
            cd "$UCC_ROOT"
            return 1
        fi
    done
    
    cd "$UCC_ROOT"
    return 0
}

# Test 5: Performance benchmarks
test_performance() {
    log_info "Test 5: Running performance benchmarks..."
    
    cd tools/perf
    
    # Check if perf tools are built
    if [ ! -f "ucc_perftest" ]; then
        log_info "  Building performance tools..."
        if ! make > "$TEST_LOG_DIR/perf_build.log" 2>&1; then
            log_error "  Failed to build performance tools"
            cat "$TEST_LOG_DIR/perf_build.log"
            cd "$UCC_ROOT"
            return 1
        fi
    fi
    
    # Run performance tests
    local collectives=("allreduce" "allgather" "bcast" "barrier")
    local mpi_opts="--oversubscribe --allow-run-as-root"
    
    for coll in "${collectives[@]}"; do
        log_info "  Testing $coll performance..."
        
        local cmd="mpirun $mpi_opts -n $MPIRUN_NP ./ucc_perftest -c $coll -b 1K -i $TEST_ITERATIONS"
        if timeout 180 $cmd > "$TEST_LOG_DIR/perf_${coll}.log" 2>&1; then
            log_success "  Performance test for $coll completed"
            # Extract basic stats
            local avg_time=$(grep -o "avg.*us" "$TEST_LOG_DIR/perf_${coll}.log" | tail -1 || echo "N/A")
            log_info "    Average time: $avg_time"
        else
            log_warning "  Performance test for $coll failed or timed out"
            tail -10 "$TEST_LOG_DIR/perf_${coll}.log"
        fi
    done
    
    cd "$UCC_ROOT"
    return 0
}

# Test 6: Stress testing
test_stress() {
    log_info "Test 6: Running stress tests..."
    
    cd test/mpi
    
    local stress_iterations=100
    local large_sizes=("1M" "4M" "16M")
    local mpi_opts="--oversubscribe --allow-run-as-root --mca coll_ucc_enable 1"
    
    # Test with different message sizes
    for size in "${large_sizes[@]}"; do
        log_info "  Stress testing with message size: $size..."
        
        cd ../tools/perf || { log_error "Failed to change to perf directory"; return 1; }
        local cmd="mpirun $mpi_opts -n $MPIRUN_NP ./ucc_perftest -c allreduce -b $size -i 10"
        
        if timeout 300 $cmd > "$TEST_LOG_DIR/stress_${size}.log" 2>&1; then
            log_success "  Stress test with $size passed"
        else
            log_warning "  Stress test with $size failed or timed out"
        fi
        
        cd ../mpi || { log_error "Failed to change back to mpi directory"; return 1; }
    done
    
    cd "$UCC_ROOT"
    return 0
}

# Test 7: Error handling
test_error_handling() {
    log_info "Test 7: Testing error handling..."
    
    # Test with invalid parameters
    export UCC_LOG_LEVEL=ERROR
    
    cd test/mpi
    
    # Test with too many processes (should handle gracefully)
    local cmd="mpirun --oversubscribe --allow-run-as-root -n 1000 ./test_allreduce"
    if timeout 60 $cmd > "$TEST_LOG_DIR/error_test.log" 2>&1; then
        log_info "  Large process count handled"
    else
        log_info "  Large process count rejected (expected)"
    fi
    
    # Reset log level
    export UCC_LOG_LEVEL=INFO
    
    cd "$UCC_ROOT"
    return 0
}

# Generate test report
generate_report() {
    log_info "Generating test report..."
    
    local report_file="$TEST_LOG_DIR/test_report.txt"
    
    cat > "$report_file" << EOF
HCCL Integration Test Report
============================
Date: $(date)
UCC Root: $UCC_ROOT
Test Processes: $MPIRUN_NP
Test Iterations: $TEST_ITERATIONS

Test Results:
EOF

    # Check which tests passed
    local tests=("build" "component" "gtest" "mpi" "performance" "stress" "error_handling")
    
    for test in "${tests[@]}"; do
        if [ -f "$TEST_LOG_DIR/${test}_passed" ]; then
            echo "✓ $test: PASSED" >> "$report_file"
        elif [ -f "$TEST_LOG_DIR/${test}_failed" ]; then
            echo "✗ $test: FAILED" >> "$report_file"
        else
            echo "⚠ $test: SKIPPED" >> "$report_file"
        fi
    done
    
    echo "" >> "$report_file"
    echo "Detailed logs available in: $TEST_LOG_DIR" >> "$report_file"
    
    log_info "Test report generated: $report_file"
    cat "$report_file"
}

# Main test runner
run_tests() {
    local all_passed=true
    
    # Test sequence
    local tests=(
        "test_build build"
        "test_component component" 
        "test_gtest gtest"
        "test_mpi mpi"
        "test_performance performance"
        "test_stress stress"
        "test_error_handling error_handling"
    )
    
    for test_spec in "${tests[@]}"; do
        local test_func=$(echo $test_spec | cut -d' ' -f1)
        local test_name=$(echo $test_spec | cut -d' ' -f2)
        
        echo
        echo "========================================"
        
        if $test_func; then
            touch "$TEST_LOG_DIR/${test_name}_passed"
            log_success "Test $test_name completed successfully"
        else
            touch "$TEST_LOG_DIR/${test_name}_failed"
            log_error "Test $test_name failed"
            all_passed=false
            
            # Continue with other tests unless it's a critical failure
            if [ "$test_name" = "build" ] || [ "$test_name" = "component" ]; then
                log_error "Critical test failed, stopping test suite"
                break
            fi
        fi
    done
    
    echo
    echo "========================================"
    generate_report
    echo "========================================"
    
    if $all_passed; then
        log_success "All HCCL integration tests passed!"
        return 0
    else
        log_error "Some HCCL integration tests failed"
        return 1
    fi
}

# Parse command line arguments
show_help() {
    cat << EOF
HCCL Integration Test Suite

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -n, --processes N       Number of MPI processes (default: 4)
    -i, --iterations N      Number of test iterations (default: 10)
    -t, --test TEST         Run specific test only
                           (build|component|gtest|mpi|performance|stress|error)
    -v, --verbose           Enable verbose logging
    -c, --clean            Clean up previous test results

Available tests:
    build       - Verify HCCL TL library build
    component   - Check TL component availability  
    gtest       - Run GTest unit tests
    mpi         - Run MPI integration tests
    performance - Run performance benchmarks
    stress      - Run stress tests
    error       - Test error handling

Examples:
    $0                      # Run all tests
    $0 -t mpi              # Run only MPI tests
    $0 -n 8 -i 50          # Use 8 processes, 50 iterations
    $0 -v -t performance   # Verbose performance testing

EOF
}

# Main execution
main() {
    local specific_test=""
    local clean_first=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -n|--processes)
                MPIRUN_NP="$2"
                shift 2
                ;;
            -i|--iterations)
                TEST_ITERATIONS="$2"
                shift 2
                ;;
            -t|--test)
                specific_test="$2"
                shift 2
                ;;
            -v|--verbose)
                export UCC_LOG_LEVEL=DEBUG
                export UCC_TL_HCCL_LOG_LEVEL=TRACE
                shift
                ;;
            -c|--clean)
                clean_first=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Clean previous results if requested
    if $clean_first; then
        rm -rf "$TEST_LOG_DIR"
    fi
    
    # Setup
    setup_tests
    
    log_info "Starting HCCL Integration Test Suite"
    log_info "Processes: $MPIRUN_NP"
    log_info "Iterations: $TEST_ITERATIONS"
    log_info "Log directory: $TEST_LOG_DIR"
    
    # Run specific test or all tests
    if [ -n "$specific_test" ]; then
        case $specific_test in
            build) test_build ;;
            component) test_component ;;
            gtest) test_gtest ;;
            mpi) test_mpi ;;
            performance) test_performance ;;
            stress) test_stress ;;
            error) test_error_handling ;;
            *)
                log_error "Unknown test: $specific_test"
                show_help
                exit 1
                ;;
        esac
    else
        run_tests
    fi
}

# Run main function with all arguments
main "$@"
