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

### 1. Unit Tests
```bash
# Run UCC tests with HCCL backend
make check UCC_TL=hccl
```

### 2. Integration Tests
```bash
# MPI integration test
mpirun -n 4 --mca coll_ucc_enable 1 --mca coll_ucc_priority 100 ./test_allreduce
```

### 3. Performance Benchmarks
```bash
# Performance testing
./ucc_perftest -c allreduce -b hccl -n 1000 -s 1024
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
