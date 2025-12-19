#pragma once

/**
 * CUDA host/device annotation macros.
 *
 * These macros provide portable host/device annotations that compile to
 * nothing when not building with CUDA. This allows GPU headers to be
 * included in non-CUDA translation units without errors.
 *
 * Macros:
 *   VRTIGO_HOST_DEVICE - Functions callable from both host and device code
 *   VRTIGO_DEVICE      - Functions callable only from device code
 *   VRTIGO_HOST        - Functions callable only from host code
 *   VRTIGO_GLOBAL      - Kernel entry points (callable from host, execute on device)
 *
 * Detection macros:
 *   __CUDACC__     - Defined when compiling with nvcc (CUDA compiler driver)
 *   __CUDA_ARCH__  - Defined only during device compilation pass, indicates
 *                    the compute capability being compiled for (e.g., 750 for sm_75)
 *
 * Usage:
 *   VRTIGO_HOST_DEVICE T add(T a, T b) { return a + b; }
 *   VRTIGO_GLOBAL void kernel(T* data, size_t n) { ... }
 */

#ifdef __CUDACC__

    #define VRTIGO_HOST_DEVICE __host__ __device__
    #define VRTIGO_DEVICE      __device__
    #define VRTIGO_HOST        __host__
    #define VRTIGO_GLOBAL      __global__

#else

    #define VRTIGO_HOST_DEVICE
    #define VRTIGO_DEVICE
    #define VRTIGO_HOST
    #define VRTIGO_GLOBAL

#endif // __CUDACC__
