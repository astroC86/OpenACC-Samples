/*
  This code compares sum reductions between:
  1. Cooperative Groups multi-block kernel
  2. CUB's DeviceReduce::Sum algorithm

  It assumes the input size is a power of 2.

  COMMAND LINE ARGUMENTS

  "--n=<N>"         :Specify the number of elements to reduce (default 33554432)
  "--threads=<N>"   :Specify the number of threads per block (default 128)
  "--maxblocks=<N>" :Specify the maximum number of thread blocks to launch
 (kernel 6 only, default 64)
  "--iterations=<N>":Specify the number of test iterations (default 100)
*/

// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_functions.h"

// CUB includes
#include <cub/cub.cuh>

const char *sSDKsample = "reductionComparison";

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime_api.h>

namespace cg = cooperative_groups;

/*
  Parallel sum reduction using shared memory
  - takes log(n) steps for n input elements
  - uses n/2 threads
  - only works for power-of-2 arrays

  This version adds multiple elements per thread sequentially. This reduces the
  overall cost of the algorithm while keeping the work complexity O(n) and the
  step complexity O(log n).
  (Brent's Theorem optimization)

  See the CUDA SDK "reduction" sample for more information.
*/

__device__ void reduceBlock(double *sdata, const cg::thread_block &cta)
{
    const unsigned int        tid    = cta.thread_rank();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    sdata[tid] = cg::reduce(tile32, sdata[tid], cg::plus<double>());
    cg::sync(cta);

    double beta = 0.0;
    if (cta.thread_rank() == 0) {
        beta = 0;
        for (int i = 0; i < blockDim.x; i += tile32.size()) {
            beta += sdata[i];
        }
        sdata[0] = beta;
    }
    cg::sync(cta);
}

// This reduction kernel reduces an arbitrary size array in a single kernel
// invocation
//
// For more details on the reduction algorithm (notably the multi-pass
// approach), see the "reduction" sample in the CUDA SDK.
extern "C" __global__ void reduceSinglePassMultiBlockCG(const float *g_idata, float *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block block = cg::this_thread_block();
    cg::grid_group   grid  = cg::this_grid();

    extern double __shared__ sdata[];

    // Stride over grid and add the values to a shared memory buffer
    sdata[block.thread_rank()] = 0;

    for (int i = grid.thread_rank(); i < n; i += grid.size()) {
        sdata[block.thread_rank()] += g_idata[i];
    }

    cg::sync(block);

    // Reduce each block (called once per block)
    reduceBlock(sdata, block);
    // Write out the result to global memory
    if (block.thread_rank() == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
    cg::sync(grid);

    if (grid.thread_rank() == 0) {
        for (int block = 1; block < gridDim.x; block++) {
            g_odata[0] += g_odata[block];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for cooperative kernel launch
////////////////////////////////////////////////////////////////////////////////
void call_reduceSinglePassMultiBlockCG(int size, int threads, int numBlocks, float *d_idata, float *d_odata)
{
    int   smemSize     = threads * sizeof(double);
    void *kernelArgs[] = {
        (void *)&d_idata,
        (void *)&d_odata,
        (void *)&size,
    };

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);

    cudaLaunchCooperativeKernel((void *)reduceSinglePassMultiBlockCG, dimGrid, dimBlock, kernelArgs, smemSize, NULL);
    // check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");
}

////////////////////////////////////////////////////////////////////////////////
// CUB reduction wrapper
////////////////////////////////////////////////////////////////////////////////
float call_cubDeviceReduce(int size, float *d_idata, float *d_odata, void *d_temp_storage, size_t temp_storage_bytes)
{
    float result = 0.0f;
    
    // Perform the reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_idata, d_odata, size);
    
    // Copy result back to host
    cudaMemcpy(&result, d_odata, sizeof(float), cudaMemcpyDeviceToHost);
    
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Get temporary storage size for CUB
////////////////////////////////////////////////////////////////////////////////
size_t getCubTempStorageSize(int size, float *d_idata, float *d_odata)
{
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(NULL, temp_storage_bytes, d_idata, d_odata, size);
    return temp_storage_bytes;
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, int device);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    cudaDeviceProp deviceProp = {0};
    int            dev;

    printf("%s Starting...\n\n", sSDKsample);

    dev = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    
    printf("GPU: %s\n", deviceProp.name);
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Memory Clock Rate: %.2f GHz\n", deviceProp.memoryClockRate * 1e-6);
    printf("Memory Bus Width: %d bits\n", deviceProp.memoryBusWidth);
    printf("Peak Memory Bandwidth: %.2f GB/s\n\n", 
           2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6);
    
    if (!deviceProp.cooperativeLaunch) {
        printf("Selected GPU (%d) does not support Cooperative Kernel Launch, "
               "Only CUB reduction will be tested\n", dev);
    }

    bool bTestPassed = false;
    bTestPassed      = runTest(argc, argv, dev);

    exit(bTestPassed ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
template <class T> T reduceCPU(T *data, int size)
{
    T sum = data[0];
    T c   = (T)0.0;

    for (int i = 1; i < size; i++) {
        T y = data[i] - c;
        T t = sum + y;
        c   = (t - sum) - y;
        sum = t;
    }

    return sum;
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the reduction
// We set threads / block to the minimum of maxThreads and n/2.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    if (n == 1) {
        threads = 1;
        blocks  = 1;
    }
    else {
        checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, reduceSinglePassMultiBlockCG));
    }

    blocks = min(maxBlocks, blocks);
}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time for Cooperative Groups kernel.
////////////////////////////////////////////////////////////////////////////////
float benchmarkCooperativeReduce(int                 n,
                                int                 numThreads,
                                int                 numBlocks,
                                int                 testIterations,
                                StopWatchInterface *timer,
                                float              *d_idata,
                                float              *d_odata)
{
    float       gpu_result = 0;
    cudaError_t error;

    printf("Launching Cooperative Groups Multi-Block kernel\n");
    
    // Warm up
    call_reduceSinglePassMultiBlockCG(n, numThreads, numBlocks, d_idata, d_odata);
    cudaDeviceSynchronize();
    
    sdkResetTimer(&timer);
    for (int i = 0; i < testIterations; ++i) {
        sdkStartTimer(&timer);
        call_reduceSinglePassMultiBlockCG(n, numThreads, numBlocks, d_idata, d_odata);
        cudaDeviceSynchronize();
        sdkStopTimer(&timer);
    }

    // copy final sum from device to host
    error = cudaMemcpy(&gpu_result, d_odata, sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaErrors(error);

    return gpu_result;
}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time for CUB.
////////////////////////////////////////////////////////////////////////////////
float benchmarkCubReduce(int                 n,
                        int                 testIterations,
                        StopWatchInterface *timer,
                        float              *d_idata,
                        float              *d_odata,
                        void               *d_temp_storage,
                        size_t              temp_storage_bytes)
{
    float gpu_result = 0;

    printf("Launching CUB DeviceReduce::Sum\n");
    
    // Warm up
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_idata, d_odata, n);
    cudaDeviceSynchronize();
    
    sdkResetTimer(&timer);
    for (int i = 0; i < testIterations; ++i) {
        sdkStartTimer(&timer);
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_idata, d_odata, n);
        cudaDeviceSynchronize();
        sdkStopTimer(&timer);
    }

    // copy final sum from device to host
    cudaMemcpy(&gpu_result, d_odata, sizeof(float), cudaMemcpyDeviceToHost);

    return gpu_result;
}

////////////////////////////////////////////////////////////////////////////////
// The main function which runs the reduction test.
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, int device)
{
    int  size        = 1 << 25; // number of elements to reduce
    bool bTestPassed = true;
    int  testIterations = 100;

    if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
        size = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }
    
    if (checkCmdLineFlag(argc, (const char **)argv, "iterations")) {
        testIterations = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");
    }

    printf("Testing with %d elements\n", size);
    printf("Running %d iterations per test\n\n", testIterations);

    // Set the device to be used
    cudaDeviceProp prop = {0};
    checkCudaErrors(cudaSetDevice(device));
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));

    // create random input data on CPU
    unsigned int bytes = size * sizeof(float);

    float *h_idata = (float *)malloc(bytes);

    for (int i = 0; i < size; i++) {
        // Keep the numbers small so we don't get truncation error in the sum
        h_idata[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }

    // Determine the launch configuration (threads, blocks)
    int maxThreads = 0;
    int maxBlocks  = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "threads")) {
        maxThreads = getCmdLineArgumentInt(argc, (const char **)argv, "threads");
    }
    else {
        maxThreads = prop.maxThreadsPerBlock;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "maxblocks")) {
        maxBlocks = getCmdLineArgumentInt(argc, (const char **)argv, "maxblocks");
    }
    else {
        maxBlocks = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock);
    }

    int numBlocks  = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

    // We calculate the occupancy to know how many block can actually fit on the
    // GPU
    int numBlocksPerSm = 0;
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, reduceSinglePassMultiBlockCG, numThreads, numThreads * sizeof(double)));

    int numSms = prop.multiProcessorCount;
    if (numBlocks > numBlocksPerSm * numSms) {
        numBlocks = numBlocksPerSm * numSms;
    }
    
    printf("Configuration for Cooperative Groups kernel:\n");
    printf("  numThreads: %d\n", numThreads);
    printf("  numBlocks: %d\n\n", numBlocks);

    // allocate device memory and data
    float *d_idata = NULL;
    float *d_odata = NULL;
    void  *d_temp_storage = NULL;

    checkCudaErrors(cudaMalloc((void **)&d_idata, bytes));
    checkCudaErrors(cudaMalloc((void **)&d_odata, sizeof(float)));

    // Get CUB temporary storage size
    size_t temp_storage_bytes = getCubTempStorageSize(size, d_idata, d_odata);
    checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    printf("CUB temporary storage required: %.2f MB\n\n", temp_storage_bytes / (1024.0 * 1024.0));

    // copy data directly to device memory
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);

    float coop_result = 0;
    float cub_result = 0;
    float coop_time = 0;
    float cub_time = 0;
    
    // Test CUB reduction
    printf("=== CUB DeviceReduce::Sum Performance ===\n");
    cub_result = benchmarkCubReduce(size, testIterations, timer, d_idata, d_odata, d_temp_storage, temp_storage_bytes);
    cub_time = sdkGetAverageTimerValue(&timer);
    printf("Average time: %.4f ms\n", cub_time);
    printf("Bandwidth:    %.2f GB/s\n", (size * sizeof(float)) / (cub_time * 1.0e6));
    printf("Result:       %.12f\n\n", cub_result);

    // Test Cooperative Groups reduction (if supported)
    if (prop.cooperativeLaunch) {
        printf("=== Cooperative Groups Multi-Block Performance ===\n");
        coop_result = benchmarkCooperativeReduce(size, numThreads, numBlocks, testIterations, timer, d_idata, d_odata);
        coop_time = sdkGetAverageTimerValue(&timer);
        printf("Average time: %.4f ms\n", coop_time);
        printf("Bandwidth:    %.2f GB/s\n", (size * sizeof(float)) / (coop_time * 1.0e6));
        printf("Result:       %.12f\n\n", coop_result);
    }

    // compute reference solution
    float cpu_result = reduceCPU<float>(h_idata, size);
    printf("=== CPU Reference ===\n");
    printf("CPU result:   %.12f\n\n", cpu_result);

    // Performance comparison
    printf("=== Performance Comparison ===\n");
    if (prop.cooperativeLaunch) {
        printf("CUB time:           %.4f ms\n", cub_time);
        printf("Cooperative time:   %.4f ms\n", coop_time);
        printf("Speedup (CUB vs Coop): %.2fx %s\n", 
               coop_time / cub_time, 
               (cub_time < coop_time) ? "(CUB faster)" : "(Cooperative faster)");
        printf("CUB bandwidth:      %.2f GB/s\n", (size * sizeof(float)) / (cub_time * 1.0e6));
        printf("Cooperative bandwidth: %.2f GB/s\n\n", (size * sizeof(float)) / (coop_time * 1.0e6));
    }

    // Accuracy check
    printf("=== Accuracy Check ===\n");
    double threshold = 1e-8 * size;
    double cub_diff = abs((double)cub_result - (double)cpu_result);
    bool cub_passed = (cub_diff < threshold);
    
    printf("CUB accuracy:    %s (diff: %.2e, threshold: %.2e)\n", 
           cub_passed ? "PASS" : "FAIL", cub_diff, threshold);
    
    if (prop.cooperativeLaunch) {
        double coop_diff = abs((double)coop_result - (double)cpu_result);
        bool coop_passed = (coop_diff < threshold);
        printf("Coop accuracy:   %s (diff: %.2e, threshold: %.2e)\n", 
               coop_passed ? "PASS" : "FAIL", coop_diff, threshold);
        bTestPassed = cub_passed && coop_passed;
    } else {
        bTestPassed = cub_passed;
    }

    // cleanup
    sdkDeleteTimer(&timer);
    free(h_idata);
    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaFree(d_temp_storage);

    return bTestPassed;
}
