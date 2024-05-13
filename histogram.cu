#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>

#define BLOCK_SIZE 256
#define NUM_BINS 101 //make sure that the number of bins is n+1, where n is the highest possible number in the input range


__global__ void histogram(const int* input, int size, float* histogram) {
    __shared__ int localHistogram[NUM_BINS];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    // Initialize local histogram to zeros
    if (tid < NUM_BINS)
        localHistogram[tid] = 0;
    
    __syncthreads();

    // Compute local histogram
    for (int i = index; i < size; i += blockDim.x * gridDim.x) {
        atomicAdd(&localHistogram[input[i]], 1);
        //localHistogram[input[i]]++;
    }

    __syncthreads();

    // Reduce local histograms into global histogram
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&histogram[i], localHistogram[i]);
    }

    __syncthreads();
    if(tid < NUM_BINS){
        histogram[tid] = static_cast<float>(histogram[tid])/size;
    }
}

int main() {
    const int N = 10; // Size of the input array
    int h_input[10] = {10, 20, 30, 40, 40, 50, 60, 71, 90, 100};
    float* h_histogram = new float[NUM_BINS];
    int* d_input;
    float* d_histogram;

    // Initialize input array with random integers
    // srand(time(nullptr));
    // for (int i = 0; i < N; ++i) {
    //     h_input[i] = rand() % NUM_BINS; // Random integer between 0 and NUM_BINS-1
    // }

    // Allocate device memory for input and histogram
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_histogram, NUM_BINS * sizeof(float));

    // Copy input array from host to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize histogram array to zeros
    cudaMemset(d_histogram, 0.0, NUM_BINS * sizeof(float));

    // Launch kernel to compute histogram
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    histogram<<<numBlocks, BLOCK_SIZE>>>(d_input, N, d_histogram);

    // Copy histogram from device to host
    cudaMemcpy(h_histogram, d_histogram, NUM_BINS * sizeof(float), cudaMemcpyDeviceToHost);

    // Print histogram
    std::cout << "Histogram:\n";
    for (int i = 0; i < NUM_BINS; ++i) {
        std::cout << "Bin " << i << ": " << h_histogram[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_histogram);

    // Free host memory
    //delete[] h_input;
    delete[] h_histogram;

    return 0;
}
