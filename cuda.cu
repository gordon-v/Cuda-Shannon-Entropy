#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define STREAM_SIZE 10000

// Device kernel for calculating entropy within a window on the stream
__global__ void shannon_entropy_window_kernel(const float* stream, float* entropies, int window_size, int start_index) {
    //printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = start_index + i;
    if (idx < STREAM_SIZE && idx >= start_index) {
        float p = stream[idx];
        float term = 0.0f;
        if (p > 0.0f) {
            term = -p * log2f(p);
        }
        atomicAdd(&entropies[i], term);
    }
}

int main(int argc, char* argv[]) {
    int window_size;
    
    cudaDeviceSynchronize();
    // Input validation (assuming command-line arguments for simplicity)
    if (argc != 2 || sscanf(argv[1], "%d", &window_size) != 1 || window_size <= 0) {
        std::cerr << "Usage: " << argv[0] << " <window_size>" << std::endl;
        return 1;
    }

    // Allocate memory on the host (CPU) for the stream
    float* stream_host = new float[STREAM_SIZE];

    // Initialize stream data (assuming random values for demonstration)
    for (int i = 0; i < STREAM_SIZE; ++i) {
        stream_host[i] = (float)rand() / RAND_MAX; // Random values between 0.0 and 1.0
    }
    // Allocate memory on the device (GPU) for the stream and entropies
    float* stream_device;
    float* entropies_device;
    cudaMalloc(&stream_device, STREAM_SIZE * sizeof(float));
    cudaMalloc(&entropies_device, (STREAM_SIZE - window_size + 1) * sizeof(float)); // Allocate for all possible window positions

    // Copy stream data from host to device
    cudaMemcpy(stream_device, stream_host, STREAM_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Sliding window processing loop
    for (int start_index = 0; start_index <= STREAM_SIZE - window_size; start_index+=window_size) {
        // Allocate memory on the device for temporary entropy results within a window
        float* temp_entropies_device;
        cudaMalloc(&temp_entropies_device, window_size * sizeof(float));

        // Launch the kernel for this window
        int blocks = (window_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        shannon_entropy_window_kernel<<<blocks, THREADS_PER_BLOCK>>>(stream_device, temp_entropies_device, window_size, start_index);

        // Error checking
        cudaDeviceSynchronize();

        // Copy temporary entropy results from device to host
        float* temp_entropies_host = new float[window_size];
        cudaMemcpy(temp_entropies_host, temp_entropies_device, window_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Calculate final entropy by summing temporary results
        float entropy = 0.0f;
        for (int i = 0; i < window_size; ++i) {
            entropy += temp_entropies_host[i];
        }

        // Process and potentially store entropy for this window (replace with your logic)
        std::cout << "Entropy for window starting at " << start_index << ": "<< entropy << std::endl;
    }

    return 0;
}