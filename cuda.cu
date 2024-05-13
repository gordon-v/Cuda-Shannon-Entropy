#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define STREAM_SIZE 10 //up to size of input file
#define BLOCK_SIZE 256
#define NUM_BINS 257 //for histogram


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

// Function to load random numbers from a file and initialize an array
void loadArrayFromFile(int* arr, int N, const std::string& filename) {
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        std::cerr << "Error: Unable to open file." << std::endl;
        return;
    }

    // Load numbers from file into array
    for (int i = 0; i < N; ++i) {
        inFile >> arr[i];
    }

    inFile.close();
    std::cout << "Array loaded from file: " << filename << std::endl;
}


// Device kernel for calculating entropy within a window on the stream
__global__ void shannon_entropy_window_kernel(const float* stream, float* entropies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < STREAM_SIZE) {
        float p = stream[i];
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
    int* stream_host = new int[STREAM_SIZE];

    // Initialize stream data by loading from file
   
    loadArrayFromFile(stream_host, STREAM_SIZE, "shannon-input.txt");
    
    // Allocate memory on the device (GPU) for the stream and entropies
    float* stream_device;
    float* entropies_device;

    //TODO: Here only a size of window_size is needed
    // cudaMalloc(&stream_device, STREAM_SIZE * sizeof(float));                        
    // cudaMalloc(&entropies_device, (STREAM_SIZE - window_size + 1) * sizeof(float)); // Allocate for all possible window positions

    // // Copy stream data from host to device
    // cudaMemcpy(stream_device, stream_host, STREAM_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Sliding window processing loop
    for (int start_index = 0; start_index <= STREAM_SIZE - window_size; start_index++) {
        //Allocate memory on the device for calculating a histogram of the window
        float* h_histogram = new float[NUM_BINS]; //for returning results to host
        int* h_input = new int [window_size]; // for storing window of input stream on host
        h_input = &stream_host[start_index];
        
        // Allocate memory on the device for temporary entropy results within a window
        int* d_input;
        float* d_histogram;

        // Allocate device memory for input and histogram
        cudaMalloc(&d_input, window_size * sizeof(int));
        cudaMalloc(&d_histogram, NUM_BINS * sizeof(float));


        // Copy input array from host to device
        cudaMemcpy(d_input, h_input, window_size * sizeof(int), cudaMemcpyHostToDevice);
        
        // Initialize histogram array to zeros
        cudaMemset(d_histogram, 0.0, NUM_BINS * sizeof(float));

        
        // Launch kernel to compute histogram
        int numBlocks = (window_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        histogram<<<numBlocks, BLOCK_SIZE>>>(d_input, window_size, d_histogram);

        // Copy histogram from device to host
        cudaMemcpy(h_histogram, d_histogram, NUM_BINS * sizeof(float), cudaMemcpyDeviceToHost);

        // Print histogram
        printf("Histogram:\n");
        for (int i = 0; i < NUM_BINS; ++i) {
            std::cout << "Bin " << i << ": " << h_histogram[i] << std::endl;
        }

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_histogram);
        // Free host memory
        delete[] h_input;
        delete[] h_histogram;
        
//////////////////////////////////////////

        // // Launch the kernel for this window
        // int blocks = (window_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        // shannon_entropy_window_kernel<<<blocks, THREADS_PER_BLOCK>>>(stream_device, temp_entropies_device, window_size, start_index);

        // // Error checking
        // cudaDeviceSynchronize();

        // // Copy temporary entropy results from device to host
        // float* temp_entropies_host = new float[window_size];
        // cudaMemcpy(temp_entropies_host, temp_entropies_device, window_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // // Calculate final entropy by summing temporary results
        // float entropy = 0.0f;
        // for (int i = 0; i < window_size; ++i) {
        //     entropy += temp_entropies_host[i];
        // }

        // // Process and potentially store entropy for this window (replace with your logic)
        // std::cout << "Entropy for window starting at " << start_index << ": "<< entropy << std::endl;
    }

    return 0;
}