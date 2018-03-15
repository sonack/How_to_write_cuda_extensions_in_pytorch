#ifndef _CAFFE_CUDA_MACRO
#define _CAFFE_CUDA_MACRO

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK \
do { \
  cudaError_t err = cudaGetLastError(); \
  if (cudaSuccess != err) \
      fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err)); \
} while (0)

#endif