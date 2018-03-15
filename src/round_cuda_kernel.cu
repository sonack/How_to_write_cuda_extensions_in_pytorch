#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "caffe_cuda_macro.h"
#include "round_cuda_kernel.h"

__global__ void round_forward_cuda_kernel(const int n, float *bottom_data, float *top_data) {
    CUDA_KERNEL_LOOP(i, n) {
        top_data[i] = bottom_data[i] > 0.5 ? 1 : 0;
    }
}


__global__ void round_backward_cuda_kernel(const int n, float *top_diff, float *bottom_diff) {
    CUDA_KERNEL_LOOP(i, n) {
        bottom_diff[i] = top_diff[i];
    }
}


void round_forward_cuda(float *bottom_data, float *top_data, int count, cudaStream_t stream)
{
    round_forward_cuda_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(count, bottom_data, top_data);
    CUDA_POST_KERNEL_CHECK;
}


void round_backward_cuda(float *top_diff, float *bottom_diff, int count, cudaStream_t stream)
{
    round_backward_cuda_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(count, top_diff, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
}