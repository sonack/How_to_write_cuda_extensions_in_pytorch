#ifndef _ROUND_CUDA_KERNEL
#define _ROUND_CUDA_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

void round_forward_cuda(float *bottom_data, float *top_data, int count, cudaStream_t stream);
void round_backward_cuda(float *top_diff, float *bottom_diff, int count, cudaStream_t stream);
#ifdef __cplusplus
}
#endif

#endif