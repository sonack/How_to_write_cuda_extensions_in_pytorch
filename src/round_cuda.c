#include <THC/THC.h>
#include "round_cuda_kernel.h"

extern THCState *state;

int round_forward_wrapper(THCudaTensor *bottom_tensor, THCudaTensor *top_tensor, int count) {
    float *bottom_data = THCudaTensor_data(state, bottom_tensor);
    float *top_data = THCudaTensor_data(state, top_tensor);
    cudaStream_t stream = THCState_getCurrentStream(state);

    round_forward_cuda(bottom_data, top_data, count, stream);
    return 1;
}

int round_backward_wrapper(THCudaTensor *bottom_tensor, THCudaTensor *top_tensor, int count) {
    float *bottom_diff = THCudaTensor_data(state, bottom_tensor);
    float *top_diff = THCudaTensor_data(state, top_tensor);
    cudaStream_t stream = THCState_getCurrentStream(state);

    round_backward_cuda(top_diff, bottom_diff, count, stream);
    return 1;
}
