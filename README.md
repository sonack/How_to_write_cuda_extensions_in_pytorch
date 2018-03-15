# How_to_write_cuda_extensions_in_pytorch

**How to write cuda kernels or c functions in pytorch, especially for former caffe users.**

## Motivation
[Pytorch](https://github.com/pytorch/pytorch) is a very easy-using, flexible and extensible deep learning research framework, which could be enriched with new operations(modules/functions) accelerated by c or cuda code using the power of GPUs.

However, [the offical docs about how to create a C extension](http://pytorch.org/tutorials/advanced/c_extension.html) is so rough and not covering about cuda usage, and I wanted to transplant some new layers' `.cu` files of Caffe written in cuda kernel, but finding there was limited material describing it, so I created this repo in order to remind myself later or teach others on how to write custom cuda extensions in pytorch.

## Basic Steps

For simplicity, I will create a new `Differential Round Function` below, which is a elment-wise operation on any dimensional tensor whose any element's range is (0,1) (just think as after a sigmoid activation), and the forward calcuation formula is just like the ordinal round function (f(x) = 1, when 1 \> x \>=0.5 else f(x) = 0, when 0 \< x \< 0.5), however, it is not differential causing the BP algorithm can't work normally, so in the backward phase its gradient will be replaced with the gradient of the corresponding approximating function f(x) = x, x∈(0,1), i.e. gradient is 1 w.r.t the round function itself, so the gradient of input is just the same as the gradient of output. I will use a similarly caffe's `CUDA_KERNEL_LOOP` macro in pytorch's cuda file which iterate over each element position of the tensor.

#### 1. Environment Preparation

First, please refer to the [official docs about how to create a C extension](http://pytorch.org/tutorials/advanced/c_extension.html), and run the demo to check if your environment is ready, maybe you need to install the cffi python package and the python-dev package to continue, you can use the following commands to install.

```shell
sudo apt install python-dev # for python2 users or python3-dev for python3
pip install cffi # may need sudo, or pip2 for explicitly installing for python2
```


#### 2. Prepare your directories and files structrue

Firstly, I made a new directory named `round_cuda`, which will contain all the code we need today and created some new files, the structure is below:

    ➜  round_cuda git:(master) ✗ tree .
    .
    ├── build               # the dir will contain the built .so file
    ├── build_ffi.py        # which is the python script to build the extension
    ├── include             
    │   ├── caffe_cuda_macro.h  # contain the same caffe cuda MACRO for easy use
    │   ├── round_cuda.h        # declare the C functions which pytorch can communicate with
    │   └── round_cuda_kernel.h # declare cuda kernel functions which will be called in the above C functions
    ├── Makefile    # there is a template file later you can modify even if you know nothing about Makefile
    ├── src
    │   ├── round_cuda.c    # the round_cuda.h declared functions' implementation
    │   └── round_cuda_kernel.cu    # the round_cuda_kernel.h kernel wrapper functions and the kernels' implementation
    └── test.py     # testing script

    3 directories, 8 files

#### 3. Write the header files

  * caffe_cuda_macro.h

    You don't need to modify it mostly, it defines the similar Caffe Cuda Macros, such as `CUDA_KERNEL_LOOP` and `CUDA_POST_KERNEL_CHECK`.

  * round_cuda.h

    The C functions that pytorch can directly communicate with is declared here, I declared 2 functions here, one for forward pass and one for backward, the parameters type is `ThCudaTensor *` for input and output tensors, and the last `count` is the total number of elements of the tensor, because there is no shape information in the `ThCudaTensor *` parameter.

    ```c++
    int round_forward_wrapper(THCudaTensor *bottom_tensor, THCudaTensor *top_tensor, int count);
    int round_backward_wrapper(THCudaTensor *bottom_tensor, THCudaTensor *top_tensor, int count);
    ```

    Notably, the return value's type is `int`, and return **1** rather than 0 when finished normally.

    The functions declared will be implemented in the following file `src/round_cuda.c`.

  * round_cuda_kernel.h

    The cuda wrapper function's declaration is in this file, the parameters is similar to `round_cuda.h`'s' C functions, but the data type is changed from `ThCudaTensor *` to the real data `float *`, and there is always a `stream` parameter which capsuled the  Cuda calculation position for pytorch to find it.

    What's more, these declarations must be wrapped in an `extern C` structure because they are C++ functions which wrap the kernel functions waited to be called in C functions.

    These wrapper functions will be called in the `round_cuda.h` C functions.

#### 4. Write the main code

  * round_cuda.c

    The file implements the C functions.

    First, you can use `extern THCState *state` to refer the pytorch state, which will be passed as a parameter later always.

    Next, you need to use the API `THCudaTensor_data` to get the float pointer from a `THCudaTensor *`, and `THCState_getCurrentStream` to get current Cuda stream, then pass them to the `round_cuda_kernel.h` declared cuda wrapper functions.

    Finally, don't forget to return 1 to be safe.

  * round_cuda_kernel.cu

    Finally, we reach to the `.cu` file, but at the beginning, you need to include `caffe_cuda_macro.h` to use Caffe's macro and `round_cuda_kernel.h`.

    Then, define your own `__global__` kernel's and wrap them in the `round_cuda_kernel.h` declared cuda wrapper functions, they should be passed the float pointer of your tensors and other parameters like the current pytorch cuda stream, **YOU CAN USE CAFFE MACRO STYLE TO WRITE YOUR OWN KERNEL**, for example:

    ```c++
    __global__ void round_forward_cuda_kernel(const int n, float *bottom_data, float *top_data) {
        CUDA_KERNEL_LOOP(i, n) {
          top_data[i] = bottom_data[i] > 0.5 ? 1 : 0;
        }
    }
    ```

    and call above kernel just like:

    ```c++
    void round_forward_cuda(float *bottom_data, float *top_data, int count, cudaStream_t stream)
    {
        round_forward_cuda_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(count, bottom_data, top_data);
        CUDA_POST_KERNEL_CHECK;
    }
    ```


#### 5. Write python building script using pytorch ffi interface and Makefile
You can regard them as template scripts and always replace `round` with your filename at the corresponding positions, it isn't hard.

#### 6. Just type `make` in the root directory, then you will have a directory named `round`, the generated package directory's name and structure is determined by the function `create_extension`'s first parameter in file `build_ffi.py`.

#### 7. You can write a test script to check your code, and encapsule it with Pytorch's `Function` and `Module` for later use.


That's all, thanks for your attention. If you still get a bit confused, don't hesitate to open an issue for discussing with me.

Enjoy your using of pytorch with Cuda accelerating!


**References:**

1. [https://github.com/chrischoy/pytorch-custom-cuda-tutorial]
2. [https://github.com/pytorch/extension-ffi]
3. [https://github.com/sniklaus/pytorch-extension]
4. [http://pytorch.org/tutorials/advanced/c_extension.html]
