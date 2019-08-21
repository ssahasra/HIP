#include <amd_hostcall.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

#include "common.h"

static void
callee(ulong *output, ulong *input)
{
    output[0] = input[0] + 1;
    output[1] = input[1] + input[2];
}

__global__ void
kernel(ulong fptr, ulong *retval0, ulong *retval1)
{
    uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    ulong arg0 = (ulong)fptr;
    ulong arg1 = tid;
    ulong arg2 = 42;
    ulong arg3 = tid % 23;
    ulong arg4 = 0;
    ulong arg5 = 0;
    ulong arg6 = 0;
    ulong arg7 = 0;

    long2 result = {0, 0};
    if (tid % 71 != 1) {
        result.data = __ockl_call_host_function(arg0, arg1, arg2, arg3, arg4,
                                                arg5, arg6, arg7);
        retval0[tid] = result.x;
        retval1[tid] = result.y;
    }
}

static void
test()
{
    uint num_blocks = 5;
    uint threads_per_block = 1000;
    uint num_threads = num_blocks * threads_per_block;

    void *retval0_void;
    HIPCHECK(hipHostMalloc(&retval0_void, 8 * num_threads));
    uint64_t *retval0 = (uint64_t *)retval0_void;
    for (uint i = 0; i != num_threads; ++i) {
        retval0[i] = 0x23232323;
    }

    void *retval1_void;
    HIPCHECK(hipHostMalloc(&retval1_void, 8 * num_threads));
    uint64_t *retval1 = (uint64_t *)retval1_void;
    for (uint i = 0; i != num_threads; ++i) {
        retval1[i] = 0x23232323;
    }

    hipLaunchKernelGGL(kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                       (ulong)callee, retval0, retval1);

    hipStreamSynchronize(0);

    for (uint ii = 0; ii != num_threads; ++ii) {
        ulong value = retval0[ii];
        if (ii % 71 == 1) {
            ASSERT(value == 0x23232323);
        } else {
            ASSERT(value == ii + 1);
        }
    }
}

int
main(int argc, char **argv)
{
    ASSERT(set_flags(argc, argv));
    test();
    test_passed();
}
