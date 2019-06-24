#include <amd_hostcall.h>
#include <functional>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

#include "common.h"

void
print_things_0(ulong *output, ulong *input)
{
    auto fmt = reinterpret_cast<const char *>(input);
    auto arg0 = input[2];
    auto arg1 = input[3];
    output[0] = fprintf(stdout, fmt, arg0, arg1);
}

__global__ void
kernel0(ulong fptr)
{
    uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    ulong arg0 = fptr;

    const char *str = "(%lu -> %lu)\n";
    ulong arg1 = 0;
    for (int ii = 0; ii != 8; ++ii) {
        arg1 |= (ulong)str[ii] << (8 * ii);
    }
    ulong arg2 = 0;
    for (int ii = 0; ii != 7; ++ii) {
        arg2 |= (ulong)str[ii + 8] << (8 * ii);
    }

    ulong arg3 = 42;
    ulong arg4 = tid;
    ulong arg5 = 0;
    ulong arg6 = 0;
    ulong arg7 = 0;

    __ockl_call_host_function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
}

void
print_things_1(ulong *output, const ulong *input)
{
    auto name = reinterpret_cast<const char *>(input[0]);
    auto tid = input[1];
    output[0] = fprintf(stdout, "kernel: %s; tid: %lu\n", name, tid);
}

__global__ void
kernel1(ulong fptr, ulong name)
{
    uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    ulong arg0 = fptr;
    ulong arg1 = name;
    ulong arg2 = tid;
    ulong arg3 = 0;
    ulong arg4 = 0;
    ulong arg5 = 0;
    ulong arg6 = 0;
    ulong arg7 = 0;

    __ockl_call_host_function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
}

static void
test()
{
    uint num_blocks = 1;
    uint threads_per_block = 1;
    uint num_threads = num_blocks * threads_per_block;

    hipLaunchKernelGGL(kernel0, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                       (ulong)print_things_0);
    hipStreamSynchronize(0);

    const char *name = "kernel1";
    hipLaunchKernelGGL(kernel1, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                       (ulong)print_things_1, (ulong)name);
    hipStreamSynchronize(0);
}

int
main(int argc, char **argv)
{
    ASSERT(set_flags(argc, argv));
    test();
    test_passed();
}
