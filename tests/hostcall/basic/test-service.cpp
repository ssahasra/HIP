#include <amd_hostcall.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

#include "common.h"

static int
handler(void *state, uint32_t service, ulong *payload)
{
    *payload = *payload + 1;
    return 0;
}

__global__ void
kernel(ulong *retval)
{
    uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    ulong arg0 = tid;
    ulong arg1 = 0;
    ulong arg2 = 0;
    ulong arg3 = 0;
    ulong arg4 = 0;
    ulong arg5 = 0;
    ulong arg6 = 0;
    ulong arg7 = 0;

    long2 result = {0, 0};
    if (tid % 71 != 1) {
        result.data = __ockl_hostcall_preview(SERVICE_TEST, arg0, arg1, arg2,
                                              arg3, arg4, arg5, arg6, arg7);
        retval[tid] = result.x;
    }
}

static void
test()
{
    uint num_blocks = 5;
    uint threads_per_block = 1000;
    uint num_threads = num_blocks * threads_per_block;

    void *retval_void;
    HIPCHECK(hipHostMalloc(&retval_void, 8 * num_threads));
    uint64_t *retval = (uint64_t *)retval_void;
    for (uint i = 0; i != num_threads; ++i) {
        retval[i] = 0x23232323;
    }

    amd_hostcall_register_service(SERVICE_TEST, handler, nullptr);

    hipLaunchKernelGGL(kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                       retval);

    hipStreamSynchronize(0);

    for (uint ii = 0; ii != num_threads; ++ii) {
        ulong value = retval[ii];
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
