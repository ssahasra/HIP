#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

#include "common.h"

// Global string constants don't work inside device functions, so we
// use a macro to repeat the declaration in host and device contexts.
DECLARE_DATA();

__global__ void
kernel_mixed0(int *retval)
{
    DECLARE_DATA();

    uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    ulong result = 0;

    // Three strings passed as divergent values to the same hostcall.
    const char *msg;
    switch (tid % 3) {
    case 0:
        msg = msg_short;
        break;
    case 1:
        msg = msg_long1;
        break;
    case 2:
        msg = msg_long2;
        break;
    }

    result = __ockl_printf_begin(0);
    result = __ockl_printf_append_string_n(result, msg, test_strlen(msg) + 1, 1);
    retval[tid] = result;
}

static void
test_mixed0(int *retval, uint num_blocks, uint threads_per_block)
{
    uint num_threads = num_blocks * threads_per_block;
    for (uint i = 0; i != num_threads; ++i) {
        retval[i] = 0x23232323;
    }


    hipLaunchKernelGGL(kernel_mixed0, dim3(num_blocks), dim3(threads_per_block),
                       0, 0, retval);
    hipStreamSynchronize(0);

    for (uint ii = 0; ii != num_threads; ++ii) {
        switch (ii % 3) {
        case 0:
            ASSERT(retval[ii] == strlen(msg_short));
            break;
        case 1:
            ASSERT(retval[ii] == strlen(msg_long1));
            break;
        case 2:
            ASSERT(retval[ii] == strlen(msg_long2));
            break;
        }
    }
}

__global__ void
kernel_mixed1(int *retval)
{
    DECLARE_DATA();

    const uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    const uint is_last = 1;
    ulong desc = 0;

    // Three strings passed to divergent hostcalls.
    switch (tid % 3) {
    case 0:
        desc = __ockl_printf_begin(0);
        desc = __ockl_printf_append_string_n(desc, "%s", test_strlen("%s") + 1, 0);
        desc = __ockl_printf_append_string_n(desc, msg_short, test_strlen(msg_short) + 1, is_last);
        break;
    case 1:
        desc = __ockl_printf_begin(0);
        desc = __ockl_printf_append_string_n(desc, "%s", test_strlen("%s") + 1, 0);
        desc = __ockl_printf_append_string_n(desc, msg_long1, test_strlen(msg_long1) + 1, is_last);
        break;
    case 2:
        desc = __ockl_printf_begin(0);
        desc = __ockl_printf_append_string_n(desc, "%s", test_strlen("%s") + 1, 0);
        desc = __ockl_printf_append_string_n(desc, msg_long2, test_strlen(msg_long2) + 1, is_last);
        break;
    }

    retval[tid] = desc;
}

static void
test_mixed1(int *retval, uint num_blocks, uint threads_per_block)
{
    uint num_threads = num_blocks * threads_per_block;
    for (uint i = 0; i != num_threads; ++i) {
        retval[i] = 0x23232323;
    }

    hipLaunchKernelGGL(kernel_mixed1, dim3(num_blocks), dim3(threads_per_block),
                       0, 0, retval);
    hipStreamSynchronize(0);

    for (uint ii = 0; ii != num_threads; ++ii) {
        switch (ii % 3) {
        case 0:
            ASSERT(retval[ii] == strlen(msg_short));
            break;
        case 1:
            ASSERT(retval[ii] == strlen(msg_long1));
            break;
        case 2:
            ASSERT(retval[ii] == strlen(msg_long2));
            break;
        }
    }
}

__global__ void
kernel_mixed2(int *retval)
{
    DECLARE_DATA();

    const uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    const uint is_last = 1;
    ulong desc = 0;

    // Three different strings. All workitems print all three, but
    // in different orders.
    const char *msg[] = {msg_short, msg_long1, msg_long2};
    retval[tid] = 0;

    desc = __ockl_printf_begin(0);
    desc = __ockl_printf_append_string_n(desc, "%s%s%s", test_strlen("%s%s%s") + 1, 0);
    desc = __ockl_printf_append_string_n(desc, msg[tid % 3], test_strlen(msg[tid % 3]) + 1, 0);
    desc = __ockl_printf_append_string_n(desc, msg[(tid + 1) % 3], test_strlen(msg[(tid + 1) % 3]) + 1, 0);
    retval[tid] = __ockl_printf_append_string_n(desc, msg[(tid + 2) % 3], test_strlen(msg[(tid + 2) % 3]) + 1, is_last);
}

static void
test_mixed2(int *retval, uint num_blocks, uint threads_per_block)
{
    uint num_threads = num_blocks * threads_per_block;
    for (uint i = 0; i != num_threads; ++i) {
        retval[i] = 0x23232323;
    }


    hipLaunchKernelGGL(kernel_mixed2, dim3(num_blocks), dim3(threads_per_block),
                       0, 0, retval);
    hipStreamSynchronize(0);

    for (uint ii = 0; ii != num_threads; ++ii) {
        ASSERT(retval[ii] == strlen(msg_short) + strlen(msg_long1) + strlen(msg_long2));
    }
}

__global__ void
kernel_mixed3(int *retval)
{
    DECLARE_DATA();

    const uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    ulong result = 0;
    ulong desc = 0;
    const uint is_last = 1;

    desc = __ockl_printf_begin(0);
    result += __ockl_printf_append_string_n(desc, msg_long1, test_strlen(msg_long1) + 1, is_last);

    if (tid % 3 == 0) {
        desc = __ockl_printf_begin(0);
        result += __ockl_printf_append_args(desc, 2, qword0, qword1,
                                            0, 0, 0, 0, 0, is_last);
    }

    desc = __ockl_printf_begin(0);
    result += __ockl_printf_append_string_n(desc, msg_long2, test_strlen(msg_long2) + 1, is_last);

    retval[tid] = result;
}

static void
test_mixed3(int *retval, uint num_blocks, uint threads_per_block)
{
    uint num_threads = num_blocks * threads_per_block;
    for (uint i = 0; i != num_threads; ++i) {
        retval[i] = 0x23232323;
    }


    hipLaunchKernelGGL(kernel_mixed3, dim3(num_blocks), dim3(threads_per_block),
                       0, 0, retval);
    hipStreamSynchronize(0);

    for (uint ii = 0; ii != num_threads; ++ii) {
        if (ii % 3 == 0) {
            ASSERT(retval[ii] == strlen(msg_long1) + qlen - 1 + strlen(msg_long2));
        } else {
            ASSERT(retval[ii] == strlen(msg_long1) + strlen(msg_long2));
        }
    }
}

int
main(int argc, char **argv)
{
    ASSERT(set_flags(argc, argv));

    uint num_blocks = 1;
    uint threads_per_block = 64;
    uint num_threads = num_blocks * threads_per_block;

    void *retval_void;
    HIPCHECK(hipHostMalloc(&retval_void, 4 * num_threads));
    auto retval = reinterpret_cast<int*>(retval_void);

    test_mixed0(retval, num_blocks, threads_per_block);
    test_mixed1(retval, num_blocks, threads_per_block);
    test_mixed2(retval, num_blocks, threads_per_block);
    test_mixed3(retval, num_blocks, threads_per_block);

    test_passed();
}
