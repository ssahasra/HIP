#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <iostream>

#include "common.h"

// Global string constants don't work inside device functions, so we
// use a macro to repeat the declaration in host and device contexts.
DECLARE_DATA();

__global__ void
kernel_null(int *retval)
{
    DECLARE_DATA();

    const uint is_last = 1;
    ulong result = 0;

    result = __ockl_printf_begin(0);
    result = __ockl_printf_append_string_n(result, NULL, 42, is_last);
    *retval = result;
}

static void
test_null(int *retval)
{
    *retval = 0x23232323;

    hipLaunchKernelGGL(kernel_null, 1, 1, 0, 0, retval);
    hipStreamSynchronize(0);

    ASSERT(*retval == 0);
}

__global__ void
kernel_empty(int *retval)
{
    DECLARE_DATA();

    const uint is_last = 1;
    ulong result = 0;

    result = __ockl_printf_begin(0);
    result = __ockl_printf_append_string_n(result, "", 2, is_last);
    *retval = result;
}

static void
test_empty(int *retval)
{
    *retval = 0x23232323;

    hipLaunchKernelGGL(kernel_empty, 1, 1, 0, 0, retval);
    hipStreamSynchronize(0);

    ASSERT(*retval == 0);
}

__global__ void
kernel_string(int *retval)
{
    DECLARE_DATA();

    const uint is_last = 1;
    ulong result = 0;

    result = __ockl_printf_begin(0);
    result = __ockl_printf_append_string_n(result, msg_long1, test_strlen(msg_long1) + 1, is_last);
    *retval = result;
}

static void
test_string(int *retval)
{
    *retval = 0x23232323;

    hipLaunchKernelGGL(kernel_string, 1, 1, 0, 0, retval);
    hipStreamSynchronize(0);

    ASSERT(*retval == strlen(msg_long1));
}

__global__ void
kernel_qwords(int *retval)
{
    DECLARE_DATA();
    ulong result = 0;
    const uint is_last = 1;

    result = __ockl_printf_begin(0);
    result = __ockl_printf_append_args(result, 2, qword0, qword1,
                                       0, 0, 0, 0, 0, is_last);
    *retval = result;
}

static void
test_qwords(int *retval)
{
    *retval = 0x23232323;

    hipLaunchKernelGGL(kernel_qwords, 1, 1, 0, 0, retval);
    hipStreamSynchronize(0);

    ASSERT(*retval == qlen - 1);
}

__global__ void
kernel_integers(int *retval)
{
    const char *msg = "Integer: %u.\n";
    const uint is_last = 1;

    ulong desc = __ockl_printf_begin(0);
    desc = __ockl_printf_append_string_n(desc, msg, test_strlen(msg) + 1, 0);
    *retval = __ockl_printf_append_args(desc, 1, (int)42,
                                        0, 0, 0, 0, 0, 0, is_last);
}

static void
test_integers(int *retval)
{
    *retval = 0x23232323;

    hipLaunchKernelGGL(kernel_integers, 1, 1,
                       0, 0, retval);
    hipStreamSynchronize(0);
    ASSERT(*retval == 13);
}

__global__ void
kernel_mixed(int *retval)
{
    const char *msg = "%s Pointer: %p. %s Floating point: %f. %s\n";
    const char *str_begin = "Begin.";
    const char *str_middle = "Middle.";
    const char *str_end = "End.";
    const int *ptr = (const int *)(~0UL - 42);
    const double dbl = 4.2;
    const uint is_last = 1;
    const uint is_not_last = 0;

    ulong desc = __ockl_printf_begin(0);
    // Format string.
    desc = __ockl_printf_append_string_n(desc, msg, test_strlen(msg) + 1, is_not_last);
    // First string argument.
    desc = __ockl_printf_append_string_n(desc, str_begin, test_strlen(str_begin) + 1, is_not_last);
    // Pointer argument.
    desc = __ockl_printf_append_args(desc, 1, (ulong)ptr,
                                     0, 0, 0, 0, 0, 0, is_not_last);
    // Second string argument.
    desc = __ockl_printf_append_string_n(desc, str_middle, test_strlen(str_middle) + 1, is_not_last);
    // Float argument.
    //
    // Technically, this is UB; so is the union trick. The only "not
    // UB" way in C++ is to memcpy the bytes of a double into a
    // ulong. But this works too.
    desc = __ockl_printf_append_args(desc, 1, *(uint64_t*)&dbl,
                                     0, 0, 0, 0, 0, 0, is_not_last);
    // Third string argument.
    *retval = __ockl_printf_append_string_n(desc, str_end, test_strlen(str_end) + 1, is_last);
}

static void
test_mixed(int *retval)
{
    *retval = 0x23232323;

    hipLaunchKernelGGL(kernel_mixed, 1, 1,
                       0, 0, retval);
    hipStreamSynchronize(0);
    ASSERT(*retval == 75);
}

int
main(int argc, char **argv)
{
    ASSERT(set_flags(argc, argv));

    uint num_blocks = 1;
    uint threads_per_block = 1;
    uint num_threads = num_blocks * threads_per_block;

    void *retval_void;
    HIPCHECK(hipHostMalloc(&retval_void, 4 * num_threads));
    auto retval = reinterpret_cast<int*>(retval_void);

    test_null(retval);
    test_empty(retval);
    test_string(retval);
    test_qwords(retval);
    test_integers(retval);
    test_mixed(retval);

    test_passed();
}
