#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

#include "common.h"

__global__ void
flags()
{
    printf("%08d\n", 42);
    printf("%08i\n", -42);
    printf("%08u\n", 42);
    printf("%08g\n", 123.456);
    printf("%0+8d\n", 42);
    printf("%+d\n", -42);
    printf("%+08d\n", 42);
    printf("%-8s\n", "xyzzy");
    printf("% i\n", -42);
    printf("%-16.8d\n", 42);
    printf("%16.8d\n", 42);
}

int
main(int argc, char **argv)
{
    ASSERT(set_flags(argc, argv));
    hipLaunchKernelGGL(flags, dim3(1), dim3(1), 0, 0);
    hipStreamSynchronize(0);
    test_passed();
}
