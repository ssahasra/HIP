#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

#include "common.h"

__global__ void
alt_forms()
{
    printf("%#o\n", 042);
    printf("%#x\n", 0x42);
    printf("%#X\n", 0x42);
    printf("%#08x\n", 0x42);
    printf("%#f\n", -123.456);
    printf("%#F\n", 123.456);
    printf("%#e\n", 123.456);
    printf("%#E\n", -123.456);
    printf("%#g\n", -123.456);
    printf("%#G\n", 123.456);
    printf("%#a\n", 123.456);
    printf("%#A\n", -123.456);
    printf("%#.8x\n", 0x42);
    printf("%#16.8x\n", 0x42);
    printf("%-#16.8x\n", 0x42);
}

int
main(int argc, char **argv)
{
    ASSERT(set_flags(argc, argv));
    hipLaunchKernelGGL(alt_forms, dim3(1), dim3(1), 0, 0);
    hipStreamSynchronize(0);
    test_passed();
}
