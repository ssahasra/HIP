#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

#include "common.h"

__global__ void
width_precision()
{
    printf("%16d\n", 42);
    printf("%.8d\n", 42);
    printf("%16.5d\n", -42);
    printf("%.8x\n", 0x42);
    printf("%.8o\n", 042);
    printf("%16.8e\n", 12345.67891);
    printf("%16.8f\n", -12345.67891);
    printf("%16.8g\n", 12345.67891);
    printf("%8.4e\n", -12345.67891);
    printf("%8.4f\n", 12345.67891);
    printf("%8.4g\n", 12345.67891);
    printf("%4.2f\n", 12345.67891);
    printf("%.1f\n", 12345.67891);
    printf("%.5s\n", "helloxyz");
}

int
main(int argc, char **argv)
{
    ASSERT(set_flags(argc, argv));
    hipLaunchKernelGGL(width_precision, dim3(1), dim3(1), 0, 0);
    hipStreamSynchronize(0);
    test_passed();
}
